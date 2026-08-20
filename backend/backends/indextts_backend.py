"""IndexTTS 2.5 backend with a dependency-isolated inference worker."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..services.generation_assets import resolve_emotion_audio_asset
from ..utils.audio import load_audio
from .base import combine_voice_prompts as _combine_voice_prompts, model_load_progress

logger = logging.getLogger(__name__)

INDEXTTS_SOURCE_COMMIT = "ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c"
INDEXTTS_MODEL_REPO = "IndexTeam/IndexTTS-2.5"
INDEXTTS_MODEL_REVISION = "c39ce5ba981572cb187443877ff559dfb246ce63"
INDEXTTS_W2V_REPO = "facebook/w2v-bert-2.0"
INDEXTTS_W2V_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
INDEXTTS_CAMPPLUS_REPO = "funasr/campplus"
INDEXTTS_CAMPPLUS_REVISION = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"
INDEXTTS_BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
INDEXTTS_BIGVGAN_REVISION = "633ff708ed5b74903e86ff1298cf4a98e921c513"

_RESPONSE_PREFIX = "VOICEBOX_INDEXTTS_JSON:"

# Exact sizes are from the pinned Hugging Face revisions.  Checking them makes
# "Downloaded" mean ready for offline generation, including all auxiliary
# checkpoints, instead of merely finding one primary weight file.
INDEXTTS_REQUIRED_FILES: dict[str, int] = {
    "LICENSE": 10_553,
    "README.md": 4_272,
    "codec.pth": 607_290_935,
    "config.yaml": 2_860,
    "feat1.pt": 57_170,
    "feat2.pt": 374_866,
    "gpt.pth": 3_259_599_833,
    "multilingual_zh_ja_yue_char_del.tiktoken": 907_395,
    "qwen0.6bemo4-merge/Modelfile": 360,
    "qwen0.6bemo4-merge/added_tokens.json": 707,
    "qwen0.6bemo4-merge/chat_template.jinja": 550,
    "qwen0.6bemo4-merge/config.json": 727,
    "qwen0.6bemo4-merge/generation_config.json": 117,
    "qwen0.6bemo4-merge/merges.txt": 1_671_853,
    "qwen0.6bemo4-merge/model.safetensors": 1_192_135_096,
    "qwen0.6bemo4-merge/special_tokens_map.json": 616,
    "qwen0.6bemo4-merge/tokenizer.json": 11_422_654,
    "qwen0.6bemo4-merge/tokenizer_config.json": 5_433,
    "qwen0.6bemo4-merge/vocab.json": 2_776_833,
    "s2mel.pth": 414_908_601,
    "wav2vec2bert_stats.pt": 9_343,
    "hf_cache/w2v-bert-2.0/config.json": 1_874,
    "hf_cache/w2v-bert-2.0/model.safetensors": 2_322_063_736,
    "hf_cache/w2v-bert-2.0/preprocessor_config.json": 275,
    "hf_cache/campplus_cn_common.bin": 28_036_335,
    "hf_cache/bigvgan/config.json": 1_405,
    "hf_cache/bigvgan/bigvgan_generator.pt": 449_228_171,
}
INDEXTTS_EXPECTED_BYTES = sum(INDEXTTS_REQUIRED_FILES.values())
INDEXTTS_PRIMARY_FILES = [path for path in INDEXTTS_REQUIRED_FILES if not path.startswith("hf_cache/")]


class IndexTTSBackend:
    """Voicebox TTSBackend implementation for IndexTTS 2.5."""

    def __init__(self) -> None:
        self.model_size = "default"
        self._process: asyncio.subprocess.Process | None = None
        self._loaded = False
        self._load_lock = asyncio.Lock()
        self._ipc_lock = asyncio.Lock()
        self._request_id = 0
        self._worker_wait_task: asyncio.Task | None = None

    def _get_model_dir(self) -> Path:
        override = os.environ.get("VOICEBOX_INDEXTTS_MODEL_DIR")
        if override:
            return Path(override).expanduser().resolve()
        from huggingface_hub import constants as hf_constants

        return (Path(hf_constants.HF_HUB_CACHE) / "voicebox-indextts-2.5").resolve()

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(self._get_model_dir())

    def _is_model_cached(self, model_size: str = "default") -> bool:
        model_dir = self._get_model_dir()
        if not model_dir.is_dir() or any(model_dir.rglob("*.incomplete")):
            return False
        for relative_path, expected_size in INDEXTTS_REQUIRED_FILES.items():
            path = model_dir / relative_path
            try:
                if not path.is_file() or path.stat().st_size != expected_size:
                    return False
            except OSError:
                return False
        return True

    def get_downloaded_size_mb(self) -> float | None:
        model_dir = self._get_model_dir()
        if not model_dir.exists():
            return None
        total = sum(
            path.stat().st_size
            for path in model_dir.rglob("*")
            if path.is_file() and not path.name.endswith(".incomplete")
        )
        return total / (1024 * 1024)

    async def download_model(self) -> None:
        await asyncio.to_thread(self._download_model_sync)

    def _download_model_sync(self) -> None:
        if self._is_model_cached():
            return
        from huggingface_hub import snapshot_download

        model_dir = self._get_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        with model_load_progress("indextts-2.5", False, filter_non_downloads=False):
            snapshot_download(
                repo_id=INDEXTTS_MODEL_REPO,
                revision=INDEXTTS_MODEL_REVISION,
                local_dir=model_dir,
                allow_patterns=INDEXTTS_PRIMARY_FILES,
            )
            snapshot_download(
                repo_id=INDEXTTS_W2V_REPO,
                revision=INDEXTTS_W2V_REVISION,
                local_dir=model_dir / "hf_cache" / "w2v-bert-2.0",
                allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
            )
            snapshot_download(
                repo_id=INDEXTTS_CAMPPLUS_REPO,
                revision=INDEXTTS_CAMPPLUS_REVISION,
                local_dir=model_dir / "hf_cache",
                allow_patterns=["campplus_cn_common.bin"],
            )
            snapshot_download(
                repo_id=INDEXTTS_BIGVGAN_REPO,
                revision=INDEXTTS_BIGVGAN_REVISION,
                local_dir=model_dir / "hf_cache" / "bigvgan",
                allow_patterns=["config.json", "bigvgan_generator.pt"],
            )
            if not self._is_model_cached():
                raise RuntimeError("IndexTTS download finished but one or more required files failed validation")

    def delete_model_files(self) -> None:
        self.unload_model()
        model_dir = self._get_model_dir()
        if not model_dir.exists():
            raise FileNotFoundError("IndexTTS 2.5 is not downloaded")
        shutil.rmtree(model_dir)

    def is_loaded(self) -> bool:
        return bool(self._loaded and self._process is not None and self._process.returncode is None)

    async def load_model(self, model_size: str = "default") -> None:
        if self.is_loaded():
            return
        async with self._load_lock:
            if self.is_loaded():
                return
            if not self._is_model_cached():
                raise RuntimeError("IndexTTS 2.5 is not downloaded. Download it from the Models tab first.")
            await self._start_worker()
            try:
                await self._send_command(
                    {
                        "action": "load",
                        "model_dir": str(self._get_model_dir()),
                    }
                )
            except BaseException:
                self._terminate_worker()
                raise
            self._loaded = True

    async def _start_worker(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        configured = os.environ.get("VOICEBOX_INDEXTTS_WORKER_PYTHON")
        bundled = Path("/opt/indextts-venv/bin/python")
        python_executable = configured or (str(bundled) if bundled.exists() else sys.executable)
        worker_env = os.environ.copy()
        worker_env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        self._process = await asyncio.create_subprocess_exec(
            python_executable,
            "-m",
            "backend.workers.indextts_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=worker_env,
        )

    async def _send_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._ipc_lock:
            process = self._process
            if process is None or process.returncode is not None or process.stdin is None or process.stdout is None:
                raise RuntimeError("IndexTTS worker is not running")
            self._request_id += 1
            request_id = self._request_id
            payload = {**payload, "request_id": request_id}
            process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await process.stdin.drain()

            recent_logs: list[str] = []
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    detail = "\n".join(recent_logs[-10:])
                    raise RuntimeError(f"IndexTTS worker exited unexpectedly{': ' + detail if detail else ''}")
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                if not line.startswith(_RESPONSE_PREFIX):
                    recent_logs.append(line)
                    if line:
                        logger.info("[IndexTTS worker] %s", line)
                    continue
                try:
                    response = json.loads(line[len(_RESPONSE_PREFIX) :])
                except json.JSONDecodeError:
                    logger.warning("Ignoring malformed IndexTTS worker response")
                    continue
                if response.get("request_id") != request_id:
                    logger.warning("Ignoring stale IndexTTS worker response %s", response.get("request_id"))
                    continue
                if not response.get("ok"):
                    raise RuntimeError(response.get("error") or "IndexTTS worker request failed")
                return response

    def _terminate_worker(self) -> None:
        process = self._process
        self._process = None
        self._loaded = False
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            loop = asyncio.get_running_loop()
            self._worker_wait_task = loop.create_task(process.wait())
        except RuntimeError:
            pass

    def unload_model(self) -> None:
        self._terminate_worker()

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        return {"ref_audio": str(audio_path), "ref_text": reference_text}, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=24_000)

    @staticmethod
    def _normalize_options(engine_options: dict[str, Any] | None) -> dict[str, Any]:
        options = (engine_options or {}).get("indextts", {})
        return options if isinstance(options, dict) else {}

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, int]:
        await self.load_model()
        reference_audio = Path(str(voice_prompt.get("ref_audio", "")))
        if not reference_audio.is_file():
            raise ValueError("IndexTTS requires a valid speaker reference audio sample")

        options = self._normalize_options(engine_options)
        mode = options.get("emotion_mode", "natural")
        emotion_audio: str | None = None
        if mode == "audio":
            asset_id = options.get("emotion_audio_asset_id")
            if not asset_id:
                raise ValueError("Emotion reference audio mode requires an uploaded audio asset")
            emotion_audio = str(resolve_emotion_audio_asset(str(asset_id)))

        temp_dir = config.get_cache_dir() / "indextts-output"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as handle:
            output_path = Path(handle.name)
        output_path.unlink(missing_ok=True)

        try:
            await self._send_command(
                {
                    "action": "generate",
                    "text": text,
                    "language": language,
                    "seed": seed,
                    "speaker_audio": str(reference_audio),
                    "output_path": str(output_path),
                    "emotion_mode": mode,
                    "emotion_text": options.get("emotion_text"),
                    "emotion_vector": options.get("emotion_vector"),
                    "emotion_audio": emotion_audio,
                    "emo_alpha": options.get("emo_alpha", 0.6),
                    "use_random": options.get("use_random", False),
                    "duration_factor": options.get("duration_factor", 1.0),
                }
            )
            if not output_path.is_file():
                raise RuntimeError("IndexTTS did not produce an output WAV")
            audio, sample_rate = await asyncio.to_thread(load_audio, str(output_path))
            return np.asarray(audio, dtype=np.float32), sample_rate
        except asyncio.CancelledError:
            self._terminate_worker()
            raise
        finally:
            output_path.unlink(missing_ok=True)
