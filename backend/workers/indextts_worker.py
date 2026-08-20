"""JSON-lines subprocess worker for the pinned IndexTTS 2.5 runtime."""

from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

_RESPONSE_PREFIX = "VOICEBOX_INDEXTTS_JSON:"
_EMOTION_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)


def _respond(request_id: int | None, *, ok: bool, **payload) -> None:
    response = {"request_id": request_id, "ok": ok, **payload}
    print(_RESPONSE_PREFIX + json.dumps(response), flush=True)


def _ordered_emotion_vector(value):
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) != len(_EMOTION_ORDER):
            raise ValueError("Emotion vector must contain exactly eight values")
        return [float(item) for item in value]
    if not isinstance(value, dict):
        raise ValueError("Emotion vector must be an object")
    return [float(value[name]) for name in _EMOTION_ORDER]


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prefer_bundled_unidic() -> None:
    """Keep fugashi on unidic-lite when the host also has empty unidic metadata."""
    import unidic_lite

    sys.modules["unidic"] = unidic_lite


def main() -> None:
    model = None
    for raw_line in sys.stdin:
        request_id = None
        try:
            request = json.loads(raw_line)
            request_id = request.get("request_id")
            action = request.get("action")
            if action == "load":
                import torch

                _prefer_bundled_unidic()
                from indextts.infer_v2_5 import IndexTTS2

                model_dir = Path(request["model_dir"]).resolve()
                model = IndexTTS2(
                    cfg_path=str(model_dir / "config.yaml"),
                    model_dir=str(model_dir),
                    use_bf16=torch.cuda.is_available(),
                    device="cuda:0" if torch.cuda.is_available() else "cpu",
                    use_cuda_kernel=False,
                    use_deepspeed=False,
                    use_accel=False,
                    use_torch_compile=False,
                    use_qwen_emo=True,
                )
                _respond(request_id, ok=True, device=str(model.device))
                continue

            if action == "generate":
                if model is None:
                    raise RuntimeError("IndexTTS model is not loaded")
                _seed_everything(request.get("seed"))
                mode = request.get("emotion_mode", "natural")
                use_emo_text = mode in {"auto", "instruction"}
                emo_text = request.get("emotion_text") if mode == "instruction" else None
                emo_vector = _ordered_emotion_vector(request.get("emotion_vector")) if mode == "vector" else None
                emo_audio = request.get("emotion_audio") if mode == "audio" else None
                output_path = request["output_path"]
                result = model.infer(
                    spk_audio_prompt=request["speaker_audio"],
                    text=request["text"],
                    output_path=output_path,
                    lang=request["language"],
                    emo_audio_prompt=emo_audio,
                    emo_alpha=float(request.get("emo_alpha", 0.6)),
                    emo_vector=emo_vector,
                    use_emo_text=use_emo_text,
                    emo_text=emo_text,
                    use_random=bool(request.get("use_random", False)),
                    duration_factor=float(request.get("duration_factor", 1.0)),
                    verbose=False,
                )
                if result is None or not Path(output_path).is_file():
                    raise RuntimeError("IndexTTS inference returned no audio")
                _respond(request_id, ok=True, output_path=output_path)
                continue

            if action == "ping":
                _respond(request_id, ok=True, loaded=model is not None)
                continue
            raise ValueError(f"Unknown worker action: {action}")
        except Exception as exc:
            traceback.print_exc()
            _respond(request_id, ok=False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
