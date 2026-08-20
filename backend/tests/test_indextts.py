"""Focused unit coverage for the IndexTTS 2.5 integration.

All upstream inference and Hugging Face calls are mocked.  Ordinary pytest
must never need the multi-gigabyte model download.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, text

from backend import models
from backend.backends import (
    _tts_backends,
    get_model_config,
    get_tts_backend_for_engine,
    get_tts_model_configs,
    indextts_backend as indextts_module,
)
from backend.backends.indextts_backend import IndexTTSBackend
from backend.database import (
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    ProfileSample as DBProfileSample,
    VoiceProfile as DBVoiceProfile,
)
from backend.database.migrations import run_migrations
from backend.routes import generations
from backend.services import export_import, generation_assets, profiles
from backend.utils.advanced_tts import generate_with_advanced_controls
from backend.workers.indextts_worker import _ordered_emotion_vector


def _index_form_options(**overrides):
    values = {
        "emotion_mode": "natural",
        "emo_alpha": 0.6,
        "use_random": False,
        "duration_factor": 1.0,
    }
    values.update(overrides)
    return values


def test_indextts_model_config_and_factory_registration(monkeypatch):
    config = get_model_config("indextts-2.5")

    assert config is not None
    assert config.engine == "indextts"
    assert config.hf_repo_id == "IndexTeam/IndexTTS-2.5"
    assert config.display_name == "IndexTTS 2.5 (Expressive Clone)"
    assert config.languages == ["zh", "en", "ja", "es", "ar"]
    assert config.backend_managed is True
    assert config.size_mb == 7907
    assert [cfg.model_name for cfg in get_tts_model_configs()].count("indextts-2.5") == 1

    monkeypatch.delitem(_tts_backends, "indextts", raising=False)
    assert isinstance(get_tts_backend_for_engine("indextts"), IndexTTSBackend)


def test_indextts_profile_compatibility_rules():
    cloned = SimpleNamespace(id="clone", voice_type="cloned")
    profiles.validate_profile_engine(cloned, "indextts")
    assert (
        profiles._validate_profile_fields(
            voice_type="cloned",
            preset_engine=None,
            preset_voice_id=None,
            design_prompt=None,
            default_engine="indextts",
        )
        is None
    )

    preset = SimpleNamespace(
        id="preset",
        voice_type="preset",
        preset_engine="kokoro",
        preset_voice_id="af_heart",
    )
    with pytest.raises(ValueError, match="only supports engine 'kokoro'"):
        profiles.validate_profile_engine(preset, "indextts")

    designed = SimpleNamespace(
        id="designed",
        voice_type="designed",
        design_prompt="A warm narrator",
    )
    with pytest.raises(ValueError, match="requires a cloned profile"):
        profiles.validate_profile_engine(designed, "indextts")


def test_options_validate_modes_ranges_and_vector_order():
    vector = models.IndexTTSEmotionVector(
        happy=0.01,
        angry=0.02,
        sad=0.03,
        afraid=0.04,
        disgusted=0.05,
        melancholic=0.06,
        surprised=0.07,
        calm=0.08,
    )
    assert vector.as_ordered_list() == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    assert _ordered_emotion_vector(vector.model_dump()) == vector.as_ordered_list()

    models.IndexTTSOptions(emotion_mode="natural")
    models.IndexTTSOptions(emotion_mode="auto")
    models.IndexTTSOptions(emotion_mode="instruction", emotion_text="calm and grave")
    models.IndexTTSOptions(emotion_mode="vector", emotion_vector=vector)
    models.IndexTTSOptions(emotion_mode="audio", emotion_audio_asset_id=uuid.uuid4())

    invalid_options = [
        {"emotion_mode": "instruction"},
        {"emotion_mode": "natural", "emotion_text": "conflict"},
        {"emotion_mode": "vector"},
        {"emotion_mode": "audio"},
        {"emo_alpha": -0.01},
        {"emo_alpha": 1.01},
        {"duration_factor": 0.49},
        {"duration_factor": 2.01},
        {"emotion_mode": "natural", "use_random": True},
        {"emotion_mode": "audio", "emotion_audio_asset_id": uuid.uuid4(), "use_random": True},
    ]
    for values in invalid_options:
        with pytest.raises(ValidationError):
            models.IndexTTSOptions(**values)

    with pytest.raises(ValidationError, match=r"sum to 0\.8 or less"):
        models.IndexTTSEmotionVector(happy=0.5, calm=0.31)
    with pytest.raises(ValidationError):
        models.IndexTTSEmotionVector(angry=1.01)


def test_generation_request_rejects_indextts_options_for_other_engines():
    engine_options = models.EngineOptions(indextts=models.IndexTTSOptions())
    with pytest.raises(ValidationError, match="only be used with engine='indextts'"):
        models.GenerationRequest(
            profile_id="profile",
            text="Hello",
            language="en",
            engine="qwen",
            engine_options=engine_options,
        )

    request = models.GenerationRequest(
        profile_id="profile",
        text="Hello",
        language="en",
        engine="indextts",
        engine_options=engine_options,
    )
    serialized = generations._serialize_engine_options(request, "indextts")
    assert serialized == {"indextts": _index_form_options()}


def test_model_readiness_requires_every_exact_file_and_no_incomplete(tmp_path, monkeypatch):
    backend = IndexTTSBackend()
    monkeypatch.setattr(backend, "_get_model_dir", lambda: tmp_path)
    monkeypatch.setattr(
        indextts_module,
        "INDEXTTS_REQUIRED_FILES",
        {"config.yaml": 3, "hf_cache/aux/model.bin": 4},
    )

    (tmp_path / "config.yaml").write_bytes(b"cfg")
    assert backend._is_model_cached() is False
    aux = tmp_path / "hf_cache" / "aux" / "model.bin"
    aux.parent.mkdir(parents=True)
    aux.write_bytes(b"data")
    assert backend._is_model_cached() is True

    aux.write_bytes(b"bad")
    assert backend._is_model_cached() is False
    aux.write_bytes(b"data")
    (tmp_path / "download.incomplete").write_bytes(b"")
    assert backend._is_model_cached() is False


def test_download_invokes_all_pinned_repositories(tmp_path, monkeypatch):
    backend = IndexTTSBackend()
    monkeypatch.setattr(backend, "_get_model_dir", lambda: tmp_path)
    required = {
        "primary.bin": 1,
        "hf_cache/w2v-bert-2.0/model.safetensors": 2,
        "hf_cache/campplus_cn_common.bin": 3,
        "hf_cache/bigvgan/bigvgan_generator.pt": 4,
    }
    monkeypatch.setattr(indextts_module, "INDEXTTS_REQUIRED_FILES", required)
    monkeypatch.setattr(indextts_module, "INDEXTTS_PRIMARY_FILES", ["primary.bin"])
    monkeypatch.setattr(
        indextts_module,
        "model_load_progress",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        repo_id = kwargs["repo_id"]
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        if repo_id == indextts_module.INDEXTTS_MODEL_REPO:
            (local_dir / "primary.bin").write_bytes(b"p")
        elif repo_id == indextts_module.INDEXTTS_W2V_REPO:
            (local_dir / "model.safetensors").write_bytes(b"wv")
        elif repo_id == indextts_module.INDEXTTS_CAMPPLUS_REPO:
            (local_dir / "campplus_cn_common.bin").write_bytes(b"cmp")
        else:
            (local_dir / "bigvgan_generator.pt").write_bytes(b"vgan")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    backend._download_model_sync()

    assert [(call["repo_id"], call["revision"]) for call in calls] == [
        (indextts_module.INDEXTTS_MODEL_REPO, indextts_module.INDEXTTS_MODEL_REVISION),
        (indextts_module.INDEXTTS_W2V_REPO, indextts_module.INDEXTTS_W2V_REVISION),
        (indextts_module.INDEXTTS_CAMPPLUS_REPO, indextts_module.INDEXTTS_CAMPPLUS_REVISION),
        (indextts_module.INDEXTTS_BIGVGAN_REPO, indextts_module.INDEXTTS_BIGVGAN_REVISION),
    ]
    assert backend._is_model_cached() is True


class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_load_and_unload_use_isolated_worker(monkeypatch):
    backend = IndexTTSBackend()
    process = _FakeProcess()
    commands = []

    monkeypatch.setattr(backend, "_is_model_cached", lambda *_args: True)

    async def fake_start():
        backend._process = process

    async def fake_send(payload):
        commands.append(payload)
        return {"ok": True}

    monkeypatch.setattr(backend, "_start_worker", fake_start)
    monkeypatch.setattr(backend, "_send_command", fake_send)
    await backend.load_model()

    assert backend.is_loaded() is True
    assert commands == [{"action": "load", "model_dir": str(backend._get_model_dir())}]
    backend.unload_model()
    assert process.terminated is True
    assert backend.is_loaded() is False


@pytest.mark.asyncio
async def test_generation_forwards_seed_text_vector_audio_and_cleans_temp_file(tmp_path, monkeypatch):
    backend = IndexTTSBackend()
    speaker = tmp_path / "speaker.wav"
    speaker.write_bytes(b"speaker")
    emotion = tmp_path / "emotion.wav"
    emotion.write_bytes(b"emotion")
    output_paths = []
    calls = []

    async def already_loaded(*_args, **_kwargs):
        return None

    async def fake_send(payload):
        calls.append(payload)
        output = Path(payload["output_path"])
        output_paths.append(output)
        output.write_bytes(b"wav")
        return {"ok": True}

    monkeypatch.setattr(backend, "load_model", already_loaded)
    monkeypatch.setattr(backend, "_send_command", fake_send)
    monkeypatch.setattr(indextts_module.config, "get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(indextts_module, "load_audio", lambda _path: (np.ones(12, dtype=np.float32), 22_050))
    monkeypatch.setattr(indextts_module, "resolve_emotion_audio_asset", lambda _asset_id: emotion)

    vector_options = {
        "indextts": _index_form_options(
            emotion_mode="vector",
            emotion_vector={
                "happy": 0.0,
                "angry": 0.0,
                "sad": 0.1,
                "afraid": 0.0,
                "disgusted": 0.0,
                "melancholic": 0.0,
                "surprised": 0.0,
                "calm": 0.5,
            },
            emo_alpha=0.55,
            use_random=True,
            duration_factor=1.2,
        )
    }
    audio, sample_rate = await backend.generate(
        "Measured delivery",
        {"ref_audio": str(speaker)},
        language="en",
        seed=1234,
        engine_options=vector_options,
    )
    assert audio.shape == (12,)
    assert sample_rate == 22_050
    assert calls[0]["seed"] == 1234
    assert calls[0]["emotion_vector"] == vector_options["indextts"]["emotion_vector"]
    assert calls[0]["duration_factor"] == 1.2
    assert calls[0]["use_random"] is True
    assert output_paths[0].exists() is False

    asset_id = str(uuid.uuid4())
    await backend.generate(
        "Emotion audio",
        {"ref_audio": str(speaker)},
        engine_options={
            "indextts": _index_form_options(
                emotion_mode="audio",
                emotion_audio_asset_id=asset_id,
            )
        },
    )
    assert calls[1]["emotion_audio"] == str(emotion)
    assert output_paths[1].exists() is False

    await backend.generate(
        "Instruction",
        {"ref_audio": str(speaker)},
        engine_options={
            "indextts": _index_form_options(
                emotion_mode="instruction",
                emotion_text="calm, grave, measured",
            )
        },
    )
    assert calls[2]["emotion_text"] == "calm, grave, measured"


@pytest.mark.asyncio
async def test_generation_cleans_temp_output_after_worker_failure(tmp_path, monkeypatch):
    backend = IndexTTSBackend()
    speaker = tmp_path / "speaker.wav"
    speaker.write_bytes(b"speaker")
    captured_output = None

    async def already_loaded(*_args, **_kwargs):
        return None

    async def fail_after_output(payload):
        nonlocal captured_output
        captured_output = Path(payload["output_path"])
        captured_output.write_bytes(b"partial")
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(backend, "load_model", already_loaded)
    monkeypatch.setattr(backend, "_send_command", fail_after_output)
    monkeypatch.setattr(indextts_module.config, "get_cache_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="upstream failed"):
        await backend.generate("Hello", {"ref_audio": str(speaker)})
    assert captured_output is not None
    assert captured_output.exists() is False


@pytest.mark.asyncio
async def test_profile_chooses_longest_useful_reference_with_stable_tie_break(tmp_path, monkeypatch):
    profile = SimpleNamespace(id="profile", voice_type="cloned")
    sample_a = SimpleNamespace(id="a", audio_path="a.wav", reference_text="A")
    sample_b = SimpleNamespace(id="b", audio_path="b.wav", reference_text="B")
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"audio")

    class FakeQuery:
        def __init__(self, value):
            self.value = value

        def filter_by(self, **_kwargs):
            return self

        def first(self):
            return self.value

        def all(self):
            return self.value

    class FakeDB:
        def query(self, model):
            if model is DBVoiceProfile:
                return FakeQuery(profile)
            if model is DBProfileSample:
                return FakeQuery([sample_b, sample_a])
            raise AssertionError(model)

    class FakeBackend:
        async def create_voice_prompt(self, audio_path, reference_text, use_cache=True):
            return {"ref_audio": audio_path, "ref_text": reference_text}, False

    monkeypatch.setattr(profiles.config, "resolve_storage_path", lambda value: tmp_path / value)
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: FakeBackend())
    monkeypatch.setattr("soundfile.info", lambda _path: SimpleNamespace(duration=20.0))

    prompt = await profiles.create_voice_prompt_for_profile("profile", FakeDB(), engine="indextts")
    assert prompt == {"ref_audio": str(tmp_path / "a.wav"), "ref_text": "A"}


@pytest.mark.asyncio
async def test_pause_spans_keep_same_indextts_options():
    class FakeBackend:
        def __init__(self):
            self.calls = []

        async def generate(
            self,
            text,
            voice_prompt,
            language="en",
            seed=None,
            instruct=None,
            engine_options=None,
        ):
            self.calls.append((text, seed, engine_options))
            return np.ones(100, dtype=np.float32), 1000

    backend = FakeBackend()
    options = {"indextts": _index_form_options(emotion_mode="auto")}
    audio, sample_rate = await generate_with_advanced_controls(
        backend,
        "Before. [pause 500] After.",
        {},
        seed=42,
        engine_options=options,
    )

    assert sample_rate == 1000
    assert np.all(audio[100:600] == 0)
    assert [(seed, passed) for _text, seed, passed in backend.calls] == [(42, options), (42, options)]


class _FakeDB:
    def __init__(self, generation):
        self.generation = generation

    def query(self, model):
        assert model is DBGeneration
        return self

    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return self.generation

    def commit(self):
        return None

    def refresh(self, _value):
        return None


class _FakeTaskManager:
    def start_generation(self, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_retry_and_regenerate_preserve_engine_options(monkeypatch):
    options = {
        "indextts": _index_form_options(
            emotion_mode="instruction",
            emotion_text="restrained concern",
            duration_factor=1.2,
        )
    }
    generation = SimpleNamespace(
        id="generation",
        profile_id="profile",
        text="Text",
        language="en",
        audio_path="",
        duration=0,
        seed=77,
        instruct=None,
        engine="indextts",
        model_size=None,
        engine_options=options,
        status="failed",
        error="failed",
        is_favorited=False,
        source="manual",
        created_at=datetime.utcnow(),
    )
    captured = []
    monkeypatch.setattr(generations, "run_generation", lambda **kwargs: kwargs)
    monkeypatch.setattr(generations, "enqueue_generation", lambda _generation_id, payload: captured.append(payload))
    monkeypatch.setattr(generations, "get_task_manager", lambda: _FakeTaskManager())

    await generations.retry_generation("generation", db=_FakeDB(generation))
    assert captured[-1]["mode"] == "retry"
    assert captured[-1]["engine_options"] == options
    assert captured[-1]["seed"] == 77

    generation.status = "completed"
    await generations.regenerate_generation("generation", db=_FakeDB(generation))
    assert captured[-1]["mode"] == "regenerate"
    assert captured[-1]["engine_options"] == options
    assert captured[-1]["seed"] == 77


@pytest.mark.asyncio
async def test_single_history_response_preserves_engine_options():
    # Import the application first so history.py can reuse its response-header
    # helper through the same initialization order used in production.
    from backend.app import app as application
    from backend.routes import history as history_routes

    assert application is not None
    options = {"indextts": _index_form_options(emotion_mode="auto")}
    generation = SimpleNamespace(
        id="generation",
        profile_id="profile",
        text="Text",
        language="en",
        audio_path="generation.wav",
        duration=1.0,
        seed=12,
        instruct=None,
        engine="indextts",
        model_size=None,
        engine_options=options,
        status="completed",
        error=None,
        is_favorited=False,
        created_at=datetime.utcnow(),
    )

    class Query:
        def join(self, *_args):
            return self

        def filter(self, *_args):
            return self

        def first(self):
            return generation, "Profile"

    class DB:
        def query(self, *_args):
            return Query()

    response = await history_routes.get_generation("generation", db=DB())
    assert response.engine_options is not None
    assert response.engine_options.model_dump(mode="json", exclude_none=True) == options


def test_generation_options_migration_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as connection:
        connection.execute(
            text(
                "CREATE TABLE generations ("
                "id VARCHAR PRIMARY KEY, audio_path VARCHAR, status VARCHAR, error TEXT, "
                "engine VARCHAR, model_size VARCHAR, is_favorited BOOLEAN, source VARCHAR)"
            )
        )
        connection.commit()

    run_migrations(engine)
    run_migrations(engine)
    columns = {column["name"] for column in inspect(engine).get_columns("generations")}
    assert "engine_options" in columns


@pytest.mark.asyncio
async def test_emotion_asset_is_opaque_reusable_and_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(generation_assets, "get_generation_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(
        generation_assets,
        "validate_and_load_reference_audio",
        lambda *_args: (True, None, np.ones(48_000, dtype=np.float32), 24_000),
    )
    monkeypatch.setattr(
        generation_assets,
        "save_audio",
        lambda _audio, path, _sample_rate: Path(path).write_bytes(b"normalized-wav"),
    )

    asset_id, duration, expires_at = await generation_assets.store_emotion_audio_asset("upload.webm")
    assert str(uuid.UUID(asset_id)) == asset_id
    assert duration == 2.0
    assert expires_at.tzinfo is not None
    assert generation_assets.resolve_emotion_audio_asset(asset_id).name == f"{asset_id}.wav"
    with pytest.raises(ValueError, match="Invalid emotion audio asset ID"):
        generation_assets.resolve_emotion_audio_asset("../../etc/passwd")

    path = tmp_path / f"{asset_id}.wav"
    stale = time.time() - generation_assets.EMOTION_ASSET_TTL_SECONDS - 1
    os.utime(path, (stale, stale))
    with pytest.raises(ValueError, match="missing or has expired"):
        generation_assets.resolve_emotion_audio_asset(asset_id)
    assert path.exists() is False


@pytest.mark.asyncio
async def test_generation_export_import_round_trips_emotion_audio(tmp_path, monkeypatch):
    old_asset_id = str(uuid.uuid4())
    new_asset_id = str(uuid.uuid4())
    audio_path = tmp_path / "generation.wav"
    emotion_path = tmp_path / "emotion.wav"
    audio_path.write_bytes(b"generation-audio")
    emotion_path.write_bytes(b"emotion-audio")
    options = {
        "indextts": _index_form_options(
            emotion_mode="audio",
            emotion_audio_asset_id=old_asset_id,
        )
    }
    source_generation = SimpleNamespace(
        id="generation",
        profile_id="profile",
        text="Text",
        language="en",
        audio_path=str(audio_path),
        duration=1.0,
        seed=11,
        instruct=None,
        engine="indextts",
        model_size=None,
        engine_options=options,
        created_at=datetime.utcnow(),
    )
    source_profile = SimpleNamespace(
        id="profile",
        name="Profile",
        description=None,
        language="en",
    )

    class Query:
        def __init__(self, *, first=None, all_values=None):
            self.first_value = first
            self.all_values = all_values or []

        def filter_by(self, **_kwargs):
            return self

        def order_by(self, *_args):
            return self

        def first(self):
            return self.first_value

        def all(self):
            return self.all_values

    class ExportDB:
        def query(self, model):
            if model is DBGeneration:
                return Query(first=source_generation)
            if model is DBVoiceProfile:
                return Query(first=source_profile)
            if model is DBGenerationVersion:
                return Query(all_values=[])
            raise AssertionError(model)

    monkeypatch.setattr(export_import.config, "resolve_storage_path", lambda value: Path(value))
    monkeypatch.setattr(generation_assets, "resolve_emotion_audio_asset", lambda _asset_id: emotion_path)
    archive = export_import.export_generation_to_zip("generation", ExportDB())
    with zipfile.ZipFile(io.BytesIO(archive)) as exported:
        assert "assets/emotion-reference.wav" in exported.namelist()

    imported = []

    class ImportDB:
        def query(self, model):
            if model is DBVoiceProfile:
                return Query(first=source_profile)
            raise AssertionError(model)

        def add(self, value):
            imported.append(value)

        def commit(self):
            return None

        def refresh(self, _value):
            return None

    async def fake_store(_path):
        return new_asset_id, 2.0, datetime.utcnow()

    output_dir = tmp_path / "imported"
    monkeypatch.setattr(export_import.config, "get_generations_dir", lambda: output_dir)
    monkeypatch.setattr(export_import.config, "to_storage_path", lambda value: str(value))
    monkeypatch.setattr(generation_assets, "store_emotion_audio_asset", fake_store)

    result = await export_import.import_generation_from_zip(archive, ImportDB())
    assert result["text"] == "Text"
    assert imported[0].engine == "indextts"
    assert imported[0].engine_options["indextts"]["emotion_audio_asset_id"] == new_asset_id
