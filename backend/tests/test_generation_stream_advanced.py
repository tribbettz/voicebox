from types import SimpleNamespace

import numpy as np
import pytest

from backend import models
from backend.routes import generations


class _FakeBackend:
    def __init__(self):
        self.calls = []

    async def generate(
        self,
        text,
        voice_prompt,
        language="en",
        seed=None,
        instruct=None,
    ):
        self.calls.append(
            {
                "text": text,
                "voice_prompt": voice_prompt,
                "language": language,
                "seed": seed,
                "instruct": instruct,
            }
        )
        return np.ones(100, dtype=np.float32), 1000


@pytest.mark.asyncio
async def test_generate_stream_uses_shared_pause_orchestration(monkeypatch):
    backend = _FakeBackend()
    captured = {}
    profile = SimpleNamespace(default_engine="qwen_custom_voice", effects_chain=None)

    async def _get_profile(_profile_id, _db):
        return profile

    async def _noop_async(*_args, **_kwargs):
        return None

    async def _create_prompt(*_args, **_kwargs):
        return {"prompt": "same"}

    def _to_wav_bytes(audio, sample_rate):
        captured["audio"] = audio
        captured["sample_rate"] = sample_rate
        return b"wav-bytes"

    monkeypatch.setattr(generations.profiles, "get_profile", _get_profile)
    monkeypatch.setattr(generations.profiles, "validate_profile_engine", lambda *_args: None)
    monkeypatch.setattr(generations.profiles, "create_voice_prompt_for_profile", _create_prompt)
    monkeypatch.setattr(generations.tts, "audio_to_wav_bytes", _to_wav_bytes)

    import backend.backends as backends

    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda _engine: backend)
    monkeypatch.setattr(backends, "ensure_model_cached_or_raise", _noop_async)
    monkeypatch.setattr(backends, "load_engine_model", _noop_async)
    monkeypatch.setattr(backends, "engine_needs_trim", lambda _engine: False)
    monkeypatch.setattr(backends, "engine_retries_runaway", lambda _engine: False)

    request = models.GenerationRequest(
        profile_id="profile-id",
        text="Before. [pause 10] After.",
        language="en",
        seed=42,
        engine="qwen_custom_voice",
        instruct="Speak warmly",
        normalize=False,
    )

    response = await generations.stream_speech(request, db=object())
    body = b"".join([chunk async for chunk in response.body_iterator])

    assert body == b"wav-bytes"
    assert captured["sample_rate"] == 1000
    assert len(captured["audio"]) == 210
    assert np.all(captured["audio"][100:110] == 0)
    assert [call["seed"] for call in backend.calls] == [42, 42]
    assert [call["instruct"] for call in backend.calls] == ["Speak warmly", "Speak warmly"]
