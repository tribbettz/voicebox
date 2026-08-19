import asyncio

import numpy as np

from backend.backends.pytorch_backend import PyTorchTTSBackend
from backend.utils.advanced_tts import (
    QWEN_ADVANCED_PREFIX,
    QWEN_FINAL_TAIL_MS,
    PauseToken,
    QwenAdvancedBackend,
    TextToken,
    decode_qwen_advanced_instruct,
    generate_with_advanced_controls,
    parse_pause_syntax,
    prepare_advanced_backend,
)


def test_parse_pause_syntax_supports_ms_seconds_and_bare_milliseconds():
    tokens = parse_pause_syntax("First. [pause 1200] Second. [pause 500ms] Third. [pause 1.2s] Done.")

    assert tokens == [
        TextToken("First."),
        PauseToken(1200),
        TextToken("Second."),
        PauseToken(500),
        TextToken("Third."),
        PauseToken(1200),
        TextToken("Done."),
    ]


def test_parse_pause_syntax_does_not_consume_other_bracket_tags():
    tokens = parse_pause_syntax("Hello [cough] there. [pause 250] Continue.")

    assert tokens == [
        TextToken("Hello [cough] there."),
        PauseToken(250),
        TextToken("Continue."),
    ]


def test_pause_is_capped_at_thirty_seconds():
    assert parse_pause_syntax("A [pause 999s] B") == [
        TextToken("A"),
        PauseToken(30_000),
        TextToken("B"),
    ]


def test_decode_qwen_advanced_instruct_filters_unknown_values():
    encoded = QWEN_ADVANCED_PREFIX + '{"temperature":0.8,"top_p":0.9,"top_k":40,"repetition_penalty":1.08,"bad":"x"}'

    controls, instruct = decode_qwen_advanced_instruct(encoded)

    assert controls == {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.08,
    }
    assert instruct is None


def test_decode_qwen_advanced_instruct_rejects_out_of_range_and_wrong_types():
    encoded = QWEN_ADVANCED_PREFIX + '{"temperature":9,"top_p":0.9,"top_k":12.5,"repetition_penalty":true}'

    controls, instruct = decode_qwen_advanced_instruct(encoded)

    assert controls == {"top_p": 0.9}
    assert instruct is None


def test_decode_normal_instruct_passes_through():
    controls, instruct = decode_qwen_advanced_instruct("Speak warmly")
    assert controls == {}
    assert instruct == "Speak warmly"


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
        self.calls.append({"text": text, "seed": seed, "instruct": instruct})
        # 100 samples of non-silent audio per generated text span at 1 kHz.
        return np.ones(100, dtype=np.float32), 1000


def test_generate_with_advanced_controls_inserts_exact_silence():
    audio, sample_rate = asyncio.run(
        generate_with_advanced_controls(
            _FakeBackend(),
            "Before. [pause 500] After.",
            {},
            max_chunk_chars=800,
            crossfade_ms=50,
        )
    )

    assert sample_rate == 1000
    assert len(audio) == 700
    assert np.all(audio[:95] == 1)
    assert audio[99] > 0
    assert np.all(audio[100:600] == 0)
    assert audio[600] > 0
    assert np.all(audio[605:] == 1)


def test_generate_with_advanced_controls_handles_leading_pause():
    audio, sample_rate = asyncio.run(
        generate_with_advanced_controls(
            _FakeBackend(),
            "[pause 250] Hello.",
            {},
        )
    )

    assert sample_rate == 1000
    assert len(audio) == 350
    assert np.all(audio[:250] == 0)
    assert audio[250] > 0
    assert np.all(audio[255:] == 1)


def test_pause_boundary_fades_do_not_change_silence_duration():
    audio, _ = asyncio.run(
        generate_with_advanced_controls(
            _FakeBackend(),
            "Before. [pause 10] After.",
            {},
        )
    )

    np.testing.assert_allclose(audio[95:100], [1.0, 0.8, 0.6, 0.4, 0.2])
    assert np.all(audio[100:110] == 0)
    np.testing.assert_allclose(audio[110:115], [0.2, 0.4, 0.6, 0.8, 1.0])


def test_multiple_and_trailing_pauses_have_exact_sample_counts():
    audio, sample_rate = asyncio.run(
        generate_with_advanced_controls(
            _FakeBackend(),
            "One. [pause 10] Two. [pause 20]",
            {},
            final_tail_ms=QWEN_FINAL_TAIL_MS,
        )
    )

    assert sample_rate == 1000
    assert len(audio) == 230
    assert np.all(audio[100:110] == 0)
    assert np.all(audio[210:230] == 0)


def test_pause_only_text_uses_deterministic_default_sample_rate():
    audio, sample_rate = asyncio.run(generate_with_advanced_controls(_FakeBackend(), "[pause 10] [pause 20]", {}))

    assert sample_rate == 24_000
    assert len(audio) == 720
    assert np.all(audio == 0)


def test_explicit_pause_spans_reuse_fixed_seed():
    backend = _FakeBackend()

    asyncio.run(
        generate_with_advanced_controls(
            backend,
            "Before. [pause 10] After.",
            {},
            seed=42,
        )
    )

    assert [call["seed"] for call in backend.calls] == [42, 42]


def test_blank_seed_chooses_one_fresh_seed_for_all_pause_spans(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr("backend.utils.advanced_tts.secrets.randbits", lambda _bits: 8675309)

    asyncio.run(
        generate_with_advanced_controls(
            backend,
            "Before. [pause 10] After.",
            {},
        )
    )

    assert [call["seed"] for call in backend.calls] == [8675309, 8675309]


def test_final_tail_fades_speech_and_appends_silence():
    audio, sample_rate = asyncio.run(
        generate_with_advanced_controls(
            _FakeBackend(),
            "Santa Claus.",
            {},
            final_tail_ms=QWEN_FINAL_TAIL_MS,
        )
    )

    assert sample_rate == 1000
    assert len(audio) == 200
    assert 0 < audio[99] < 1
    assert np.all(audio[100:] == 0)


def test_prepare_advanced_backend_strips_base_qwen_instruct():
    backend = _FakeBackend()

    prepared, instruct = prepare_advanced_backend(backend, "qwen", "Speak warmly")

    assert prepared is backend
    assert instruct is None


def test_prepare_advanced_backend_preserves_non_qwen_instruct():
    backend = _FakeBackend()

    prepared, instruct = prepare_advanced_backend(backend, "qwen_custom_voice", "Speak warmly")

    assert prepared is backend
    assert instruct == "Speak warmly"


class _FakeQwenModel:
    def __init__(self):
        self.kwargs = None

    def generate_voice_clone(self, **kwargs):
        self.kwargs = kwargs
        return [np.ones(10, dtype=np.float32)], 24_000


class _FakeQwenBackend:
    def __init__(self):
        self.model = _FakeQwenModel()
        self.device = "cpu"

    async def load_model_async(self, _model_size):
        return None


def test_qwen_advanced_backend_uses_full_text_mode_and_sampling_controls():
    backend = _FakeQwenBackend()
    wrapped = QwenAdvancedBackend(backend, {"temperature": 0.7, "top_k": 40})

    audio, sample_rate = asyncio.run(wrapped.generate("Hello.", {}, seed=None))

    assert len(audio) == 10
    assert sample_rate == 24_000
    assert backend.model.kwargs == {
        "text": "Hello.",
        "voice_clone_prompt": {},
        "language": "english",
        "non_streaming_mode": True,
        "temperature": 0.7,
        "subtalker_temperature": 0.7,
        "top_k": 40,
        "subtalker_top_k": 40,
    }


def test_plain_qwen_backend_uses_full_text_mode_without_fake_instruct():
    backend = object.__new__(PyTorchTTSBackend)
    backend.model = _FakeQwenModel()
    backend.device = "cpu"

    async def _already_loaded(_model_size):
        return None

    backend.load_model_async = _already_loaded

    audio, sample_rate = asyncio.run(backend.generate("Hello.", {}, seed=None, instruct="Unsupported Base instruction"))

    assert len(audio) == 10
    assert sample_rate == 24_000
    assert backend.model.kwargs == {
        "text": "Hello.",
        "voice_clone_prompt": {},
        "language": "english",
        "non_streaming_mode": True,
    }
