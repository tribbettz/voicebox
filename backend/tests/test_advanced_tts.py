import asyncio

import numpy as np

from backend.utils.advanced_tts import (
    PauseToken,
    QWEN_ADVANCED_PREFIX,
    TextToken,
    decode_qwen_advanced_instruct,
    generate_with_advanced_controls,
    parse_pause_syntax,
)


def test_parse_pause_syntax_supports_ms_seconds_and_bare_milliseconds():
    tokens = parse_pause_syntax(
        "First. [pause 1200] Second. [pause 500ms] Third. [pause 1.2s] Done."
    )

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
    encoded = (
        QWEN_ADVANCED_PREFIX
        + '{"temperature":0.8,"top_p":0.9,"top_k":40,"repetition_penalty":1.08,"bad":"x"}'
    )

    controls, instruct = decode_qwen_advanced_instruct(encoded)

    assert controls == {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.08,
    }
    assert instruct is None


def test_decode_normal_instruct_passes_through():
    controls, instruct = decode_qwen_advanced_instruct("Speak warmly")
    assert controls == {}
    assert instruct == "Speak warmly"


class _FakeBackend:
    async def generate(
        self,
        text,
        voice_prompt,
        language="en",
        seed=None,
        instruct=None,
    ):
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
    assert np.all(audio[:100] == 1)
    assert np.all(audio[100:600] == 0)
    assert np.all(audio[600:] == 1)


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
    assert np.all(audio[250:] == 1)
