"""Advanced generation helpers for explicit pauses and Qwen Base controls.

This module intentionally sits on top of the existing TTS backend protocol so
Voicebox's engine architecture remains unchanged. The normal generation path
can opt into these helpers without teaching every backend about Qwen-specific
sampling knobs.
"""

from __future__ import annotations

import json
import math
import re
import secrets
from dataclasses import dataclass

import numpy as np

from .chunked_tts import generate_chunked

QWEN_ADVANCED_PREFIX = "__VOICEBOX_QWEN_ADVANCED__:"

# [pause 1200] (milliseconds), [pause 500ms], [pause 1.2s]
_PAUSE_RE = re.compile(
    r"\[pause\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)?\]",
    re.IGNORECASE,
)
MAX_PAUSE_MS = 30_000
PAUSE_EDGE_FADE_MS = 5
QWEN_FINAL_TAIL_MS = 100

_QWEN_CONTROL_RANGES: dict[str, tuple[float, float, bool]] = {
    "temperature": (0.1, 1.5, False),
    "top_p": (0.1, 1.0, False),
    "top_k": (1, 100, True),
    "repetition_penalty": (1.0, 1.5, False),
}


@dataclass(frozen=True)
class PauseToken:
    milliseconds: int


@dataclass(frozen=True)
class TextToken:
    text: str


def parse_pause_syntax(text: str) -> list[TextToken | PauseToken]:
    """Split text into speakable spans and explicit-silence tokens.

    Bare values are milliseconds for compatibility with the documented
    ``[pause 1200]`` form. ``ms`` and ``s`` suffixes are also accepted.
    Pauses are clamped to 30 seconds to prevent accidental huge allocations.
    """
    tokens: list[TextToken | PauseToken] = []
    cursor = 0

    for match in _PAUSE_RE.finditer(text):
        before = text[cursor : match.start()].strip()
        if before:
            tokens.append(TextToken(before))

        value = float(match.group("value"))
        unit = (match.group("unit") or "ms").lower()
        milliseconds = round(value * 1000) if unit == "s" else round(value)
        milliseconds = max(0, min(milliseconds, MAX_PAUSE_MS))
        if milliseconds:
            tokens.append(PauseToken(milliseconds))
        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        tokens.append(TextToken(tail))

    return tokens


def decode_qwen_advanced_instruct(instruct: str | None) -> tuple[dict[str, float | int], str | None]:
    """Decode Voicebox's internal Qwen Base control envelope.

    CustomVoice uses ``instruct`` as natural-language delivery guidance. Base
    Qwen does not, so the existing optional field provides a backward-compatible
    transport for per-generation Base sampling controls without a DB migration.
    Non-envelope strings pass through untouched.
    """
    if not instruct or not instruct.startswith(QWEN_ADVANCED_PREFIX):
        return {}, instruct

    raw = instruct[len(QWEN_ADVANCED_PREFIX) :]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}, None

    if not isinstance(data, dict):
        return {}, None

    controls: dict[str, float | int] = {}
    for name, (minimum, maximum, integer_only) in _QWEN_CONTROL_RANGES.items():
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value) or not minimum <= value <= maximum:
            continue
        if integer_only:
            if not float(value).is_integer():
                continue
            value = int(value)
        controls[name] = value

    return controls, None


class QwenAdvancedBackend:
    """Thin proxy that adds supported Qwen Base generation kwargs."""

    def __init__(self, backend, controls: dict[str, float | int]):
        self._backend = backend
        self._controls = controls

    def __getattr__(self, name):
        return getattr(self._backend, name)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ):
        # PyTorchTTSBackend keeps the upstream Qwen model at ``model``.
        await self._backend.load_model_async(None)

        from ..backends import LANGUAGE_CODE_TO_NAME
        from ..backends.base import manual_seed

        if seed is not None:
            manual_seed(seed, self._backend.device)

        kwargs = dict(self._controls)
        # Qwen exposes a second acoustic/subtalker sampler. Mirror the main
        # controls there so a single UI control has predictable meaning.
        if "temperature" in kwargs:
            kwargs["subtalker_temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            kwargs["subtalker_top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            kwargs["subtalker_top_k"] = kwargs["top_k"]

        def _generate_sync():
            wavs, sample_rate = self._backend.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=voice_prompt,
                language=LANGUAGE_CODE_TO_NAME.get(language, "auto"),
                # Voicebox always has the complete text span available. Giving
                # Qwen the full-text prompt arrangement reduces premature EOS
                # on short final spans; False only simulates streamed input.
                non_streaming_mode=True,
                **kwargs,
            )
            return wavs[0], sample_rate

        import asyncio

        return await asyncio.to_thread(_generate_sync)


def prepare_advanced_backend(backend, engine: str, instruct: str | None):
    """Apply engine-specific advanced controls and return effective instruct.

    Qwen Base does not support natural-language delivery instructions. Its
    persisted ``instruct`` field is only an internal transport for the sampling
    envelope, which must never reach model inference.
    """
    if engine != "qwen":
        return backend, instruct

    controls, _ = decode_qwen_advanced_instruct(instruct)
    if controls:
        backend = QwenAdvancedBackend(backend, controls)
    return backend, None


def _fade_speech_edge(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fade_in: bool = False,
    fade_out: bool = False,
) -> np.ndarray:
    """Apply a tiny edge fade without consuming any requested silence."""
    result = np.array(audio, dtype=np.float32, copy=True)
    fade_samples = min(len(result), round(sample_rate * PAUSE_EDGE_FADE_MS / 1000))
    if fade_samples <= 0:
        return result

    # Do not include an exact zero in the speech samples: the inserted zero run
    # must remain exactly the requested length when measured from the waveform.
    ramp = np.linspace(1.0 / fade_samples, 1.0, fade_samples, dtype=np.float32)
    if fade_in:
        result[:fade_samples] *= ramp
    if fade_out:
        result[-fade_samples:] *= ramp[::-1]
    return result


async def generate_with_advanced_controls(
    backend,
    text: str,
    voice_prompt: dict,
    *,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    engine_options: dict | None = None,
    max_chunk_chars: int = 800,
    crossfade_ms: int = 50,
    final_tail_ms: int = 0,
    trim_fn=None,
    runaway_detector=None,
):
    """Generate with explicit-pause syntax while preserving existing chunking.

    Text spans are generated through Voicebox's existing ``generate_chunked``
    helper. Explicit pauses are inserted as real zero-valued PCM and therefore
    cannot be ignored by a TTS model. Crossfade is kept within text spans but
    never across an explicit pause boundary. Speech receives a 5 ms edge fade
    next to inserted silence to avoid clicks without changing pause duration.
    """
    tokens = parse_pause_syntax(text)
    if not tokens:
        return np.array([], dtype=np.float32), 24_000

    generated_parts: list[tuple[str, np.ndarray | int]] = []
    sample_rate: int | None = None
    text_count = sum(isinstance(token, TextToken) for token in tokens)

    # An explicit pause forces independent model calls. Reusing one seed keeps
    # their stochastic starting point stable, which materially reduces changes
    # in cloned timbre/prosody across the artificial boundary. A blank field is
    # still a fresh generation: it simply chooses one request-local seed.
    span_seed = seed
    if text_count > 1 and span_seed is None:
        span_seed = secrets.randbits(31)

    for token in tokens:
        if isinstance(token, PauseToken):
            generated_parts.append(("pause", token.milliseconds))
            continue

        audio, sr = await generate_chunked(
            backend,
            token.text,
            voice_prompt,
            language=language,
            seed=span_seed,
            instruct=instruct,
            engine_options=engine_options,
            max_chunk_chars=max_chunk_chars,
            crossfade_ms=crossfade_ms,
            trim_fn=trim_fn,
            runaway_detector=runaway_detector,
        )
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"TTS sample rate changed between text spans: {sample_rate} -> {sr}")
        generated_parts.append(("speech", np.asarray(audio, dtype=np.float32)))

    if sample_rate is None:
        # Text consisting only of pause directives has no engine sample rate;
        # use the standard Voicebox/Qwen rate for a deterministic silent WAV.
        sample_rate = 24_000
    audio_parts: list[np.ndarray] = []
    for index, (kind, value) in enumerate(generated_parts):
        if kind == "pause":
            milliseconds = int(value)
            audio_parts.append(np.zeros(round(sample_rate * milliseconds / 1000), dtype=np.float32))
            continue

        speech = np.asarray(value, dtype=np.float32)
        previous_is_pause = index > 0 and generated_parts[index - 1][0] == "pause"
        next_is_pause = index + 1 < len(generated_parts) and generated_parts[index + 1][0] == "pause"
        append_tail = index == len(generated_parts) - 1 and final_tail_ms > 0
        speech = _fade_speech_edge(
            speech,
            sample_rate,
            fade_in=previous_is_pause,
            fade_out=next_is_pause or append_tail,
        )
        audio_parts.append(speech)

        if append_tail:
            audio_parts.append(np.zeros(round(sample_rate * final_tail_ms / 1000), dtype=np.float32))

    return np.concatenate(audio_parts) if audio_parts else np.array([], dtype=np.float32), sample_rate
