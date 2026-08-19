"""Advanced generation helpers for explicit pauses and Qwen Base controls.

This module intentionally sits on top of the existing TTS backend protocol so
Voicebox's engine architecture remains unchanged. The normal generation path
can opt into these helpers without teaching every backend about Qwen-specific
sampling knobs.
"""

from __future__ import annotations

import json
import re
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

    allowed = {"temperature", "top_p", "top_k", "repetition_penalty"}
    return {k: v for k, v in data.items() if k in allowed and isinstance(v, (int, float))}, None


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
                **kwargs,
            )
            return wavs[0], sample_rate

        import asyncio

        return await asyncio.to_thread(_generate_sync)


async def generate_with_advanced_controls(
    backend,
    text: str,
    voice_prompt: dict,
    *,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = 800,
    crossfade_ms: int = 50,
    trim_fn=None,
    runaway_detector=None,
):
    """Generate with explicit-pause syntax while preserving existing chunking.

    Text spans are generated through Voicebox's existing ``generate_chunked``
    helper. Explicit pauses are inserted as real zero-valued PCM and therefore
    cannot be ignored by a TTS model. Crossfade is kept within text spans but
    never across an explicit pause boundary.
    """
    tokens = parse_pause_syntax(text)
    if not tokens:
        return np.array([], dtype=np.float32), 24_000

    audio_parts: list[np.ndarray] = []
    sample_rate: int | None = None
    text_index = 0

    for token in tokens:
        if isinstance(token, PauseToken):
            if sample_rate is None:
                # Defer leading pauses until we know the engine sample rate.
                audio_parts.append(np.array([-(token.milliseconds)], dtype=np.float32))
            else:
                count = round(sample_rate * token.milliseconds / 1000)
                audio_parts.append(np.zeros(count, dtype=np.float32))
            continue

        chunk_seed = seed + text_index if seed is not None else None
        text_index += 1
        audio, sr = await generate_chunked(
            backend,
            token.text,
            voice_prompt,
            language=language,
            seed=chunk_seed,
            instruct=instruct,
            max_chunk_chars=max_chunk_chars,
            crossfade_ms=crossfade_ms,
            trim_fn=trim_fn,
            runaway_detector=runaway_detector,
        )
        sample_rate = sr

        # Resolve any leading-pause sentinels now that sample rate is known.
        for index, part in enumerate(audio_parts):
            if len(part) == 1 and part[0] < 0:
                milliseconds = int(-part[0])
                audio_parts[index] = np.zeros(
                    round(sample_rate * milliseconds / 1000), dtype=np.float32
                )
        audio_parts.append(np.asarray(audio, dtype=np.float32))

    if sample_rate is None:
        # Text consisting only of pause directives has no engine sample rate;
        # use the standard Voicebox/Qwen rate for a deterministic silent WAV.
        sample_rate = 24_000
        resolved: list[np.ndarray] = []
        for part in audio_parts:
            milliseconds = int(-part[0]) if len(part) == 1 and part[0] < 0 else 0
            resolved.append(
                np.zeros(round(sample_rate * milliseconds / 1000), dtype=np.float32)
            )
        audio_parts = resolved

    return np.concatenate(audio_parts) if audio_parts else np.array([], dtype=np.float32), sample_rate
