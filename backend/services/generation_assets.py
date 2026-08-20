"""Bounded-lifetime assets used by TTS generation requests.

IndexTTS can take a second reference recording for emotion/delivery.  Clients
upload that recording here and receive an opaque UUID; generation requests
never contain a client-controlled filesystem path.  Assets deliberately live
for seven days so retry/regenerate can reproduce a request, then are removed
by the next upload/resolve cleanup pass.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .. import config
from ..utils.audio import save_audio, validate_and_load_reference_audio

EMOTION_ASSET_TTL_SECONDS = 7 * 24 * 60 * 60
EMOTION_ASSET_MAX_BYTES = 20 * 1024 * 1024
EMOTION_ASSET_MAX_DURATION_SECONDS = 15.0
EMOTION_ASSET_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".webm",
    ".opus",
}


def get_generation_assets_dir() -> Path:
    path = config.get_cache_dir() / "generation-assets" / "emotion-audio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_stale_emotion_assets(now: float | None = None) -> int:
    """Remove expired emotion references and return the number deleted."""
    cutoff = (now if now is not None else time.time()) - EMOTION_ASSET_TTL_SECONDS
    removed = 0
    for path in get_generation_assets_dir().glob("*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


async def store_emotion_audio_asset(source_path: str) -> tuple[str, float, datetime]:
    """Validate, normalize, and persist an uploaded emotion reference."""
    cleanup_stale_emotion_assets()
    is_valid, error, audio, sample_rate = await asyncio.to_thread(
        validate_and_load_reference_audio,
        source_path,
        2.0,
        EMOTION_ASSET_MAX_DURATION_SECONDS,
    )
    if not is_valid or audio is None or sample_rate is None:
        raise ValueError(error or "Invalid emotion reference audio")

    asset_id = str(uuid.uuid4())
    target = get_generation_assets_dir() / f"{asset_id}.wav"
    await asyncio.to_thread(save_audio, audio, str(target), sample_rate)
    duration = len(audio) / sample_rate
    expires_at = datetime.now(UTC) + timedelta(seconds=EMOTION_ASSET_TTL_SECONDS)
    return asset_id, duration, expires_at


def resolve_emotion_audio_asset(asset_id: str) -> Path:
    """Resolve an opaque asset ID, rejecting invalid, missing, or stale IDs."""
    cleanup_stale_emotion_assets()
    try:
        normalized_id = str(uuid.UUID(str(asset_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid emotion audio asset ID") from exc

    asset_dir = get_generation_assets_dir().resolve()
    path = (asset_dir / f"{normalized_id}.wav").resolve()
    if not path.is_relative_to(asset_dir):
        raise ValueError("Invalid emotion audio asset ID")
    if not path.is_file():
        raise ValueError("Emotion reference audio is missing or has expired; upload it again")
    if time.time() - path.stat().st_mtime > EMOTION_ASSET_TTL_SECONDS:
        path.unlink(missing_ok=True)
        raise ValueError("Emotion reference audio has expired; upload it again")
    return path
