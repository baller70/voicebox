"""
Captures service — persists raw audio alongside its STT transcript and,
optionally, an LLM-refined version.

A capture is a single voice input event (dictation, long-form recording, or
uploaded file). Storage mirrors the generations flow: audio lives under
``data/captures/<id>.wav`` and rows live in the ``captures`` table.
"""

import contextlib
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf
from sqlalchemy.orm import Session

from .. import config
from ..database import Capture as DBCapture
from ..models import CaptureResponse, RefinementFlagsModel
from ..utils.audio import load_audio
from .refinement import RefinementFlags, refine_transcript
from . import groq_stt
from .dictation_text import polish_dictation_text
from .transcribe import get_whisper_model

logger = logging.getLogger(__name__)


VALID_SOURCES = {"dictation", "recording", "file"}
TRANSCRIPTION_FAILED_PREFIX = "[Transcription failed:"
# Suffixes whisper's miniaudio loader can read directly. Anything outside
# this set has to go through librosa for decode + a soundfile transcode
# before whisper sees it.
WHISPER_NATIVE_FORMATS = (".wav", ".mp3", ".flac", ".ogg")


def _read_duration_ms(audio_path: Path) -> int | None:
    """Read duration from container metadata without decoding the waveform."""
    try:
        info = sf.info(str(audio_path))
    except Exception:
        return None
    if info.samplerate <= 0:
        return None
    return int((info.frames / info.samplerate) * 1000)


def _to_response(row: DBCapture) -> CaptureResponse:
    flags_model: Optional[RefinementFlagsModel] = None
    if row.refinement_flags:
        try:
            flags_model = RefinementFlagsModel(**json.loads(row.refinement_flags))
        except (ValueError, TypeError):
            flags_model = None

    return CaptureResponse(
        id=row.id,
        audio_path=row.audio_path,
        source=row.source,
        language=row.language,
        duration_ms=row.duration_ms,
        transcript_raw=row.transcript_raw or "",
        transcript_refined=row.transcript_refined,
        stt_model=row.stt_model,
        llm_model=row.llm_model,
        refinement_flags=flags_model,
        created_at=row.created_at,
    )


def failed_transcript(error: Exception | str) -> str:
    message = str(error).strip() or "Unknown error"
    return f"{TRANSCRIPTION_FAILED_PREFIX} {message}]"


def is_transcription_failed_text(text: str | None) -> bool:
    return bool(text and text.startswith(TRANSCRIPTION_FAILED_PREFIX))


async def _transcribe_with_fallback(
    audio_path: Path,
    language: Optional[str],
    stt_model: Optional[str],
) -> tuple[str, str]:
    """Prefer Groq STT, with the selected local Whisper model as fallback."""
    groq_error: Exception | None = None
    if groq_stt.is_enabled():
        try:
            transcript = await groq_stt.transcribe_file(str(audio_path), language)
            return transcript, "groq-whisper-large-v3-turbo"
        except Exception as exc:
            groq_error = exc
            logger.warning(
                "Groq STT failed; falling back to local Whisper for %s: %s",
                audio_path.name,
                exc,
            )

    whisper = get_whisper_model()
    resolved_stt = stt_model or whisper.model_size
    try:
        transcript = await whisper.transcribe(str(audio_path), language, resolved_stt)
        return transcript, resolved_stt
    except Exception as local_error:
        if groq_error is not None:
            raise RuntimeError(
                f"Groq STT failed: {groq_error}; local Whisper fallback failed: {local_error}"
            ) from local_error
        raise


async def create_capture(
    *,
    audio_bytes: bytes,
    filename: str,
    source: str,
    language: Optional[str],
    stt_model: Optional[str],
    transcript_raw: Optional[str] = None,
    db: Session,
) -> CaptureResponse:
    """Persist raw audio, run STT, store the row."""
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source '{source}'. Must be one of {sorted(VALID_SOURCES)}")

    capture_id = str(uuid.uuid4())
    suffix = Path(filename).suffix.lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"):
        suffix = ".wav"

    raw_path = config.get_captures_dir() / f"{capture_id}{suffix}"
    written_files: list[Path] = []

    try:
        raw_path.write_bytes(audio_bytes)
        written_files.append(raw_path)

        duration_ms = _read_duration_ms(raw_path)

        if suffix in WHISPER_NATIVE_FORMATS:
            # Native dictation arrives as WAV. Do not decode the entire file
            # before Groq STT; metadata is enough for duration, and local
            # Whisper can consume these formats directly if fallback is needed.
            audio_path = raw_path
        else:
            # Decode once with librosa — its audioread fallback handles webm/opus
            # via ffmpeg, which miniaudio (used inside mlx-audio's whisper) can't.
            # The decoded array gives us duration and becomes the canonical WAV
            # for local fallback.
            try:
                audio, sr = load_audio(str(raw_path))
                duration_ms = int((len(audio) / sr) * 1000) if sr else duration_ms
            except Exception as decode_err:
                logger.warning(
                    "Could not decode capture %s (%s): %r", capture_id, suffix, decode_err
                )
                raise ValueError(
                    f"Could not decode {suffix} audio — the recording may be empty or corrupt"
                ) from decode_err

            audio_path = config.get_captures_dir() / f"{capture_id}.wav"
            sf.write(str(audio_path), audio, sr, format="WAV")
            written_files.append(audio_path)
            with contextlib.suppress(OSError):
                raw_path.unlink()
                written_files.remove(raw_path)

        precomputed_transcript = (transcript_raw or "").strip()
        if precomputed_transcript:
            transcript = polish_dictation_text(precomputed_transcript)
            resolved_stt = "progressive-groq-whisper-large-v3-turbo"
        else:
            try:
                transcript, resolved_stt = await _transcribe_with_fallback(
                    audio_path, language, stt_model
                )
                transcript = polish_dictation_text(transcript)
            except Exception as transcribe_err:
                logger.exception("Transcription failed for capture %s", capture_id)
                transcript = failed_transcript(transcribe_err)
                resolved_stt = stt_model or "unknown"

        row = DBCapture(
            id=capture_id,
            audio_path=config.to_storage_path(audio_path),
            source=source,
            language=language,
            duration_ms=duration_ms,
            transcript_raw=transcript,
            stt_model=resolved_stt,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception:
        # Anything between the first write and the commit means the audio on
        # disk has no row pointing at it — clean up so data/captures doesn't
        # accumulate orphan blobs across failed transcribes.
        for path in written_files:
            try:
                path.unlink()
            except OSError:
                pass
        raise

    return _to_response(row)


def list_captures(db: Session, limit: int = 50, offset: int = 0) -> tuple[list[CaptureResponse], int]:
    total = db.query(DBCapture).count()
    rows = (
        db.query(DBCapture)
        .order_by(DBCapture.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    return [_to_response(r) for r in rows], total


def get_capture(capture_id: str, db: Session) -> Optional[CaptureResponse]:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    return _to_response(row) if row else None


def delete_capture(capture_id: str, db: Session) -> bool:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return False

    resolved = config.resolve_storage_path(row.audio_path)
    if resolved and resolved.exists():
        try:
            resolved.unlink()
        except OSError:
            logger.exception("Failed to remove capture audio %s", resolved)

    db.delete(row)
    db.commit()
    return True


async def refine_capture(
    capture_id: str,
    flags: RefinementFlags,
    model_size: Optional[str],
    provider: str,
    db: Session,
) -> Optional[CaptureResponse]:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return None

    refined, llm_size = await refine_transcript(
        row.transcript_raw or "",
        flags,
        model_size=model_size,
        provider=provider,
    )

    row.transcript_refined = refined
    row.llm_model = llm_size
    row.refinement_flags = json.dumps(flags.to_dict())
    db.commit()
    db.refresh(row)
    return _to_response(row)


async def retranscribe_capture(
    capture_id: str,
    stt_model: Optional[str],
    language: Optional[str],
    db: Session,
) -> Optional[CaptureResponse]:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return None

    resolved = config.resolve_storage_path(row.audio_path)
    if not resolved or not resolved.exists():
        raise FileNotFoundError(f"Audio for capture {capture_id} is missing")

    try:
        transcript, resolved_stt = await _transcribe_with_fallback(
            resolved, language, stt_model
        )
        transcript = polish_dictation_text(transcript)
    except Exception as transcribe_err:
        logger.exception("Retranscription failed for capture %s", capture_id)
        transcript = failed_transcript(transcribe_err)
        resolved_stt = stt_model or "unknown"

    row.transcript_raw = transcript
    row.stt_model = resolved_stt
    if language:
        row.language = language
    # Refined text is stale after a fresh STT pass — force a re-refine.
    row.transcript_refined = None
    row.llm_model = None
    row.refinement_flags = None
    db.commit()
    db.refresh(row)
    return _to_response(row)
