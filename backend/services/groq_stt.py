"""Groq-hosted speech-to-text helpers.

Used for dictation on machines where local MLX Whisper is unstable or too
slow. The API is OpenAI-compatible and returns JSON with a ``text`` field.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import soundfile as sf

from ..utils.audio import load_audio

GROQ_STT_MODEL = os.environ.get("VOICEBOX_GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_STT_MAX_DIRECT_BYTES = int(os.environ.get("VOICEBOX_GROQ_STT_MAX_DIRECT_BYTES", "18000000"))
GROQ_STT_CHUNK_SECONDS = float(os.environ.get("VOICEBOX_GROQ_STT_CHUNK_SECONDS", "110"))
GROQ_STT_CHUNK_SAMPLE_RATE = 16000
GROQ_STT_ATTEMPTS = max(1, int(os.environ.get("VOICEBOX_GROQ_STT_ATTEMPTS", "2")))
GROQ_STT_TIMEOUT_SECONDS = float(os.environ.get("VOICEBOX_GROQ_STT_TIMEOUT_SECONDS", "25"))
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

_client_lock = threading.Lock()
_client: httpx.Client | None = None


def is_enabled() -> bool:
    return os.environ.get("VOICEBOX_GROQ_STT", "1") not in {"0", "false", "False"}


async def transcribe_file(path: str, language: Optional[str] = None) -> str:
    return await asyncio.to_thread(_transcribe_file_sync, path, language)


def _transcribe_file_sync(path: str, language: Optional[str]) -> str:
    api_key = _get_api_key()
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if audio_path.stat().st_size > GROQ_STT_MAX_DIRECT_BYTES:
        return _transcribe_file_in_chunks(api_key, audio_path, language)

    try:
        return _transcribe_file_once(api_key, audio_path, language)
    except RuntimeError as e:
        if _is_payload_too_large(e):
            return _transcribe_file_in_chunks(api_key, audio_path, language)
        raise


def _transcribe_file_once(api_key: str, audio_path: Path, language: Optional[str]) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data = {
        "model": GROQ_STT_MODEL,
        "response_format": "json",
    }
    if language:
        data["language"] = language

    payload = None
    for attempt in range(1, GROQ_STT_ATTEMPTS + 1):
        try:
            with audio_path.open("rb") as audio_file:
                response = _get_client().post(
                    GROQ_STT_URL,
                    data=data,
                    files={
                        "file": (
                            audio_path.name,
                            audio_file,
                            mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream",
                        )
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            response.raise_for_status()
            payload = response.json()
            break
        except httpx.HTTPStatusError as e:
            error = RuntimeError(f"Groq STT request failed: {_format_error(e.response.text)}")
            if attempt >= GROQ_STT_ATTEMPTS or not _is_transient_http_status(e.response.status_code):
                raise error from e
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as e:
            _reset_client()
            if attempt >= GROQ_STT_ATTEMPTS:
                raise RuntimeError(f"Groq STT request failed after {attempt} attempts: {e}") from e

        time.sleep(0.5 * attempt)

    if payload is None:
        raise RuntimeError("Groq STT request failed without a response")

    text = payload.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"Groq STT response did not include text: {payload!r}")
    return text.strip()


def _is_transient_http_status(status: int) -> bool:
    return status in {408, 425, 429} or status >= 500


def _transcribe_file_in_chunks(api_key: str, audio_path: Path, language: Optional[str]) -> str:
    audio, sample_rate = load_audio(
        str(audio_path),
        sample_rate=GROQ_STT_CHUNK_SAMPLE_RATE,
        mono=True,
    )
    chunk_samples = max(1, int(sample_rate * GROQ_STT_CHUNK_SECONDS))
    parts: list[str] = []

    with tempfile.TemporaryDirectory(prefix="voicebox-groq-stt-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, start in enumerate(range(0, len(audio), chunk_samples), start=1):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) == 0:
                continue
            chunk_path = tmp_root / f"{audio_path.stem}.part{idx:03d}.wav"
            sf.write(str(chunk_path), chunk, sample_rate, format="WAV", subtype="PCM_16")
            text = _transcribe_file_once(api_key, chunk_path, language)
            if text:
                parts.append(text)

    return " ".join(parts).strip()


def _is_payload_too_large(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "413" in message or "payload too large" in message or "request entity too large" in message


def _get_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                timeout=GROQ_STT_TIMEOUT_SECONDS,
                headers={"User-Agent": "VoiceBox-Groq-STT/1.0"},
                limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
            )
        return _client


def _reset_client() -> None:
    global _client
    with _client_lock:
        client = _client
        _client = None
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def _get_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", os.environ.get("USER", ""), "-s", "GROQ_API_KEY", "-w"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as e:
        raise RuntimeError("GROQ_API_KEY is not set and was not found in macOS Keychain.") from e

    api_key = result.stdout.strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is empty.")
    return api_key


def _format_error(body: str) -> str:
    try:
        payload = json.loads(body)
        return payload.get("error", {}).get("message") or body
    except Exception:
        return body
