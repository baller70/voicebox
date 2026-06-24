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
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional


GROQ_STT_MODEL = os.environ.get("VOICEBOX_GROQ_STT_MODEL", "whisper-large-v3-turbo")


def is_enabled() -> bool:
    return os.environ.get("VOICEBOX_GROQ_STT", "1") not in {"0", "false", "False"}


async def transcribe_file(path: str, language: Optional[str] = None) -> str:
    return await asyncio.to_thread(_transcribe_file_sync, path, language)


def _transcribe_file_sync(path: str, language: Optional[str]) -> str:
    api_key = _get_api_key()
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    fields = {
        "model": GROQ_STT_MODEL,
        "response_format": "json",
    }
    if language:
        fields["language"] = language

    body, content_type = _multipart_body(fields, audio_path)
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "User-Agent": "VoiceBox-Groq-STT/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq STT request failed: {_format_error(detail)}") from e

    text = payload.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"Groq STT response did not include text: {payload!r}")
    return text.strip()


def _multipart_body(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = f"----voicebox-groq-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )

    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


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
