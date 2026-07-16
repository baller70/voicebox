"""Groq-hosted TTS backend.

This backend uses Groq's OpenAI-compatible speech endpoint. It is stateless:
there is no local model download and no reference-audio cloning.
"""

import asyncio
import json
import os
import tempfile
import urllib.error
import urllib.request
from typing import Optional, Tuple

import numpy as np

from ..services.groq_keys import describe_key, ordered_api_keys, should_rotate_http_status
from ..utils.audio import load_audio

GROQ_TTS_MODEL_ENGLISH = "canopylabs/orpheus-v1-english"
GROQ_TTS_MODEL_ARABIC = "canopylabs/orpheus-arabic-saudi"
GROQ_TTS_SAMPLE_RATE = 24000
GROQ_DEFAULT_VOICE = "troy"

# (voice_id, display_name, gender, lang_code)
GROQ_TTS_VOICES = [
    ("autumn", "Autumn", "female", "en"),
    ("diana", "Diana", "female", "en"),
    ("hannah", "Hannah", "female", "en"),
    ("austin", "Austin", "male", "en"),
    ("daniel", "Daniel", "male", "en"),
    ("troy", "Troy", "male", "en"),
    ("fahad", "Fahad", "male", "ar"),
    ("sultan", "Sultan", "male", "ar"),
    ("lulwa", "Lulwa", "female", "ar"),
    ("noura", "Noura", "female", "ar"),
]

_VOICE_LANG = {voice_id: lang for voice_id, _name, _gender, lang in GROQ_TTS_VOICES}


class GroqTTSBackend:
    """Fast hosted TTS through Groq Orpheus."""

    def __init__(self):
        self.model_size = "default"

    async def load_model(self, model_size: str = "default") -> None:
        self.model_size = model_size

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        raise ValueError("Groq TTS uses preset voices and does not support local voice cloning.")

    async def combine_voice_prompts(self, audio_paths: list[str], reference_texts: list[str]) -> Tuple[np.ndarray, str]:
        raise ValueError("Groq TTS uses preset voices and does not support local voice cloning.")

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        return await asyncio.to_thread(self._generate_sync, text, voice_prompt, language, instruct)

    def unload_model(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def _get_model_path(self, model_size: str) -> str:
        return "groq-api"

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return True

    def _generate_sync(
        self,
        text: str,
        voice_prompt: dict,
        language: str,
        instruct: Optional[str],
    ) -> Tuple[np.ndarray, int]:
        voice = self._resolve_voice(voice_prompt, language)
        model = GROQ_TTS_MODEL_ARABIC if _VOICE_LANG.get(voice) == "ar" or language == "ar" else GROQ_TTS_MODEL_ENGLISH
        input_text = self._apply_instruct(text, instruct)

        payload = json.dumps(
            {
                "model": model,
                "voice": voice,
                "input": input_text,
                "response_format": "wav",
            }
        ).encode("utf-8")

        wav_bytes = self._post_speech_request(payload)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            return load_audio(tmp_path, sample_rate=GROQ_TTS_SAMPLE_RATE, mono=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _post_speech_request(self, payload: bytes) -> bytes:
        last_error: Exception | None = None
        for api_key, key_position, key_total in ordered_api_keys():
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/audio/speech",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "VoiceBox-Groq-TTS/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    return response.read()
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"{describe_key(key_position, key_total)} failed: {self._format_error(body)}"
                )
                if not should_rotate_http_status(e.code):
                    raise RuntimeError(f"Groq TTS request failed: {last_error}") from e
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"{describe_key(key_position, key_total)} failed: {e}")
        raise RuntimeError(f"Groq TTS request failed: {last_error}")

    def _resolve_voice(self, voice_prompt: dict, language: str) -> str:
        voice_id = voice_prompt.get("preset_voice_id") if isinstance(voice_prompt, dict) else None
        if voice_id in _VOICE_LANG:
            return voice_id
        if language == "ar":
            return "fahad"
        return GROQ_DEFAULT_VOICE

    def _apply_instruct(self, text: str, instruct: Optional[str]) -> str:
        if not instruct:
            return text
        return f"[{instruct.strip()}] {text}"

    def _format_error(self, body: str) -> str:
        try:
            payload = json.loads(body)
            return payload.get("error", {}).get("message") or body
        except Exception:
            return body


__all__ = ["GroqTTSBackend", "GROQ_TTS_VOICES"]
