"""Groq API key discovery and rotation helpers."""

from __future__ import annotations

import os
import subprocess
import threading


_KEYCHAIN_SERVICES = ["GROQ_API_KEY", *[f"GROQ_API_KEY_{idx}" for idx in range(2, 11)]]
_key_index = 0
_key_lock = threading.Lock()


def get_api_keys() -> list[str]:
    """Return configured Groq keys in priority order without exposing them."""
    keys: list[str] = []
    _extend_unique(keys, _split_keys(os.environ.get("GROQ_API_KEYS")))
    _extend_unique(keys, _split_keys(os.environ.get("GROQ_API_KEY")))

    for service in _KEYCHAIN_SERVICES:
        key = _find_keychain_password(service)
        if key:
            _extend_unique(keys, [key])

    if not keys:
        raise RuntimeError(
            "No Groq API keys found. Set GROQ_API_KEY/GROQ_API_KEYS or add keys to macOS Keychain."
        )
    return keys


def next_api_key() -> tuple[str, int, int]:
    """Return the next key plus its one-based position and the total key count."""
    keys = get_api_keys()
    with _key_lock:
        global _key_index
        index = _key_index % len(keys)
        _key_index += 1
    return keys[index], index + 1, len(keys)


def ordered_api_keys() -> list[tuple[str, int, int]]:
    """Return all keys, starting from the next rotation slot."""
    keys = get_api_keys()
    with _key_lock:
        global _key_index
        start = _key_index % len(keys)
        _key_index += 1
    ordered = keys[start:] + keys[:start]
    return [(key, ((start + offset) % len(keys)) + 1, len(keys)) for offset, key in enumerate(ordered)]


def should_rotate_http_status(status: int) -> bool:
    return status in {408, 425, 429} or status >= 500


def describe_key(position: int, total: int) -> str:
    return f"Groq key {position}/{total}"


def _split_keys(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",").replace(";", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _extend_unique(keys: list[str], candidates: list[str]) -> None:
    seen = set(keys)
    for candidate in candidates:
        if candidate not in seen:
            keys.append(candidate)
            seen.add(candidate)


def _find_keychain_password(service: str) -> str | None:
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                os.environ.get("USER", ""),
                "-s",
                service,
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
