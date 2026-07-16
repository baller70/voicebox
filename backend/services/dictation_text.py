"""Fast deterministic cleanup for dictation transcripts.

This runs before optional LLM refinement. It handles the things a dictation
user expects instantly: progressive chunk de-duplication, spoken punctuation,
line breaks, and light spacing/capitalization cleanup.
"""

from __future__ import annotations

import re

from .refinement import collapse_repetitive_artifacts


_MAX_OVERLAP_WORDS = 14
_MIN_OVERLAP_WORDS = 2

_PUNCTUATION_COMMANDS: tuple[tuple[str, str], ...] = (
    ("question mark", "?"),
    ("exclamation point", "!"),
    ("exclamation mark", "!"),
    ("period", "."),
    ("full stop", "."),
    ("comma", ","),
    ("semicolon", ";"),
    ("semi colon", ";"),
    ("colon", ":"),
    ("open parenthesis", "("),
    ("close parenthesis", ")"),
    ("open paren", "("),
    ("close paren", ")"),
    ("open quote", '"'),
    ("close quote", '"'),
    ("dash", "-"),
    ("hyphen", "-"),
)

_LINE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("new paragraph", "\n\n"),
    ("next paragraph", "\n\n"),
    ("new line", "\n"),
    ("next line", "\n"),
)

_CAPITALIZE_AFTER = re.compile(r"(^|[.!?]\s+|\n+)([a-z])")


def polish_dictation_text(text: str) -> str:
    """Apply fast local dictation cleanup without rephrasing the speaker."""
    cleaned = collapse_repetitive_artifacts(text.strip())
    if not cleaned:
        return ""

    cleaned = _apply_spoken_commands(cleaned)
    cleaned = _normalize_spacing(cleaned)
    cleaned = _capitalize_sentence_starts(cleaned)
    return cleaned.strip()


def merge_progressive_transcripts(parts: list[str]) -> str:
    """Merge chunk transcripts while removing duplicated chunk-boundary words."""
    merged = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not merged:
            merged = part
            continue
        merged = _merge_pair(merged, part)
    return polish_dictation_text(merged)


def _merge_pair(left: str, right: str) -> str:
    left_words = left.split()
    right_words = right.split()
    max_overlap = min(_MAX_OVERLAP_WORDS, len(left_words), len(right_words))
    for size in range(max_overlap, _MIN_OVERLAP_WORDS - 1, -1):
        if _words_key(left_words[-size:]) == _words_key(right_words[:size]):
            return " ".join([*left_words, *right_words[size:]])
    return f"{left} {right}"


def _words_key(words: list[str]) -> list[str]:
    return [re.sub(r"[^\w]", "", word).lower() for word in words]


def _apply_spoken_commands(text: str) -> str:
    cleaned = f" {text} "
    for phrase, replacement in _LINE_COMMANDS:
        cleaned = re.sub(
            rf"\s+{re.escape(phrase)}\s+",
            replacement,
            cleaned,
            flags=re.IGNORECASE,
        )
    for phrase, replacement in _PUNCTUATION_COMMANDS:
        cleaned = re.sub(
            rf"\s+{re.escape(phrase)}(?=\s|$)",
            replacement,
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned.strip()


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[^\s\n\"')\]])", r"\1 ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', ' "', text)
    return text


def _capitalize_sentence_starts(text: str) -> str:
    return _CAPITALIZE_AFTER.sub(lambda match: f"{match.group(1)}{match.group(2).upper()}", text)
