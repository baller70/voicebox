from backend.services.dictation_text import merge_progressive_transcripts, polish_dictation_text


def test_spoken_punctuation_and_line_breaks():
    raw = "send this to kevin comma new line we are ready period"
    assert polish_dictation_text(raw) == "Send this to kevin,\nWe are ready."


def test_progressive_chunk_overlap_is_removed():
    parts = [
        "we need to fix the hotkey and test it",
        "hotkey and test it with playwright today",
    ]
    assert (
        merge_progressive_transcripts(parts)
        == "We need to fix the hotkey and test it with playwright today"
    )


def test_repetitive_stt_artifacts_still_collapse():
    raw = "hello URL URL URL URL URL URL goodbye"
    assert polish_dictation_text(raw) == "Hello goodbye"
