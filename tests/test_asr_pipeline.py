"""Tests for ASR pipeline behavior."""

from pathlib import Path

from framesleuth.pipeline.asr import ASRPipeline


def test_asr_no_audio_returns_empty_transcript() -> None:
    """No-audio videos should not fail and should return empty transcript."""
    pipeline = ASRPipeline()
    transcript = pipeline.transcribe(audio_path=None, has_audio=False)
    assert transcript.segments == []
    assert transcript.words == []


def test_asr_transcribes_with_model_override(tmp_path: Path) -> None:
    """Model override should make ASR deterministic and unit-testable."""

    class Word:
        def __init__(self, word: str, start: float, end: float, probability: float) -> None:
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class Segment:
        def __init__(self) -> None:
            self.start = 0.0
            self.end = 1.0
            self.text = "hello world"
            self.no_speech_prob = 0.1
            self.words = [Word("hello", 0.0, 0.5, 0.9)]

    class Info:
        language = "en"

    class FakeModel:
        def transcribe(self, _: str, word_timestamps: bool = True, **kwargs: object):
            return [Segment()], Info()

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"wav")

    pipeline = ASRPipeline()
    transcript = pipeline.transcribe(audio_path=audio, has_audio=True, model_override=FakeModel())

    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "hello world"
    assert transcript.words and transcript.words[0]["word"] == "hello"


def test_asr_drops_low_confidence_hallucinations(tmp_path: Path) -> None:
    """Silence hallucinations (high no_speech_prob) are filtered, with their words."""

    class Word:
        def __init__(self, word: str) -> None:
            self.word = word
            self.start = 0.0
            self.end = 0.1
            self.probability = 0.5

    class Segment:
        def __init__(self, text: str, no_speech_prob: float) -> None:
            self.start = 0.0
            self.end = 1.0
            self.text = text
            self.no_speech_prob = no_speech_prob
            self.words = [Word(text)]

    class Info:
        language = "en"

    class FakeModel:
        def transcribe(self, _: str, word_timestamps: bool = True, **kwargs: object):
            # Real speech (conf 0.9) plus a "Thank you." hallucination (conf 0.28).
            return [Segment("real speech", 0.1), Segment("Thank you.", 0.72)], Info()

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"wav")

    pipeline = ASRPipeline(min_confidence=0.5)
    transcript = pipeline.transcribe(audio_path=audio, has_audio=True, model_override=FakeModel())

    assert [s.text for s in transcript.segments] == ["real speech"]
    # The dropped segment's words are dropped too — no phantom timing leaks through.
    assert transcript.words and all(w["word"] == "real speech" for w in transcript.words)


def test_asr_keeps_all_segments_by_default(tmp_path: Path) -> None:
    """With the default threshold (0.0) no filtering occurs — backward compatible."""

    class Segment:
        def __init__(self, no_speech_prob: float) -> None:
            self.start = 0.0
            self.end = 1.0
            self.text = "maybe speech"
            self.no_speech_prob = no_speech_prob
            self.words = []

    class Info:
        language = "en"

    class FakeModel:
        def transcribe(self, _: str, word_timestamps: bool = True, **kwargs: object):
            return [Segment(0.95)], Info()  # very low confidence

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"wav")

    transcript = ASRPipeline().transcribe(
        audio_path=audio, has_audio=True, model_override=FakeModel()
    )
    assert len(transcript.segments) == 1


def test_asr_forwards_vad_and_language(tmp_path: Path) -> None:
    """VAD/language settings reach faster-whisper and the language lands on output."""
    captured: dict[str, object] = {}

    class Info:
        language = "es"

    class FakeModel:
        def transcribe(self, _: str, **kwargs: object):
            captured.update(kwargs)
            return [], Info()

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"wav")

    pipeline = ASRPipeline(vad_filter=True, language="es")
    transcript = pipeline.transcribe(audio_path=audio, has_audio=True, model_override=FakeModel())

    assert captured["vad_filter"] is True
    assert captured["language"] == "es"
    assert transcript.language == "es"
