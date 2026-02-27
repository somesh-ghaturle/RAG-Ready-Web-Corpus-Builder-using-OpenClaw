"""Tests for the preprocessor module."""

from rag_corpus_builder.config import PreprocessConfig
from rag_corpus_builder.models import ExtractedDocument
from rag_corpus_builder.preprocessor import SimHasher, TextPreprocessor


def make_extracted_doc(
    text: str,
    url: str = "https://example.com/page",
    content_hash: str = "",
) -> ExtractedDocument:
    import hashlib

    return ExtractedDocument(
        url=url,
        title="Test",
        main_content=text,
        content_hash=content_hash or hashlib.sha256(text.encode()).hexdigest(),
        word_count=len(text.split()),
    )


class TestSimHasher:
    def test_identical_texts_same_hash(self) -> None:
        hasher = SimHasher()
        h1 = hasher.hash("the quick brown fox jumps over the lazy dog")
        h2 = hasher.hash("the quick brown fox jumps over the lazy dog")
        assert h1 == h2

    def test_different_texts_different_hash(self) -> None:
        hasher = SimHasher()
        h1 = hasher.hash("the quick brown fox jumps over the lazy dog")
        h2 = hasher.hash("a completely different text about programming in python")
        assert h1 != h2

    def test_similar_texts_high_similarity(self) -> None:
        hasher = SimHasher()
        h1 = hasher.hash("the quick brown fox jumps over the lazy dog near the river bank")
        h2 = hasher.hash("the quick brown fox jumps over the lazy dog near the river side")
        sim = SimHasher.similarity(h1, h2)
        assert sim > 0.7

    def test_empty_text_returns_zero(self) -> None:
        hasher = SimHasher()
        assert hasher.hash("") == 0


class TestTextPreprocessor:
    def test_whitespace_normalization(self) -> None:
        config = PreprocessConfig(
            remove_extra_whitespace=True,
            target_languages=[],
            dedup_enabled=False,
        )
        proc = TextPreprocessor(config)
        doc = make_extracted_doc("Hello    world\n\n\n\nThis   is   a   test   with lots of whitespace and enough content to pass filters")
        result = proc.process(doc)
        assert result is not None
        assert "    " not in result.clean_text
        assert "\n\n\n" not in result.clean_text

    def test_exact_dedup(self) -> None:
        config = PreprocessConfig(
            dedup_enabled=True,
            target_languages=[],
        )
        proc = TextPreprocessor(config)
        text = "This document has enough content to pass the minimum length filters used by the preprocessor module"
        doc1 = make_extracted_doc(text, url="https://example.com/a")
        doc2 = make_extracted_doc(text, url="https://example.com/b")

        result1 = proc.process(doc1)
        result2 = proc.process(doc2)
        assert result1 is not None
        assert result2 is None  # Should be filtered as exact duplicate

    def test_short_content_filtered(self) -> None:
        config = PreprocessConfig(target_languages=[], dedup_enabled=False)
        proc = TextPreprocessor(config)
        doc = make_extracted_doc("Short")
        result = proc.process(doc)
        assert result is None

    def test_language_detection(self) -> None:
        config = PreprocessConfig(
            target_languages=["en"],
            min_language_confidence=0.5,
            dedup_enabled=False,
        )
        proc = TextPreprocessor(config)
        doc = make_extracted_doc(
            "This is a long enough English text that should be detected as English with high confidence by the language detector."
        )
        result = proc.process(doc)
        assert result is not None
        assert result.language == "en"

    def test_batch_processing(self) -> None:
        config = PreprocessConfig(target_languages=[], dedup_enabled=False)
        proc = TextPreprocessor(config)
        docs = [
            make_extracted_doc(
                f"Document number {i} with enough content to pass all filters and be processed correctly",
                url=f"https://example.com/{i}",
            )
            for i in range(5)
        ]
        results = proc.process_batch(docs)
        assert len(results) == 5
