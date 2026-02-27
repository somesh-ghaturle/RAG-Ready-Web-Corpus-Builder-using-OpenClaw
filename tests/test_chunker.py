"""Tests for the chunker module."""

import pytest

from rag_corpus_builder.chunker import DocumentChunker
from rag_corpus_builder.config import ChunkConfig, ChunkStrategy
from rag_corpus_builder.models import ProcessedDocument


def make_doc(text: str, url: str = "https://example.com/test") -> ProcessedDocument:
    return ProcessedDocument(
        url=url,
        title="Test",
        clean_text=text,
        language="en",
        language_confidence=0.99,
        content_hash="abc123",
        word_count=len(text.split()),
    )


class TestRecursiveChunking:
    def test_short_text_single_chunk(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=512, chunk_overlap=0)
        chunker = DocumentChunker(config)
        doc = make_doc("This is a short text that fits in one chunk.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short text that fits in one chunk."
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_long_text_multiple_chunks(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=64, chunk_overlap=0)
        chunker = DocumentChunker(config)
        # Create text that's definitely longer than 50 tokens
        long_text = " ".join([f"Word number {i} in the document." for i in range(100)])
        doc = make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.document_url == "https://example.com/test"

    def test_empty_text_no_chunks(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=512, chunk_overlap=0)
        chunker = DocumentChunker(config)
        doc = make_doc("")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0


class TestSlidingWindowChunking:
    def test_produces_overlapping_chunks(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.SLIDING_WINDOW, chunk_size=64, chunk_overlap=10)
        chunker = DocumentChunker(config)
        long_text = " ".join([f"Sentence {i} has some words in it." for i in range(50)])
        doc = make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1


class TestSentenceChunking:
    def test_sentence_boundaries(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.SENTENCE, chunk_size=100, chunk_overlap=0)
        chunker = DocumentChunker(config)
        text = "First sentence about Python. Second sentence about Java. Third sentence about Rust. Fourth sentence about Go."
        doc = make_doc(text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text.strip()


class TestChunkMetadata:
    def test_chunk_ids_unique(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=64, chunk_overlap=0)
        chunker = DocumentChunker(config)
        long_text = " ".join([f"Paragraph {i} is here." for i in range(100)])
        doc = make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_indices_sequential(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=64, chunk_overlap=0)
        chunker = DocumentChunker(config)
        long_text = " ".join([f"Word {i}." for i in range(200)])
        doc = make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)
