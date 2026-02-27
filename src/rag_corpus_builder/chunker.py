"""Document chunking: split processed documents into RAG-ready chunks."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

import tiktoken

from rag_corpus_builder.config import ChunkConfig, ChunkStrategy
from rag_corpus_builder.models import DocumentChunk, ProcessedDocument

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split documents into token-counted chunks using configurable strategies."""

    def __init__(self, config: ChunkConfig) -> None:
        self.config = config
        self._encoder = tiktoken.get_encoding(config.tokenizer)

    def chunk_document(self, doc: ProcessedDocument) -> list[DocumentChunk]:
        """Chunk a single document using the configured strategy."""
        text = doc.clean_text.strip()
        if not text:
            return []

        strategy = self.config.strategy
        if strategy == ChunkStrategy.RECURSIVE:
            chunks_text = self._recursive_chunk(text)
        elif strategy == ChunkStrategy.SLIDING_WINDOW:
            chunks_text = self._sliding_window_chunk(text)
        elif strategy == ChunkStrategy.SENTENCE:
            chunks_text = self._sentence_chunk(text)
        elif strategy == ChunkStrategy.SEMANTIC:
            chunks_text = self._semantic_chunk(text)
        else:
            chunks_text = self._recursive_chunk(text)

        total = len(chunks_text)
        results: list[DocumentChunk] = []
        for i, chunk_text in enumerate(chunks_text):
            token_count = len(self._encoder.encode(chunk_text))
            chunk_id = hashlib.sha256(f"{doc.url}::{i}::{chunk_text[:100]}".encode()).hexdigest()[:16]

            results.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_url=doc.url,
                    document_title=doc.title,
                    text=chunk_text,
                    token_count=token_count,
                    chunk_index=i,
                    total_chunks=total,
                    metadata={
                        **doc.metadata,
                        "language": doc.language,
                        "source_word_count": doc.word_count,
                    },
                    content_hash=hashlib.sha256(chunk_text.encode()).hexdigest(),
                )
            )

        return results

    def chunk_batch(self, docs: Sequence[ProcessedDocument]) -> list[DocumentChunk]:
        """Chunk a batch of documents."""
        all_chunks: list[DocumentChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # --- Chunking Strategies ---

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _recursive_chunk(self, text: str) -> list[str]:
        """Recursively split text by decreasing separator granularity."""
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        return self._recursive_split(text, separators, 0)

    def _recursive_split(self, text: str, separators: list[str], depth: int) -> list[str]:
        """Core recursive splitting logic."""
        if self._count_tokens(text) <= self.config.chunk_size:
            return [text.strip()] if text.strip() else []

        if depth >= len(separators):
            # Force-split at token level
            return self._force_split(text)

        sep = separators[depth]
        parts = text.split(sep)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if self._count_tokens(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                if current.strip():
                    # Current chunk is ready
                    if self._count_tokens(current) <= self.config.chunk_size:
                        chunks.append(current.strip())
                    else:
                        # Recurse deeper on oversized chunk
                        chunks.extend(self._recursive_split(current, separators, depth + 1))
                current = part

        if current.strip():
            if self._count_tokens(current) <= self.config.chunk_size:
                chunks.append(current.strip())
            else:
                chunks.extend(self._recursive_split(current, separators, depth + 1))

        # Apply overlap
        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _sliding_window_chunk(self, text: str) -> list[str]:
        """Fixed-size sliding window chunking at token level."""
        tokens = self._encoder.encode(text)
        chunk_size = self.config.chunk_size
        step = chunk_size - self.config.chunk_overlap

        chunks: list[str] = []
        for i in range(0, len(tokens), max(1, step)):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self._encoder.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if i + chunk_size >= len(tokens):
                break

        return chunks

    def _sentence_chunk(self, text: str) -> list[str]:
        """Split text into sentence-based chunks."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}" if current else sentence
            if self._count_tokens(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if self._count_tokens(sentence) > self.config.chunk_size:
                    # Split long sentence
                    chunks.extend(self._force_split(sentence))
                    current = ""
                else:
                    current = sentence

        if current.strip():
            chunks.append(current.strip())

        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _semantic_chunk(self, text: str) -> list[str]:
        """Split by paragraph boundaries (semantic-ish without embeddings)."""
        paragraphs = re.split(r"\n\n+", text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            candidate = f"{current}\n\n{para}" if current else para
            if self._count_tokens(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if self._count_tokens(para) > self.config.chunk_size:
                    # Recursively split large paragraphs
                    chunks.extend(self._recursive_chunk(para))
                    current = ""
                else:
                    current = para

        if current.strip():
            chunks.append(current.strip())

        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _force_split(self, text: str) -> list[str]:
        """Force-split text at token boundaries when no separator works."""
        tokens = self._encoder.encode(text)
        chunks: list[str] = []
        for i in range(0, len(tokens), self.config.chunk_size):
            chunk_tokens = tokens[i : i + self.config.chunk_size]
            decoded = self._encoder.decode(chunk_tokens).strip()
            if decoded:
                chunks.append(decoded)
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap tokens from previous chunk to current chunk."""
        if len(chunks) <= 1:
            return chunks

        overlap_tokens = self.config.chunk_overlap
        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_tokens = self._encoder.encode(chunks[i - 1])
            overlap_text = ""
            if len(prev_tokens) > overlap_tokens:
                overlap_text = self._encoder.decode(prev_tokens[-overlap_tokens:])
            else:
                overlap_text = chunks[i - 1]

            combined = f"{overlap_text} {chunks[i]}".strip()
            result.append(combined)

        return result
