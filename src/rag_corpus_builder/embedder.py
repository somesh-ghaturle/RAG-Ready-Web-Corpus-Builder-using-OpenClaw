"""Optional embedding generation for document chunks."""

from __future__ import annotations

import logging
from typing import Sequence

from rag_corpus_builder.config import EmbeddingConfig
from rag_corpus_builder.models import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for document chunks using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s (device=%s)", self.config.model_name, self.config.device)
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install 'rag-corpus-builder[embeddings]'"
            )

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> list[DocumentChunk]:
        """Generate embeddings for a list of chunks. Returns updated chunks."""
        if not self.config.enabled:
            return list(chunks)

        self._load_model()
        assert self._model is not None

        texts = [chunk.text for chunk in chunks]
        logger.info("Generating embeddings for %d chunks (batch_size=%d)", len(texts), self.config.batch_size)

        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.config.normalize,
        )

        result: list[DocumentChunk] = []
        for chunk, embedding in zip(chunks, embeddings):
            updated = chunk.model_copy()
            updated.embedding = embedding.tolist()
            result.append(updated)

        logger.info("Generated %d embeddings (dim=%d)", len(result), len(result[0].embedding) if result else 0)
        return result

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        self._load_model()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()
