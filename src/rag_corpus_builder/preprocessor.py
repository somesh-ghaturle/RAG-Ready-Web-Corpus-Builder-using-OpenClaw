"""Text preprocessing: normalisation, language filtering, and deduplication."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import xxhash
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

from rag_corpus_builder.config import PreprocessConfig
from rag_corpus_builder.models import ExtractedDocument, ProcessedDocument

logger = logging.getLogger(__name__)


class SimHasher:
    """Simple SimHash for near-duplicate detection."""

    def __init__(self, hash_bits: int = 128) -> None:
        self._bits = hash_bits

    def hash(self, text: str) -> int:
        """Compute SimHash of text using word-level shingles."""
        tokens = text.lower().split()
        if not tokens:
            return 0

        v = [0] * self._bits
        for i in range(max(1, len(tokens) - 2)):
            shingle = " ".join(tokens[i : i + 3])
            h = xxhash.xxh128(shingle.encode()).intdigest()
            for j in range(self._bits):
                if h & (1 << j):
                    v[j] += 1
                else:
                    v[j] -= 1

        fingerprint = 0
        for j in range(self._bits):
            if v[j] > 0:
                fingerprint |= 1 << j
        return fingerprint

    @staticmethod
    def similarity(hash1: int, hash2: int, bits: int = 128) -> float:
        """Compute similarity between two SimHash values (0.0 to 1.0)."""
        xor = hash1 ^ hash2
        hamming = bin(xor).count("1")
        return 1.0 - (hamming / bits)


class TextPreprocessor:
    """Clean, normalize, filter, and deduplicate extracted documents."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self._hasher = SimHasher()
        self._seen_hashes: list[tuple[int, str]] = []  # (simhash, url)
        self._exact_hashes: set[str] = set()

    def process(self, doc: ExtractedDocument) -> ProcessedDocument | None:
        """Preprocess a single document. Returns None if filtered out."""
        text = doc.main_content

        # Normalize whitespace
        if self.config.remove_extra_whitespace:
            text = self._normalize_whitespace(text)

        # Lowercase (optional)
        if self.config.lowercase:
            text = text.lower()

        # Remove very short content
        if len(text.strip()) < 50:
            logger.debug("Filtered %s — too short after cleaning", doc.url)
            return None

        # Language detection
        language = ""
        lang_confidence = 0.0
        if self.config.target_languages:
            language, lang_confidence = self._detect_language(text)
            if self.config.target_languages and language not in self.config.target_languages:
                if lang_confidence >= self.config.min_language_confidence:
                    logger.debug(
                        "Filtered %s — language %s (conf=%.2f) not in targets",
                        doc.url,
                        language,
                        lang_confidence,
                    )
                    return None
        else:
            language, lang_confidence = self._detect_language(text)

        # Exact dedup (content hash)
        if self.config.dedup_enabled and doc.content_hash in self._exact_hashes:
            logger.debug("Filtered %s — exact duplicate", doc.url)
            return None
        self._exact_hashes.add(doc.content_hash)

        # Near-duplicate detection via SimHash
        is_duplicate = False
        if self.config.dedup_enabled:
            sim_hash = self._hasher.hash(text)
            for existing_hash, existing_url in self._seen_hashes:
                similarity = SimHasher.similarity(sim_hash, existing_hash)
                if similarity >= self.config.dedup_threshold:
                    logger.debug(
                        "Near-duplicate: %s ~ %s (sim=%.3f)",
                        doc.url,
                        existing_url,
                        similarity,
                    )
                    is_duplicate = True
                    break
            self._seen_hashes.append((sim_hash, doc.url))

        if is_duplicate:
            return None

        return ProcessedDocument(
            url=doc.url,
            title=doc.title,
            clean_text=text,
            language=language,
            language_confidence=lang_confidence,
            content_hash=doc.content_hash,
            word_count=len(text.split()),
            metadata=doc.metadata,
            is_duplicate=False,
            crawled_at=doc.crawled_at,
        )

    def process_batch(self, docs: Sequence[ExtractedDocument]) -> list[ProcessedDocument]:
        """Process a batch of documents, returning only those that pass all filters."""
        results: list[ProcessedDocument] = []
        for doc in docs:
            processed = self.process(doc)
            if processed is not None:
                results.append(processed)
        return results

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces and newlines."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\t", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _detect_language(self, text: str) -> tuple[str, float]:
        """Detect language and confidence."""
        try:
            results = detect_langs(text[:2000])  # Use first 2000 chars for speed
            if results:
                top = results[0]
                return str(top.lang), float(top.prob)
        except LangDetectException:
            pass
        return "unknown", 0.0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "unique_documents": len(self._exact_hashes),
            "simhash_entries": len(self._seen_hashes),
        }
