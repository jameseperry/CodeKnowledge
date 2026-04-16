"""Embedding backends — local (sentence-transformers) or remote (HTTP API).

Provides a unified interface for embedding text, with support for
Nomic-style search_document/search_query prefixing.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import EmbeddingConfig

log = logging.getLogger(__name__)


class Embedder(ABC):
    """Base class for embedding backends."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts for indexing.

        Returns an (N, dim) float32 ndarray.
        """

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query.

        Returns a (dim,) float32 ndarray.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""


class LocalEmbedder(Embedder):
    """Embedding via a local sentence-transformers model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Suppress all logging noise during model load (HF, httpx, torch)
            root_logger = logging.getLogger()
            saved_root_level = root_logger.level
            root_logger.setLevel(logging.ERROR)

            try:
                # Try local cache first to avoid HF network round-trips
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        self._model = SentenceTransformer(
                            self._config.model_name,
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                    except OSError:
                        # Model not cached yet — download it
                        root_logger.setLevel(saved_root_level)
                        log.info("Downloading embedding model: %s", self._config.model_name)
                        root_logger.setLevel(logging.ERROR)
                        self._model = SentenceTransformer(
                            self._config.model_name,
                            trust_remote_code=True,
                        )
            finally:
                root_logger.setLevel(saved_root_level)

            log.info("Embedding model loaded: %s", self._config.model_name)

        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"search_document: {text}" for text in texts]
        model = self._get_model()
        return model.encode(
            prefixed, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        prefixed = [f"search_query: {text}"]
        model = self._get_model()
        result = model.encode(
            prefixed, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return result[0]

    @property
    def dimensions(self) -> int:
        return self._config.dimensions


class APIEmbedder(Embedder):
    """Embedding via a remote HTTP API (OpenAI-compatible embeddings endpoint).

    Splits large batches into chunks of ``batch_size`` and dispatches up to
    ``max_concurrent`` requests in parallel.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._client = None
        self._url = config.api_url
        self._headers = self._parse_headers()
        self._batch_size = config.batch_size
        self._max_concurrent = config.max_concurrent

    @staticmethod
    def _parse_headers() -> dict[str, str]:
        headers: dict[str, str] = {}
        raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
        for pair in raw.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                headers[key.strip()] = value.strip()
        return headers

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=60.0)
            log.info("Remote embedding client ready: %s", self._url)
        return self._client

    def _post_batch(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch to the API and return ordered embeddings."""
        client = self._get_client()
        last_exc = None
        for attempt in range(3):
            resp = client.post(self._url, json={"input": texts}, headers=self._headers)
            if resp.status_code == 422:
                log.warning(
                    "Embedding API 422 on batch of %d texts. Truncating and retrying.",
                    len(texts),
                )
                texts = [t[:8000] for t in texts]
                resp = client.post(self._url, json={"input": texts}, headers=self._headers)
                if resp.status_code == 422:
                    log.error("Embedding API 422 persists after truncation.")
                    resp.raise_for_status()
            if resp.status_code >= 500:
                last_exc = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                wait = 2 ** attempt
                log.warning("Embedding API %d, retrying in %ds...", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            results = sorted(data["data"], key=lambda d: d["index"])
            return [r["embedding"] for r in results]
        raise last_exc  # type: ignore[misc]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        if len(batches) == 1:
            return self._post_batch(batches[0])

        all_vectors: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
            futures = [pool.submit(self._post_batch, batch) for batch in batches]
            for future in futures:
                all_vectors.extend(future.result())
        return all_vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"search_document: {text}" for text in texts]
        vectors = self._embed(prefixed)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefixed = [f"search_query: {text}"]
        vectors = self._embed(prefixed)
        return np.array(vectors[0], dtype=np.float32)

    @property
    def dimensions(self) -> int:
        return self._config.dimensions


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None


def get_embedder(config: EmbeddingConfig | None = None) -> Embedder:
    """Return the singleton embedder, creating it from config if needed."""
    global _embedder
    if _embedder is None:
        if config is None:
            config = EmbeddingConfig()
        if config.backend == "remote":
            if not config.api_url:
                raise ValueError("embedding.api_url is required when backend is 'remote'")
            _embedder = APIEmbedder(config)
        else:
            _embedder = LocalEmbedder(config)
    return _embedder


def reset_embedder() -> None:
    """Reset the singleton (for testing or config changes)."""
    global _embedder
    _embedder = None
