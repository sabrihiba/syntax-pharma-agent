"""Embedding generation for the pharmacology corpus.

Embed Chunk.embed_text, preserve input order, and return normalized float32
vectors suitable for cosine similarity through a matrix-vector product.
No client or API request is created at import time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI

from app import config as settings
from app.rag.chunker import Chunk


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = field(
        default_factory=lambda: settings.EMBEDDING_MODEL
    )
    batch_size: int = 32
    timeout: float = 30.0
    max_retries: int = 2
    max_input_bytes: int = 8000

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(
                "OPENAI_EMBEDDING_MODEL must be configured "
                "as a nonempty string"
            )

        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 32:
            raise ValueError(
                "batch_size must be an integer from 1 to 32"
            )

        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")

        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError(
                "max_retries must be a nonnegative integer"
            )

        if (
            type(self.max_input_bytes) is not int
            or not 1 <= self.max_input_bytes <= 8000
        ):
            raise ValueError(
                "max_input_bytes must be an integer from 1 to 8000"
            )


class Embedder:
    """Use one instance for both corpus chunks and queries.

    A custom client can be injected for tests or another compatible provider.
    The byte guard is a conservative application limit, not a token counter.
    Oversized passages are rejected, never silently truncated.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self._client = client
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not settings.API_KEY or not settings.API_KEY.strip():
                raise ValueError("OPENAI_API_KEY is missing")

            self._client = OpenAI(
                api_key=settings.API_KEY,
                base_url=settings.BASE_URL or None,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )

        return self._client

    def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        """Row i corresponds to chunks[i]; parent sections are not embedded."""
        return self.embed_texts([chunk.embed_text for chunk in chunks])

    def embed_query(self, question: str) -> np.ndarray:
        """Return a one-dimensional vector for one retrieval query."""
        return self.embed_texts([question])[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return a matrix shaped (number_of_texts, embedding_dimension).

        Empty input returns (0, known_dimension), or (0, 0) before the first
        successful embedding. API errors propagate to the caller; the SDK
        handles transient retries using max_retries.
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")

        texts = list(texts)

        # Validate every input before making the first API request.
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string")

            if not text.strip():
                raise ValueError(f"texts[{index}] must not be blank")

            if len(text.encode("utf-8")) > self.config.max_input_bytes:
                raise ValueError(
                    f"texts[{index}] exceeds max_input_bytes="
                    f"{self.config.max_input_bytes}; split the passage "
                    "before embedding"
                )

        if not texts:
            return np.empty(
                (0, self._dimension or 0),
                dtype=np.float32,
            )

        matrices: list[np.ndarray] = []
        dimension = self._dimension
        client = self._get_client()

        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start:start + self.config.batch_size]

            response = client.embeddings.create(
                model=self.config.model,
                input=batch,
                encoding_format="float",
            )
            matrix = self._read_response(response, len(batch))

            if dimension is None:
                dimension = matrix.shape[1]
            elif matrix.shape[1] != dimension:
                raise ValueError(
                    "embedding dimension changed between requests"
                )

            matrices.append(matrix)

        result = np.concatenate(matrices, axis=0)
        self._dimension = dimension
        return result

    @staticmethod
    def _read_response(response, expected_rows: int) -> np.ndarray:
        data = list(response.data)

        if len(data) != expected_rows:
            raise ValueError(
                "embedding response has an unexpected row count"
            )

        rows = [None] * expected_rows
        seen: set[int] = set()

        # Match vectors to their inputs using the API response indexes.
        for item in data:
            index = item.index

            if (
                type(index) is not int
                or not 0 <= index < expected_rows
                or index in seen
            ):
                raise ValueError(
                    "embedding response has invalid or duplicate indexes"
                )

            seen.add(index)
            rows[index] = item.embedding

        try:
            matrix = np.asarray(rows, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "embedding response contains invalid vectors"
            ) from exc

        if (
            matrix.ndim != 2
            or matrix.shape[0] != expected_rows
            or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                "embedding response contains invalid vectors"
            )

        # Scale first to avoid overflow when calculating vector lengths.
        scales = np.max(np.abs(matrix), axis=1, keepdims=True)

        if np.any(scales == 0):
            raise ValueError(
                "embedding response contains a zero vector"
            )

        matrix = matrix / scales
        matrix = matrix / np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        return matrix.astype(np.float32)