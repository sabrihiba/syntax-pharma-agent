"""Persistent embedding cache compatible with the existing Embedder.

Cache identity: model + endpoint + revision + exact UTF-8 input text.
Files contain vectors and hashes, never API keys or plaintext input text.

An endpoint/model alias may change upstream; bump revision to invalidate it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from openai import OpenAI

from app import config as settings
from app.rag.embeddings import Embedder, EmbeddingConfig

logger = logging.getLogger(__name__)

CACHE_VERSION = 1

DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / ".cache"
    / "embeddings"
)


class CachedEmbedder(Embedder):
    """Cache chunk and query embeddings across application restarts.

    Inherited embed_chunks() and embed_query() use this cache.

    A fresh instance is required after changing model or endpoint.

    Separate processes may duplicate API work on simultaneous cache misses;
    atomic replacement keeps their cache writes complete.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        client: OpenAI | None = None,
        revision: str = "1",
    ) -> None:
        super().__init__(config, client=client)

        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(
                "revision must be a nonempty string"
            )

        self._model = self.config.model
        self._endpoint = self._current_endpoint()

        identity = json.dumps(
            {
                "version": CACHE_VERSION,
                "model": self._model,
                "endpoint": self._endpoint,
                "revision": revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        self._namespace = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        self._directory = Path(cache_dir) / self._namespace
        self._lock = threading.RLock()

    def _current_endpoint(self) -> str:
        if self._client is not None:
            endpoint = self._client.base_url
        else:
            endpoint = (
                settings.BASE_URL
                or "https://api.openai.com/v1"
            )

        return str(endpoint).rstrip("/")

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    def _path_for(self, text: str) -> Path:
        return (
            self._directory
            / f"{self._text_key(text)}.json"
        )

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)):
            raise TypeError(
                "texts must be a sequence of strings"
            )

        texts = list(texts)

        # Cache hits must not bypass current input validation.
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be a string"
                )

            if not text.strip():
                raise ValueError(
                    f"texts[{index}] must not be blank"
                )

            if len(text.encode("utf-8")) > self.config.max_input_bytes:
                raise ValueError(
                    f"texts[{index}] exceeds max_input_bytes="
                    f"{self.config.max_input_bytes}; split the passage "
                    "before embedding"
                )

        with self._lock:
            if (
                self.config.model != self._model
                or self._current_endpoint() != self._endpoint
            ):
                raise ValueError(
                    "model or endpoint changed; "
                    "create a new CachedEmbedder"
                )

            if not texts:
                return np.empty(
                    (0, self._dimension or 0),
                    dtype=np.float32,
                )

            # Deduplicate exact inputs, preserving first-occurrence order.
            unique_texts = list(dict.fromkeys(texts))

            vectors: dict[str, np.ndarray] = {}
            missing: list[str] = []
            dimension = self._dimension

            for text in unique_texts:
                vector = self._read_vector(text)

                if vector is None:
                    missing.append(text)
                    continue

                if dimension is None:
                    dimension = vector.size
                elif vector.size != dimension:
                    raise ValueError(
                        "cached embedding dimensions disagree; "
                        "clear this cache namespace "
                        "or use a new revision"
                    )

                vectors[text] = vector

            # The existing Embedder checks API responses against cache size.
            self._dimension = dimension

            if missing:
                generated = super().embed_texts(missing)

                for text, vector in zip(
                    missing,
                    generated,
                    strict=True,
                ):
                    vectors[text] = vector

                # Save only after all API batches have succeeded.
                for text in missing:
                    self._write_vector(
                        text,
                        vectors[text],
                    )

            result = np.stack([
                vectors[text]
                for text in texts
            ])

            self._dimension = result.shape[1]

            return result.astype(
                np.float32,
                copy=False,
            )

    def _read_vector(self, text: str) -> np.ndarray | None:
        path = self._path_for(text)

        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )

            if not isinstance(payload, dict):
                raise ValueError("invalid cache record")

            if (
                type(payload.get("version")) is not int
                or payload["version"] != CACHE_VERSION
                or payload.get("namespace") != self._namespace
                or payload.get("key") != self._text_key(text)
            ):
                raise ValueError("invalid cache identity")

            dimension = payload.get("dimension")
            raw = np.asarray(payload.get("vector"))

            if (
                type(dimension) is not int
                or dimension <= 0
                or raw.ndim != 1
                or raw.size != dimension
                or raw.dtype.kind not in "iuf"
            ):
                raise ValueError("invalid cached vector")

            vector = raw.astype(np.float64)

            if not np.isfinite(vector).all():
                raise ValueError("nonfinite cached vector")

            norm = float(np.linalg.norm(vector))

            if not math.isclose(
                norm,
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "cached vector is not normalized"
                )

            return vector.astype(np.float32)

        except FileNotFoundError:
            return None

        except (
            ValueError,
            TypeError,
            UnicodeError,
            OverflowError,
        ):
            logger.warning(
                "Ignoring invalid embedding cache entry %s",
                path.name,
            )
            return None

    def _write_vector(
        self,
        text: str,
        vector: np.ndarray,
    ) -> None:
        self._directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": CACHE_VERSION,
            "namespace": self._namespace,
            "key": self._text_key(text),
            "dimension": int(vector.size),
            "vector": vector.tolist(),
        }

        temporary: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._directory,
                prefix=".embedding-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)

                json.dump(
                    payload,
                    handle,
                    separators=(",", ":"),
                    allow_nan=False,
                )

            os.replace(
                temporary,
                self._path_for(text),
            )

        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)