"""Offline tests for app.rag.embeddings.

No credentials or network required.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.rag import embeddings as module
from app.rag.chunker import chunk_document, load_corpus
from app.rag.embeddings import Embedder, EmbeddingConfig


def response(vectors, indexes=None):
    if indexes is None:
        indexes = range(len(vectors))

    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in zip(indexes, vectors, strict=True)
        ]
    )


def make_embedder(*responses, **config):
    client = Mock()
    client.embeddings.create.side_effect = list(responses)

    embedder = Embedder(
        EmbeddingConfig(
            model="test-embedding-model",
            **config,
        ),
        client=client,
    )
    return embedder, client


@pytest.fixture(autouse=True)
def prevent_real_clients(monkeypatch):
    # Tests must not depend on the developer's .env configuration.
    monkeypatch.setattr(
        module.settings,
        "EMBEDDING_MODEL",
        "test-embedding-model",
    )

    constructor = Mock(
        side_effect=AssertionError("Unexpected real client")
    )
    monkeypatch.setattr(module, "OpenAI", constructor)


def test_batches_preserve_order_and_use_configured_model():
    embedder, client = make_embedder(
        response([[3, 4], [0, 5]]),
        response([[5, 0]]),
        batch_size=2,
    )

    vectors = embedder.embed_texts(["one", "two", "three"])

    np.testing.assert_allclose(
        vectors,
        [[0.6, 0.8], [0, 1], [1, 0]],
        atol=1e-6,
    )
    assert vectors.dtype == np.float32
    assert embedder.dimension == 2
    assert client.embeddings.create.call_count == 2

    first_call = client.embeddings.create.call_args_list[0]
    assert first_call.kwargs == {
        "model": "test-embedding-model",
        "input": ["one", "two"],
        "encoding_format": "float",
    }

    second_call = client.embeddings.create.call_args_list[1]
    assert second_call.kwargs["input"] == ["three"]


def test_response_indexes_restore_input_order():
    embedder, _ = make_embedder(
        response(
            [[0, 4], [3, 0]],
            indexes=[1, 0],
        )
    )

    np.testing.assert_allclose(
        embedder.embed_texts(["first", "second"]),
        [[1, 0], [0, 1]],
    )


def test_query_returns_one_dimensional_vector():
    embedder, client = make_embedder(
        response([[3, 4]])
    )

    vector = embedder.embed_query("warfarin contraindications")

    assert vector.shape == (2,)
    np.testing.assert_allclose(
        vector,
        [0.6, 0.8],
        atol=1e-6,
    )
    assert client.embeddings.create.call_args.kwargs["input"] == [
        "warfarin contraindications"
    ]


def test_chunks_use_contextual_embedding_text_and_preserve_links():
    _, sections, chunks = chunk_document(
        "dailymed-warfarin.md",
        "# Warfarin\n\n"
        "## 4 CONTRAINDICATIONS\n\n"
        "Example passage.",
    )
    original_ids = [chunk.id for chunk in chunks]
    embedder, client = make_embedder(
        response([[1, 2]])
    )

    vectors = embedder.embed_chunks(chunks)

    assert vectors.shape == (1, 2)

    submitted = client.embeddings.create.call_args.kwargs["input"]
    assert submitted == [chunk.embed_text for chunk in chunks]
    assert submitted[0].startswith("Drug: Warfarin | ")
    assert chunks[0].section_id == sections[0].id
    assert [chunk.id for chunk in chunks] == original_ids


def test_normalized_vectors_support_cosine_similarity():
    embedder, _ = make_embedder(
        response([[3, 4], [-4, 3]]),
        response([[3, 4]]),
    )

    documents = embedder.embed_texts(["matching", "unrelated"])
    query = embedder.embed_query("question")
    scores = documents @ query

    np.testing.assert_allclose(
        scores,
        [1, 0],
        atol=1e-6,
    )


def test_empty_input_requires_no_client_or_api_key(monkeypatch):
    monkeypatch.setattr(module.settings, "API_KEY", None)
    embedder = Embedder()

    result = embedder.embed_texts([])

    assert result.shape == (0, 0)
    assert result.dtype == np.float32
    module.OpenAI.assert_not_called()


def test_empty_input_after_embedding_retains_dimension():
    embedder, client = make_embedder(
        response([[1, 2, 3]])
    )

    embedder.embed_query("question")

    assert embedder.embed_chunks([]).shape == (0, 3)
    assert client.embeddings.create.call_count == 1


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_blank_input_rejected_before_api_call(text):
    embedder, client = make_embedder()

    with pytest.raises(ValueError, match="blank"):
        embedder.embed_texts(["valid", text])

    client.embeddings.create.assert_not_called()


@pytest.mark.parametrize(
    "texts",
    ["single string", b"bytes", [None], [123]],
)
def test_wrong_input_types_are_rejected(texts):
    embedder, client = make_embedder()

    with pytest.raises(TypeError):
        embedder.embed_texts(texts)

    client.embeddings.create.assert_not_called()


def test_utf8_byte_limit_is_enforced_before_any_request():
    embedder, client = make_embedder(
        max_input_bytes=4
    )

    with pytest.raises(ValueError, match="max_input_bytes"):
        embedder.embed_texts(["valid", "\u00e9\u00e9\u00e9"])

    client.embeddings.create.assert_not_called()


def test_input_at_byte_limit_is_accepted():
    embedder, client = make_embedder(
        response([[1, 2]]),
        max_input_bytes=4,
    )

    embedder.embed_query("\u00e9\u00e9")

    client.embeddings.create.assert_called_once()


@pytest.mark.parametrize(
    "bad_response",
    [
        response([]),
        response(
            [[1, 2], [3, 4]],
            indexes=[0, 0],
        ),
        response(
            [[1, 2], [3, 4]],
            indexes=[0, 2],
        ),
        response(
            [[1, 2], [3, 4]],
            indexes=[-1, 1],
        ),
        response(
            [[1, 2], [3, 4]],
            indexes=[False, 1],
        ),
        response([[1], [1, 2]]),
        response([[], []]),
        response([[float("nan"), 1], [1, 2]]),
        response([[float("inf"), 1], [1, 2]]),
        response([[0, 0], [1, 2]]),
        response([["invalid", 1], [1, 2]]),
    ],
)
def test_invalid_api_responses_are_rejected(bad_response):
    embedder, _ = make_embedder(bad_response)

    with pytest.raises(ValueError):
        embedder.embed_texts(["one", "two"])


def test_dimension_change_within_batches_is_rejected():
    embedder, _ = make_embedder(
        response([[1, 2]]),
        response([[1, 2, 3]]),
        batch_size=1,
    )

    with pytest.raises(ValueError, match="dimension"):
        embedder.embed_texts(["one", "two"])

    assert embedder.dimension is None


def test_query_dimension_must_match_corpus_dimension():
    embedder, _ = make_embedder(
        response([[1, 2]]),
        response([[1, 2, 3]]),
    )

    embedder.embed_texts(["document"])

    with pytest.raises(ValueError, match="dimension"):
        embedder.embed_query("question")

    assert embedder.dimension == 2


def test_large_finite_values_normalize_without_overflow():
    embedder, _ = make_embedder(
        response([[1e308, 1e308]])
    )

    vector = embedder.embed_query("question")

    assert np.isfinite(vector).all()
    np.testing.assert_allclose(
        np.linalg.norm(vector),
        1,
        atol=1e-6,
    )


def test_api_errors_propagate_without_returning_partial_results():
    embedder, _ = make_embedder(
        response([[1, 2]]),
        RuntimeError("upstream unavailable"),
        batch_size=1,
    )

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        embedder.embed_texts(["one", "two"])

    assert embedder.dimension is None


def test_default_model_comes_from_project_config(monkeypatch):
    monkeypatch.setattr(
        module.settings,
        "EMBEDDING_MODEL",
        "proxy-model",
    )

    assert EmbeddingConfig().model == "proxy-model"


@pytest.mark.parametrize("model", [None, "", " ", "\n\t"])
def test_missing_model_is_rejected(monkeypatch, model):
    monkeypatch.setattr(
        module.settings,
        "EMBEDDING_MODEL",
        model,
    )

    with pytest.raises(ValueError, match="OPENAI_EMBEDDING_MODEL"):
        EmbeddingConfig()

    module.OpenAI.assert_not_called()


def test_client_is_lazy_and_uses_project_settings(monkeypatch):
    monkeypatch.setattr(
        module.settings,
        "API_KEY",
        "test-key",
    )
    monkeypatch.setattr(
        module.settings,
        "BASE_URL",
        "https://proxy.example/v1",
    )

    client = Mock()
    client.embeddings.create.return_value = response([[1, 2]])

    constructor = Mock(return_value=client)
    monkeypatch.setattr(module, "OpenAI", constructor)

    embedder = Embedder()
    constructor.assert_not_called()

    embedder.embed_query("first")
    embedder.embed_query("second")

    constructor.assert_called_once_with(
        api_key="test-key",
        base_url="https://proxy.example/v1",
        timeout=30.0,
        max_retries=2,
    )


def test_missing_api_key_has_clear_error(monkeypatch):
    monkeypatch.setattr(
        module.settings,
        "API_KEY",
        None,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Embedder().embed_query("question")

    module.OpenAI.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"batch_size": 0},
        {"batch_size": 33},
        {"batch_size": True},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"max_retries": -1},
        {"max_input_bytes": 0},
        {"max_input_bytes": 8001},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EmbeddingConfig(**kwargs)


def test_every_corpus_chunk_fits_embedding_byte_limit():
    # This test file belongs at the project root.
    corpus_dir = Path(__file__).resolve().parents[1] / "corpus"
    corpus = load_corpus(corpus_dir)

    limit = EmbeddingConfig(
        model="test-embedding-model"
    ).max_input_bytes

    oversized = [
        (chunk.id, len(chunk.embed_text.encode("utf-8")))
        for chunk in corpus.chunks
        if len(chunk.embed_text.encode("utf-8")) > limit
    ]

    assert corpus.chunks
    assert not oversized, (
        f"Chunks exceeding {limit} bytes: {oversized}"
    )