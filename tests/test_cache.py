"""Offline cache tests.

Temporary cache files are isolated by pytest.
No real API calls.
"""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.rag import cache as module
from app.rag.cache import CachedEmbedder
from app.rag.chunker import Corpus, chunk_document
from app.rag.embeddings import EmbeddingConfig
from app.rag.hybrid import HybridRetriever


def make_embedder(
    cache_dir,
    *,
    model="test-model",
    endpoint="https://proxy.example/v1",
    revision="1",
    max_input_bytes=8000,
    batch_size=32,
):
    client = Mock()
    client.base_url = endpoint

    def create(**kwargs):
        data = []

        for index, text in enumerate(kwargs["input"]):
            values = text.encode("utf-8")

            # Deterministic vectors without an API or random seed.
            vector = [
                float(sum(values) % 11 + 1),
                float(len(values) % 7 + 1),
                1.0,
            ]

            data.append(
                SimpleNamespace(
                    index=index,
                    embedding=vector,
                )
            )

        return SimpleNamespace(data=data)

    client.embeddings.create.side_effect = create

    embedder = CachedEmbedder(
        EmbeddingConfig(
            model=model,
            max_input_bytes=max_input_bytes,
            batch_size=batch_size,
        ),
        cache_dir=cache_dir,
        client=client,
        revision=revision,
    )

    return embedder, client


@pytest.fixture(autouse=True)
def prevent_real_clients(monkeypatch):
    monkeypatch.setattr(
        module,
        "OpenAI",
        Mock(
            side_effect=AssertionError("Real client")
        ),
    )

    # Block the constructor used by the parent Embedder as well.
    monkeypatch.setattr(
        "app.rag.embeddings.OpenAI",
        Mock(
            side_effect=AssertionError("Real client")
        ),
    )


def test_first_call_embeds_and_second_call_hits_cache(tmp_path):
    embedder, client = make_embedder(tmp_path)

    first = embedder.embed_texts(["alpha", "beta"])
    second = embedder.embed_texts(["alpha", "beta"])

    np.testing.assert_array_equal(first, second)

    assert first.dtype == np.float32
    assert client.embeddings.create.call_count == 1
    assert embedder.dimension == 3


def test_cache_survives_a_new_embedder_instance(tmp_path):
    first, _ = make_embedder(tmp_path)
    expected = first.embed_texts(["alpha", "beta"])

    second, client = make_embedder(tmp_path)

    np.testing.assert_array_equal(
        second.embed_texts(["alpha", "beta"]),
        expected,
    )

    client.embeddings.create.assert_not_called()
    assert second.dimension == 3


def test_only_changed_or_new_text_is_embedded(tmp_path):
    embedder, client = make_embedder(tmp_path)

    embedder.embed_texts(["alpha", "beta"])
    client.embeddings.create.reset_mock()

    embedder.embed_texts([
        "beta",
        "alpha changed",
        "gamma",
    ])

    assert (
        client.embeddings.create.call_args.kwargs["input"]
        == ["alpha changed", "gamma"]
    )


def test_order_and_duplicate_rows_are_preserved(tmp_path):
    embedder, client = make_embedder(tmp_path)

    vectors = embedder.embed_texts([
        "beta",
        "alpha",
        "beta",
    ])

    assert (
        client.embeddings.create.call_args.kwargs["input"]
        == ["beta", "alpha"]
    )

    np.testing.assert_array_equal(
        vectors[0],
        vectors[2],
    )

    reordered = embedder.embed_texts([
        "alpha",
        "beta",
    ])

    np.testing.assert_array_equal(
        reordered[0],
        vectors[1],
    )
    np.testing.assert_array_equal(
        reordered[1],
        vectors[0],
    )


def test_exact_text_changes_include_whitespace(tmp_path):
    embedder, client = make_embedder(tmp_path)

    embedder.embed_texts(["alpha"])
    embedder.embed_texts(["alpha "])

    assert client.embeddings.create.call_count == 2


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other-model"},
        {"endpoint": "https://other.example/v1"},
        {"revision": "2"},
    ],
)
def test_model_endpoint_and_revision_isolate_cache(tmp_path, change):
    first, _ = make_embedder(tmp_path)
    first.embed_query("alpha")

    second, client = make_embedder(
        tmp_path,
        **change,
    )
    second.embed_query("alpha")

    client.embeddings.create.assert_called_once()


def test_endpoint_trailing_slash_does_not_invalidate_cache(tmp_path):
    first, _ = make_embedder(tmp_path)
    expected = first.embed_query("alpha")

    second, client = make_embedder(
        tmp_path,
        endpoint="https://proxy.example/v1/",
    )

    np.testing.assert_array_equal(
        second.embed_query("alpha"),
        expected,
    )

    client.embeddings.create.assert_not_called()


def test_query_embeddings_are_cached(tmp_path):
    embedder, client = make_embedder(tmp_path)

    first = embedder.embed_query("question")
    second = embedder.embed_query("question")

    assert first.shape == (3,)
    np.testing.assert_array_equal(first, second)

    client.embeddings.create.assert_called_once()


def test_chunk_api_uses_embed_text(tmp_path):
    _, _, chunks = chunk_document(
        "dailymed-warfarin.md",
        "# Warfarin\n## Warnings\nPassage.",
    )

    embedder, client = make_embedder(tmp_path)
    embedder.embed_chunks(chunks)

    assert (
        client.embeddings.create.call_args.kwargs["input"]
        == [chunks[0].embed_text]
    )

    # IDs can change without invalidating identical embedding input.
    embedder.embed_chunks([
        replace(
            chunks[0],
            id="different-id",
        )
    ])

    client.embeddings.create.assert_called_once()


def test_existing_batching_is_preserved(tmp_path):
    embedder, client = make_embedder(
        tmp_path,
        batch_size=2,
    )

    embedder.embed_texts([
        "alpha",
        "beta",
        "gamma",
    ])

    assert client.embeddings.create.call_count == 2


def test_empty_input_does_not_create_files_or_call_api(tmp_path):
    directory = tmp_path / "unused"
    embedder, client = make_embedder(directory)

    assert embedder.embed_texts([]).shape == (0, 0)
    assert not directory.exists()

    client.embeddings.create.assert_not_called()


def test_empty_input_retains_cached_dimension(tmp_path):
    first, _ = make_embedder(tmp_path)
    first.embed_query("alpha")

    second, _ = make_embedder(tmp_path)
    second.embed_query("alpha")

    assert second.embed_chunks([]).shape == (0, 3)


@pytest.mark.parametrize(
    "texts",
    [
        "alpha",
        b"alpha",
        [None],
        [123],
    ],
)
def test_invalid_input_types_are_rejected(tmp_path, texts):
    embedder, client = make_embedder(tmp_path)

    with pytest.raises(TypeError):
        embedder.embed_texts(texts)

    client.embeddings.create.assert_not_called()


@pytest.mark.parametrize(
    "text",
    ["", " ", "\n\t"],
)
def test_blank_inputs_are_rejected(tmp_path, text):
    embedder, client = make_embedder(tmp_path)

    with pytest.raises(ValueError, match="blank"):
        embedder.embed_texts(["alpha", text])

    client.embeddings.create.assert_not_called()


def test_all_input_is_validated_before_requests(tmp_path):
    embedder, client = make_embedder(
        tmp_path,
        max_input_bytes=4,
    )

    with pytest.raises(ValueError, match="max_input_bytes"):
        embedder.embed_texts([
            "ok",
            "\u00e9\u00e9\u00e9",
        ])

    client.embeddings.create.assert_not_called()


def test_cache_hit_cannot_bypass_a_stricter_input_limit(tmp_path):
    first, _ = make_embedder(tmp_path)
    first.embed_query("alpha")

    second, client = make_embedder(
        tmp_path,
        max_input_bytes=4,
    )

    with pytest.raises(ValueError, match="max_input_bytes"):
        second.embed_query("alpha")

    client.embeddings.create.assert_not_called()


@pytest.mark.parametrize(
    "bad",
    [
        "{invalid",
        "[]",
        '{"version": 999}',
    ],
)
def test_malformed_cache_is_recomputed(tmp_path, bad):
    embedder, client = make_embedder(tmp_path)
    expected = embedder.embed_query("alpha")

    path = embedder._path_for("alpha")
    path.write_text(bad, encoding="utf-8")

    client.embeddings.create.reset_mock()

    np.testing.assert_array_equal(
        embedder.embed_query("alpha"),
        expected,
    )

    client.embeddings.create.assert_called_once()

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    assert payload["version"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"key": "wrong"},
        {"namespace": "wrong"},
        {"dimension": 99},
        {"vector": [0, 0, 0]},
        {"vector": [2, 0, 0]},
        {"vector": ["a", "b", "c"]},
        {"vector": [float("nan"), 0, 1]},
    ],
)
def test_invalid_cached_vectors_are_recomputed(tmp_path, change):
    embedder, client = make_embedder(tmp_path)
    embedder.embed_query("alpha")

    path = embedder._path_for("alpha")
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    payload.update(change)

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    client.embeddings.create.reset_mock()
    embedder.embed_query("alpha")

    client.embeddings.create.assert_called_once()


def test_inconsistent_cached_dimensions_fail_explicitly(tmp_path):
    embedder, _ = make_embedder(tmp_path)
    embedder.embed_texts(["alpha", "beta"])

    path = embedder._path_for("beta")
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    payload.update(
        dimension=2,
        vector=[1, 0],
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    fresh, client = make_embedder(tmp_path)

    with pytest.raises(ValueError, match="dimensions"):
        fresh.embed_texts(["alpha", "beta"])

    client.embeddings.create.assert_not_called()


def test_api_dimension_must_match_cached_vectors(tmp_path):
    first, _ = make_embedder(tmp_path)
    first.embed_query("alpha")

    second, client = make_embedder(tmp_path)
    client.embeddings.create.side_effect = None
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(
                index=0,
                embedding=[1, 0],
            )
        ]
    )

    with pytest.raises(ValueError, match="dimension"):
        second.embed_texts(["alpha", "new"])


def test_api_failure_does_not_persist_partial_batches(tmp_path):
    embedder, client = make_embedder(
        tmp_path,
        batch_size=1,
    )

    normal_create = client.embeddings.create.side_effect

    def create(**kwargs):
        if kwargs["input"] == ["beta"]:
            raise RuntimeError("unavailable")

        return normal_create(**kwargs)

    client.embeddings.create.side_effect = create

    with pytest.raises(RuntimeError, match="unavailable"):
        embedder.embed_texts(["alpha", "beta"])

    assert not embedder._path_for("alpha").exists()
    assert not embedder._path_for("beta").exists()


def test_atomic_write_replaces_destination_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    embedder, _ = make_embedder(tmp_path)
    original_replace = module.os.replace
    calls = []

    def tracking_replace(source, destination):
        calls.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(
        module.os,
        "replace",
        tracking_replace,
    )

    embedder.embed_query("alpha")

    assert len(calls) == 1

    temporary, final = calls[0]

    assert temporary.parent == final.parent
    assert not temporary.exists()
    assert final.exists()


def test_failed_replace_cleans_temp_file(tmp_path, monkeypatch):
    embedder, _ = make_embedder(tmp_path)
    temporary_paths = []

    def fail_replace(source, destination):
        temporary_paths.append(source)
        raise OSError("disk failure")

    monkeypatch.setattr(
        module.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(OSError, match="disk failure"):
        embedder.embed_query("alpha")

    assert temporary_paths
    assert all(
        not path.exists()
        for path in temporary_paths
    )
    assert not embedder._path_for("alpha").exists()


def test_cache_contains_no_plaintext_input_or_api_key(tmp_path):
    embedder, client = make_embedder(tmp_path)

    client.api_key = "SECRET-API-KEY"
    text = "UNIQUE-PLAINTEXT-PASSAGE"

    embedder.embed_query(text)

    path = embedder._path_for(text)
    contents = path.read_text(encoding="utf-8")

    assert text not in contents
    assert client.api_key not in contents
    assert text not in str(path)
    assert client.api_key not in str(path)


def test_model_change_requires_a_new_instance(tmp_path):
    embedder, client = make_embedder(tmp_path)
    embedder.config = EmbeddingConfig(
        model="other-model"
    )

    with pytest.raises(
        ValueError,
        match="new CachedEmbedder",
    ):
        embedder.embed_query("alpha")

    client.embeddings.create.assert_not_called()


def test_endpoint_change_requires_a_new_instance(tmp_path):
    embedder, client = make_embedder(tmp_path)
    client.base_url = "https://different.example/v1"

    with pytest.raises(
        ValueError,
        match="new CachedEmbedder",
    ):
        embedder.embed_query("alpha")

    client.embeddings.create.assert_not_called()


@pytest.mark.parametrize(
    "revision",
    ["", " ", None],
)
def test_invalid_revision_is_rejected(tmp_path, revision):
    with pytest.raises(ValueError, match="revision"):
        make_embedder(
            tmp_path,
            revision=revision,
        )


def test_cached_hits_can_work_without_api_credentials(
    tmp_path,
    monkeypatch,
):
    first, _ = make_embedder(tmp_path)
    expected = first.embed_query("alpha")

    monkeypatch.setattr(
        module.settings,
        "API_KEY",
        None,
    )
    monkeypatch.setattr(
        module.settings,
        "BASE_URL",
        "https://proxy.example/v1",
    )

    second = CachedEmbedder(
        EmbeddingConfig(model="test-model"),
        cache_dir=tmp_path,
    )

    np.testing.assert_array_equal(
        second.embed_query("alpha"),
        expected,
    )


def test_hybrid_build_reuses_cached_chunks_after_restart(tmp_path):
    doc, sections, chunks = chunk_document(
        "dailymed-warfarin.md",
        "# Warfarin\n"
        "## 4 CONTRAINDICATIONS\n"
        "Example passage.",
    )
    corpus = Corpus([doc], sections, chunks)

    first, _ = make_embedder(tmp_path)
    HybridRetriever.build(corpus, first)

    second, client = make_embedder(tmp_path)
    retriever = HybridRetriever.build(
        corpus,
        second,
    )

    client.embeddings.create.assert_not_called()

    assert retriever.search(
        "warfarin",
        drugs=["warfarin"],
    )
    assert client.embeddings.create.call_count == 1

    retriever.search(
        "warfarin",
        drugs=["warfarin"],
    )
    assert client.embeddings.create.call_count == 1