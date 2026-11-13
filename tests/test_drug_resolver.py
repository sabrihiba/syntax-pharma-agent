"""Offline tests for corpus-based drug-name resolution."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.agent.drugs import (
    DrugResolver,
    normalize_drug_name,
)
from app.rag.chunker import (
    Corpus,
    chunk_document,
    load_corpus,
)


def corpus_from(*documents):
    docs, sections, chunks = [], [], []

    for filename, content in documents:
        doc, parents, children = chunk_document(
            filename,
            content,
        )
        docs.append(doc)
        sections.extend(parents)
        chunks.extend(children)

    return Corpus(docs, sections, chunks)


@pytest.fixture(scope="module")
def resolver():
    corpus = load_corpus(
        Path(__file__).resolve().parents[1] / "corpus"
    )
    return DrugResolver(corpus)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("warfarin", "warfarin"),
        (" COUMADIN ", "warfarin"),
        ("Jantoven", "warfarin"),
        ("Eliquis", "apixaban"),
        ("Eliquis Sprinkle", "apixaban"),
        ("Zoloft", "sertraline"),
        ("Lipitor", "atorvastatin"),
        ("Amoxil", "amoxicillin"),
        ("Paracetamol", "acetaminophen"),
        ("APAP", "acetaminophen"),
        ("ASA", "aspirin"),
        ("salbutamol", "albuterol"),
    ],
)
def test_explicit_corpus_names_resolve(
    resolver,
    name,
    expected,
):
    result = resolver.resolve_drug(name)

    assert result.resolved
    assert result.status == "resolved"
    assert result.drug == expected
    assert result.candidates == (expected,)
    assert result.matches


def test_registered_symbol_is_ignored(resolver):
    assert (
        resolver.resolve_drug("Eliquis\u00ae").drug
        == "apixaban"
    )


def test_generic_names_include_all_corpus_drugs(resolver):
    assert len(resolver.drugs) == 10

    for drug in resolver.drugs:
        assert resolver.resolve_drug(drug).drug == drug


def test_unknown_drug_is_not_guessed(resolver):
    result = resolver.resolve_drug("naproxen")

    assert result.status == "not_found"
    assert result.drug is None
    assert not result.resolved


def test_typo_does_not_silently_select_a_drug(resolver):
    assert (
        resolver.resolve_drug("warfrin").status
        == "not_found"
    )


def test_dose_is_not_silently_removed(resolver):
    assert (
        resolver.resolve_drug("Eliquis 5 mg").status
        == "not_found"
    )


@pytest.mark.parametrize(
    "name",
    [
        "Talicia",
        "Excedrin",
        "Janumet",
        "Dayquil",
        "Caduet",
    ],
)
def test_combination_products_are_not_single_drug_aliases(
    resolver,
    name,
):
    result = resolver.resolve_drug(name)

    assert result.status == "combination"
    assert result.is_combination
    assert not result.resolved
    assert result.drug is None
    assert result.candidates
    assert (
        "not a complete ingredient list"
        in result.message
    )


def test_combination_candidates_are_only_documented_monographs(
    resolver,
):
    result = resolver.resolve_drug("Talicia")

    assert result.candidates == ("amoxicillin",)
    assert result.drug is None


def test_provenance_is_preserved(resolver):
    result = resolver.resolve_drug("Zoloft")
    record = result.matches[0]

    assert record.kind == "brand"
    assert record.file == "medlineplus-sertraline.md"
    assert record.section_id
    assert "Zoloft" in record.original_text

    json.dumps(asdict(result))


def test_alias_collision_is_ambiguous():
    corpus = corpus_from(
        (
            "medlineplus-alpha.md",
            "# Alpha\n"
            "## Brand names\n"
            "- SharedBrand",
        ),
        (
            "medlineplus-beta.md",
            "# Beta\n"
            "## Brand names\n"
            "- SharedBrand",
        ),
    )

    result = DrugResolver(corpus).resolve_drug(
        "SharedBrand"
    )

    assert result.status == "ambiguous"
    assert result.candidates == ("alpha", "beta")
    assert result.drug is None


def test_combination_in_plain_brand_section_is_detected():
    corpus = corpus_from(
        (
            "medlineplus-alpha.md",
            "# Alpha\n"
            "## Brand names\n"
            "- Combo\u00ae (containing Alpha, Beta)",
        )
    )

    result = DrugResolver(corpus).resolve_drug("Combo")

    assert result.status == "combination"


def test_combination_heading_is_respected_without_parentheses():
    corpus = corpus_from(
        (
            "medlineplus-alpha.md",
            "# Alpha\n"
            "## Brand names of combination products\n"
            "- Combo",
        )
    )

    result = DrugResolver(corpus).resolve_drug("Combo")

    assert result.is_combination


def test_other_body_mentions_do_not_become_aliases():
    corpus = corpus_from(
        (
            "medlineplus-alpha.md",
            "# Alpha\n"
            "## Precautions\n"
            "- Beta may interact with alpha.",
        )
    )

    result = DrugResolver(corpus).resolve_drug("Beta")

    assert result.status == "not_found"


def test_aliases_are_not_inferred_from_dailymed_prose():
    corpus = corpus_from(
        (
            "dailymed-alpha.md",
            "# Alpha\n"
            "## Brand names\n"
            "- UnverifiedBrand",
        )
    )

    result = DrugResolver(corpus).resolve_drug(
        "UnverifiedBrand"
    )

    assert result.status == "not_found"


def test_formulation_names_are_not_collapsed():
    corpus = corpus_from(
        (
            "medlineplus-alpha.md",
            "# Alpha\n"
            "## Brand names\n"
            "- Example XR",
        )
    )

    resolver = DrugResolver(corpus)

    assert (
        resolver.resolve_drug("Example XR").drug
        == "alpha"
    )
    assert (
        resolver.resolve_drug("Example").status
        == "not_found"
    )


def test_footnote_lines_are_not_aliases():
    corpus = corpus_from(
        (
            "medlineplus-alpha.md",
            "# Alpha\n"
            "## Brand names\n"
            "- Example\u00ae\u00b6\n\n"
            "\u00b6 This branded product is no longer on the market.",
        )
    )

    resolver = DrugResolver(corpus)

    assert (
        resolver.resolve_drug("Example").drug
        == "alpha"
    )
    assert (
        resolver.resolve_drug("This branded product").status
        == "not_found"
    )


def test_empty_corpus():
    resolver = DrugResolver(
        Corpus([], [], [])
    )

    assert resolver.drugs == ()
    assert (
        resolver.resolve_drug("alpha").status
        == "not_found"
    )


@pytest.mark.parametrize(
    "name",
    [
        None,
        123,
        "",
        " ",
        "!!!",
        "x" * 201,
    ],
)
def test_invalid_names(name, resolver):
    with pytest.raises(ValueError):
        resolver.resolve_drug(name)


def test_name_normalization():
    assert normalize_drug_name(
        "  ELIQUIS\u00ae  Sprinkle "
    ) == "eliquis sprinkle"

    assert normalize_drug_name(
        "Aspir-Mox"
    ) == normalize_drug_name(
        "Aspir Mox"
    )


def test_build_does_not_modify_corpus():
    corpus = corpus_from(
        (
            "medlineplus-alpha.md",
            "# Alpha\n"
            "## Brand names\n"
            "- Example",
        )
    )

    before = (
        list(corpus.documents),
        list(corpus.sections),
        list(corpus.chunks),
    )

    DrugResolver(corpus)

    assert before == (
        corpus.documents,
        corpus.sections,
        corpus.chunks,
    )