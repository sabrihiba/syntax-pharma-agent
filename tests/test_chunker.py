"""Tests for app.rag.chunker — real corpus and synthetic documents."""

import os
import re
from pathlib import Path

import pytest

from app.rag.chunker import (
    HEADING,
    RECORD_ROW,
    ChunkingConfig,
    _Unit,
    chunk_document,
    chunk_documents,
    hard_wrap,
    load_corpus,
    normalize,
    pack_units,
    parse_file_name,
    slugify,
    split_sentences,
    to_units,
)
# Set CORPUS_DIR if your Markdown corpus lives outside the usual folders.
def find_corpus_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    override = os.getenv("CORPUS_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"CORPUS_DIR does not exist: {path}")
        return path

    candidates = [
        root / "data" / "corpus",
        root / "data" / "raw",
        root / "corpus",
        root / "documents",
        root / "data",
    ]
    # Also locate the corpus when nested under an unexpected directory.
    candidates.extend(
        p.parent for p in root.rglob("dailymed-warfarin.md")
        if not any(part in {".venv", ".git", "node_modules"} for part in p.parts)
    )
    for folder in candidates:
        if (folder / "dailymed-warfarin.md").is_file() and (folder / "GLOSSARY.md").is_file():
            return folder
    raise FileNotFoundError(
        "Could not locate the 21-file corpus. Set CORPUS_DIR to the folder "
        "containing dailymed-warfarin.md and GLOSSARY.md."
    )


DEFAULT_CORPUS_DIR = find_corpus_dir()

SMALL = ChunkingConfig(
    max_section_chars=300, protected_max_chars=400, target_chars=200, overlap_max_chars=80
)
DEFAULT = ChunkingConfig()


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DEFAULT_CORPUS_DIR)


def chunks_of(corpus, section_prefix):
    return [c for c in corpus.chunks if c.section_id.startswith(section_prefix)]


def _read(file_name: str) -> str:
    return Path(DEFAULT_CORPUS_DIR, file_name).read_text(encoding="utf-8")


def _body_lines(file_name: str) -> list[str]:
    """Meaningful source lines: not blank, not a heading, not the Source: line.

    A heading is "#" + SPACE: table rows such as "#Cyclosporine 5.2 mg/kg | ..."
    (FDA footnote marker) are content and must be kept.
    """
    lines = [ln.strip() for ln in normalize(_read(file_name)).split("\n")]
    return [ln for ln in lines if ln and not HEADING.match(ln) and not ln.startswith("Source:")]


# ===================================================================== real corpus


def test_corpus_loads_every_file(corpus):
    assert len(corpus.documents) == 21
    assert corpus.drugs == sorted(
        [
            "acetaminophen", "albuterol", "amoxicillin", "apixaban", "aspirin",
            "atorvastatin", "ibuprofen", "metformin", "sertraline", "warfarin",
        ]
    )  # fmt: skip


# --- clinical completeness ----------------------------------------------------


def test_contraindication_sections_preserved_and_retrievable(corpus):
    """Full section retrieval must preserve every contraindication, even with children."""
    for drug in ["warfarin", "apixaban", "metformin"]:
        section = corpus.section(f"dailymed-{drug}.md#4-contraindications")
        assert section is not None, drug
        assert section.protected
        parts = chunks_of(corpus, section.id)
        if parts:
            assert all(c.protected for c in parts)
            assert all(c.text in section.text for c in parts)
        assert section.text.strip(), drug
    warfarin = corpus.section("dailymed-warfarin.md#4-contraindications").text
    for condition in [
        "Pregnancy",
        "Hemorrhagic tendencies or blood dyscrasias",
        "Malignant hypertension",
        "Major regional or lumbar block anesthesia",
        "Spinal puncture",
        "Hypersensitivity to warfarin",
    ]:
        assert condition in warfarin
    apixaban = corpus.section("dailymed-apixaban.md#4-contraindications").text
    assert "Active pathological bleeding" in apixaban
    assert "Severe hypersensitivity" in apixaban


def test_protected_sections_can_expand_even_when_split(corpus):
    """Splitting a warning is acceptable only if its complete parent is available."""
    for chunk in corpus.chunks:
        if not chunk.protected:
            continue
        section = corpus.parent(chunk)
        assert section.protected
        assert chunk.text in section.text
        if chunk.n_parts > 1:
            siblings = chunks_of(corpus, chunk.section_id)
            assert len(siblings) == chunk.n_parts


def test_warning_protection_inherits_from_ancestor():
    text = "# Metformin\n\n## 5 WARNINGS AND PRECAUTIONS\n\n### 5.1 Lactic Acidosis\n\nMonitor symptoms."
    _, sections, chunks = chunk_document("dailymed-metformin.md", text)
    child = next(s for s in sections if s.title == "5.1 Lactic Acidosis")
    assert child.protected
    assert chunks[0].protected


def test_parent_heading_aggregates_descendants():
    text = "# Metformin\n\n## 5 WARNINGS AND PRECAUTIONS\n\n### 5.1 Lactic Acidosis\n\nLactic warning.\n\n### 5.2 Vitamin B12\n\nB12 warning."
    _, sections, chunks = chunk_document("dailymed-metformin.md", text)
    parent = next(s for s in sections if s.title == "5 WARNINGS AND PRECAUTIONS")
    assert "Lactic warning." in parent.text
    assert "B12 warning." in parent.text
    assert not any(c.section_id == parent.id for c in chunks)


def test_parent_expansion_returns_the_full_section(corpus):
    for chunk in corpus.chunks:
        parent = corpus.parent(chunk)
        assert parent.id == chunk.section_id
        siblings = [c for c in corpus.chunks if c.section_id == parent.id]
        assert len(siblings) == chunk.n_parts
        assert [c.part for c in siblings] == list(range(chunk.n_parts))


# --- content integrity --------------------------------------------------------


def test_no_chunk_alters_content(corpus):
    """Every line of every chunk is verbatim text from its parent section."""
    for chunk in corpus.chunks:
        parent = corpus.parent(chunk).text
        for line in chunk.text.split("\n"):
            assert line.strip() in parent, f"{chunk.id}: {line[:80]}"


def test_every_source_line_is_covered(corpus):
    """Each meaningful line is in some chunk verbatim, or (if it was too long)
    each of its sentences is."""
    for doc in corpus.documents:
        blob = "\n".join(c.text for c in corpus.chunks if c.doc.file == doc.file)
        for line in _body_lines(doc.file):
            if line in blob:
                continue
            pieces = [p for s in split_sentences(line) for p in hard_wrap(s, DEFAULT.target_chars)]
            missing = [p for p in pieces if p not in blob]
            assert not missing, f"{doc.file}: {missing[0][:80]}"


def test_source_line_is_metadata_not_a_chunk(corpus):
    assert not any(c.text.startswith("Source:") for c in corpus.chunks)
    for doc in corpus.documents:
        if doc.source != "glossary":
            assert doc.url and doc.url.startswith("https://"), doc.file


# --- tables -------------------------------------------------------------------


def test_table_rows_are_never_cut(corpus):
    for chunk in corpus.chunks:
        source_lines = set(_body_lines(chunk.doc.file))
        for line in chunk.text.split("\n"):
            if " | " in line:
                assert line.strip() in source_lines, f"{chunk.id}: {line[:80]}"


def test_continued_tables_repeat_their_header(corpus):
    parts = chunks_of(corpus, "dailymed-warfarin.md#7.2-cyp450-interactions")
    assert len(parts) > 1
    for part in parts[1:]:
        assert part.text.startswith("Enzyme | Inhibitors | Inducers")


def test_interaction_record_keeps_title_and_intervention_together(corpus):
    """Q3 depends on this: the sertraline 'Intervention' row must stay with its drug class."""
    sertraline = [c for c in corpus.chunks if c.doc.file == "dailymed-sertraline.md"]
    hits = [c for c in sertraline if "carefully monitor the international" in c.text]
    record = [c for c in hits if "Intervention:" in c.text]
    assert record
    assert any("Drugs that Interfere with Hemostasis" in c.text for c in record)


# --- sizes and overlap ----------------------------------------------------------


def test_split_pieces_respect_target_on_corpus(corpus):
    for chunk in corpus.chunks:
        limit = DEFAULT.target_chars if chunk.is_partial else DEFAULT.protected_max_chars
        assert 0 < len(chunk.text) <= limit, chunk.id


LIST_LINE = re.compile(r"^\s*(?:[-*•■▪●]|\d+[.)])\s")


def _is_piped(line: str) -> bool:
    return " | " in line or line.startswith("|")


def _structural_headers(section_text: str) -> set[str]:
    """Lines the chunker may legitimately repeat: grid-table headers and record titles."""
    lines = [ln.strip() for ln in section_text.split("\n")]
    headers = set()
    for i, line in enumerate(lines):
        prev_piped = i > 0 and _is_piped(lines[i - 1])
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        grid_header = _is_piped(line) and not prev_piped  # first row of a grid table
        header_continuation = line.startswith("|") and prev_piped
        record_title = not _is_piped(line) and bool(RECORD_ROW.match(next_line))
        if grid_header or header_continuation or record_title:
            headers.add(line)
    return headers


def _duplicated_prefix(previous: str, body: str) -> str:
    """Longest prefix of `body` (cut at a word boundary) that `previous` ends with."""
    boundaries = [i for i, ch in enumerate(body) if ch in " \n"] + [len(body)]
    for pos in reversed(boundaries):
        if pos and previous.endswith(body[:pos]):
            return body[:pos]
    return ""


def _split_sections(corpus):
    for section in corpus.sections:
        pieces = [c.text for c in corpus.chunks if c.section_id == section.id]
        if len(pieces) > 1:
            yield section, pieces


def test_table_rows_and_list_items_are_never_duplicated(corpus):
    for section, pieces in _split_sections(corpus):
        headers = _structural_headers(section.text)
        source_lines = [ln.strip() for ln in section.text.split("\n")]
        piece_lines = [ln.strip() for piece in pieces for ln in piece.split("\n")]
        for line in set(source_lines):
            if line in headers or not (_is_piped(line) or LIST_LINE.match(line)):
                continue
            assert piece_lines.count(line) == source_lines.count(line), f"{section.id}: {line[:80]}"


def test_prose_overlap_is_at_most_one_sentence(corpus):
    checked = 0
    for section, pieces in _split_sections(corpus):
        headers = _structural_headers(section.text)
        for previous, current in zip(pieces, pieces[1:], strict=False):
            lines = current.split("\n")
            while lines and lines[0].strip() in headers:  # repeated table header: allowed
                lines = lines[1:]
            duplicated = _duplicated_prefix(previous, "\n".join(lines)).strip()
            if not duplicated:
                continue
            checked += 1
            assert not _is_piped(duplicated) and not LIST_LINE.match(duplicated), section.id
            assert len(split_sentences(duplicated)) == 1, f"{section.id}: {duplicated[:80]}"
    assert checked > 0  # the corpus does exercise the overlap path


# --- metadata, ids, citations ---------------------------------------------------


def test_every_chunk_is_self_describing(corpus):
    """The contextual header makes every chunk name its drug, source and section."""
    for c in corpus.chunks:
        if c.doc.drug:
            assert c.embed_text.startswith(f"Drug: {c.doc.drug.title()} | "), c.id
        assert f"Section: {c.section_path}" in c.embed_text
        assert c.embed_text.endswith(c.text)


def test_ids_are_unique_readable_and_stable(corpus):
    ids = [c.id for c in corpus.chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9.\-]+#[a-z0-9.\-]+#\d+", i) for i in ids)
    assert [c.id for c in load_corpus(DEFAULT_CORPUS_DIR).chunks] == ids  # deterministic


def test_citation_is_section_level_with_url(corpus):
    chunk = chunks_of(corpus, "dailymed-warfarin.md#7.3-drugs-that-increase-bleeding-risk")[0]
    citation = chunk.citation
    assert citation.id == "dailymed-warfarin.md#7.3-drugs-that-increase-bleeding-risk"
    assert citation.section_number == "7.3"
    assert citation.url.startswith("https://dailymed.nlm.nih.gov/")
    assert citation.label == "dailymed-warfarin.md, §7.3 Drugs that Increase Bleeding Risk"
    brand = chunks_of(corpus, "medlineplus-warfarin.md#brand-names")[0].citation
    assert brand.label == "medlineplus-warfarin.md, Brand names"


def test_section_numbers_and_paths(corpus):
    section = corpus.section("dailymed-warfarin.md#7.3-drugs-that-increase-bleeding-risk")
    assert section.number == "7.3"
    assert section.path == ("7 DRUG INTERACTIONS", "7.3 Drugs that Increase Bleeding Risk")
    assert "sertraline" in section.text


def test_headings_without_body_are_retained_for_expansion(corpus):
    parent = corpus.section("dailymed-warfarin.md#2-dosage-and-administration")
    assert parent is not None
    assert parent.text.strip()
    child = corpus.section("dailymed-warfarin.md#2.1-individualized-dosing")
    assert child is not None
    assert child.path[0] == "2 DOSAGE AND ADMINISTRATION"
    assert child.parent_id == parent.id
    assert child.text in parent.text


def test_boxed_warning_flag(corpus):
    boxed = corpus.section("dailymed-warfarin.md#warning-bleeding-risk")
    assert boxed.boxed_warning and boxed.protected
    assert "BOXED WARNING" in chunks_of(corpus, boxed.id)[0].embed_text


def test_glossary_has_no_drug(corpus):
    glossary = [c for c in corpus.chunks if c.doc.file == "GLOSSARY.md"]
    assert glossary and all(c.doc.drug is None and c.doc.source == "glossary" for c in glossary)


def test_superscript_restored(corpus):
    texts = " ".join(c.text for c in corpus.chunks if c.doc.drug == "metformin")
    assert "1.73 m²" in texts
    assert "1.73 m 2" not in texts and "m²is" not in texts


def test_inline_html_removed_but_content_kept(corpus):
    brand = chunks_of(corpus, "medlineplus-warfarin.md#brand-names")[0]
    assert "Coumadin®" in brand.text
    assert not any("<sup>" in c.text or "<sub>" in c.text for c in corpus.chunks)


# ============================================================ synthetic documents


def test_long_prose_overlaps_by_exactly_one_sentence():
    sentences = [f"Sentence number {i} talks about warfarin and INR monitoring." for i in range(12)]
    doc = "# Drug X\n\n## 5 NOTES\n\n" + " ".join(sentences)
    _, _, chunks = chunk_document("dailymed-drugx.md", doc, SMALL)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_sentence = split_sentences(previous.text)[-1]
        assert current.text.startswith(last_sentence + " ")
        assert split_sentences(previous.text)[-2] not in current.text  # only ONE sentence
    assert all(len(c.text) <= SMALL.target_chars for c in chunks)


def test_short_multi_sentence_line_overlaps_by_its_last_sentence_only():
    lines = [f"Line {i} first sentence here. Line {i} second sentence here." for i in range(8)]
    doc = "# T\n\n## 5 NOTES\n\n" + "\n".join(lines)
    _, _, chunks = chunk_document("dailymed-t.md", doc, SMALL)
    assert len(chunks) > 1
    for current in chunks[1:]:
        first_line = current.text.split("\n")[0]
        assert len(split_sentences(first_line)) == 1  # never a whole multi-sentence line


def test_list_items_are_never_repeated_as_overlap():
    items = "\n".join(f"- item {i} describing a side effect of the drug" for i in range(15))
    _, _, chunks = chunk_document("dailymed-t.md", f"# T\n\n## 6 ADVERSE\n\n{items}", SMALL)
    assert len(chunks) > 1
    assert sum(c.text.count("describing a side effect") for c in chunks) == 15  # no duplicates


def test_grid_table_split_repeats_header_and_counts_it_in_size():
    header = "Drug | " + "Effect on bleeding risk and required monitoring " * 2
    rows = "\n".join(f"Drug{i} | effect number {i}" for i in range(20))
    _, _, chunks = chunk_document(
        "dailymed-t.md", f"# T\n\n## 7 INTERACTIONS\n\n{header.strip()}\n{rows}", SMALL
    )
    assert len(chunks) > 1
    assert all(c.text.startswith(header.strip() + "\n") for c in chunks)
    assert all(len(c.text) <= SMALL.target_chars for c in chunks)  # header included in size
    assert sum(c.text.count("| effect number") for c in chunks) == 20  # no row lost/duplicated


def test_oversized_table_row_is_kept_whole():
    long_row = "Warfarin | " + "major bleeding risk with concomitant use " * 10  # > target
    rows = "\n".join(f"Drug{i} | minor effect {i}" for i in range(10))
    doc = f"# T\n\n## 7 INTERACTIONS\n\nDrug | Effect\n{rows}\n{long_row.strip()}"
    _, _, chunks = chunk_document("dailymed-t.md", doc, SMALL)
    holders = [c for c in chunks if "Warfarin |" in c.text]
    assert len(holders) == 1
    assert long_row.strip() in holders[0].text  # intact, never wrapped
    assert holders[0].text == f"Drug | Effect\n{long_row.strip()}"  # alone, with its header


def test_long_record_is_split_by_rows_with_title_repeated():
    rows = "\n".join(f"Intervention {i}: | monitor INR closely for case {i}" for i in range(8))
    doc = f"# T\n\n## 7 INTERACTIONS\n\nDrugs that Interfere with Hemostasis\n{rows}"
    _, _, chunks = chunk_document("dailymed-t.md", doc, SMALL)
    assert len(chunks) > 1
    assert all(c.text.startswith("Drugs that Interfere with Hemostasis\n") for c in chunks)


def test_caption_before_grid_table_is_not_a_header():
    rows = "\n".join(f"Row{i} | value {i} with some padding text here" for i in range(10))
    doc = f"# T\n\n## 12 PK\n\naFootnote line before the table.\nGroup | Value\n{rows}"
    _, _, chunks = chunk_document("dailymed-t.md", doc, SMALL)
    assert all("Group | Value" in c.text for c in chunks if "| value" in c.text)


def test_protected_section_kept_whole_up_to_its_limit():
    body = "\n".join(f"- condition {i} that is a contraindication" for i in range(9))
    assert SMALL.max_section_chars < len(body) <= SMALL.protected_max_chars
    _, _, chunks = chunk_document(
        "dailymed-t.md", f"# T\n\n## 4 CONTRAINDICATIONS\n\n{body}", SMALL
    )
    assert len(chunks) == 1 and chunks[0].protected


def test_oversized_protected_section_is_split_but_flagged_for_expansion(caplog):
    body = "\n".join(f"- condition {i} that is a contraindication" for i in range(20))
    assert len(body) > SMALL.protected_max_chars
    _, sections, chunks = chunk_document(
        "dailymed-t.md", f"# T\n\n## 4 CONTRAINDICATIONS\n\n{body}", SMALL
    )
    assert len(chunks) > 1
    assert all(c.protected and c.is_partial and c.n_parts == len(chunks) for c in chunks)
    assert sections[0].text == body  # parent keeps the complete list
    assert "split into" in caplog.text


def test_txt_file_without_headings():
    doc, sections, chunks = chunk_document("notes.txt", "Plain text about ibuprofen.", SMALL)
    assert (doc.source, doc.drug) == ("other", "notes")
    assert sections[0].path == ("Overview",) and chunks[0].text == "Plain text about ibuprofen."


def test_duplicate_headings_get_unique_ids():
    doc = "# T\n\n## Other Adverse Reactions\n\nA.\n\n## Other Adverse Reactions\n\nB."
    _, sections, _ = chunk_document("dailymed-t.md", doc, SMALL)
    assert [s.id for s in sections] == [
        "dailymed-t.md#other-adverse-reactions",
        "dailymed-t.md#other-adverse-reactions-2",
    ]


# ------------------------------------------------- review round 3: size / items / tables


def test_section_that_fits_stays_whole_even_above_target():
    """1,300 + 100 chars = 1,401 <= max_section_chars: by design ONE chunk.
    target_chars only applies when a section has to be split."""
    body = "A" * 1300 + "\n" + "b" * 100
    _, _, chunks = chunk_document("dailymed-t.md", f"# T\n\n## 6 X\n\n{body}", DEFAULT)
    assert [len(c.text) for c in chunks] == [1401]


def test_oversized_atomic_unit_is_emitted_alone_when_section_is_split():
    item = "- " + "long adverse reaction description " * 40  # ~1,400 chars > target
    filler = " ".join(f"Filler sentence {i} about dosing." for i in range(40))
    body = f"{item.strip()}\nShort line after the item.\n\n{filler}"
    assert len(body) > DEFAULT.max_section_chars  # forces splitting
    _, _, chunks = chunk_document("dailymed-t.md", f"# T\n\n## 6 X\n\n{body}", DEFAULT)
    assert chunks[0].text == item.strip()  # alone: nothing packed after it
    assert chunks[1].text.startswith("Short line after the item.")


def test_long_list_item_is_atomic():
    item = "- " + "bleeding tendency associated with ulceration " * 30  # ~1,350 chars
    rest = "\n".join(f"- short item {i}" for i in range(80))
    body = f"{item.strip()}\n{rest}"
    _, _, chunks = chunk_document("dailymed-t.md", f"# T\n\n## 6 X\n\n{body}", DEFAULT)
    holders = [c for c in chunks if "bleeding tendency" in c.text]
    assert len(holders) == 1 and holders[0].text == item.strip()


def test_pathological_list_item_is_split_without_overlap():
    sentences = [f"Clause {i} of a very long list item about hepatotoxicity." for i in range(50)]
    item = "- " + " ".join(sentences)  # ~2,900 chars > max_section_chars
    body = f"{item}\n\n" + "\n".join(f"- other item {i}" for i in range(10))
    _, _, chunks = chunk_document("dailymed-t.md", f"# T\n\n## 6 X\n\n{body}", DEFAULT)
    joined = "\n".join(c.text for c in chunks)
    for sentence in sentences:
        assert joined.count(sentence) == 1  # every clause kept, none duplicated as overlap


def test_single_pipe_in_prose_is_not_a_table():
    units = to_units("The ratio A | B is defined in the text.\nNext line.", DEFAULT)
    assert [u.kind for u in units] == ["prose", "prose"]
    units = to_units("Drug | Effect\nAspirin | bleeding\nHeparin | bleeding", DEFAULT)
    assert [u.kind for u in units] == ["row", "row"]
    assert all(u.header == "Drug | Effect" for u in units)
    units = to_units("Drugs that Interfere with Hemostasis\nIntervention: | Monitor INR.", DEFAULT)
    assert [u.kind for u in units] == ["row"]  # a single FDA record is still a table


def test_pack_units_invariant_randomized():
    """Property test: for any mix of units, a piece longer than target_chars holds
    exactly one atomic unit (+ its header); nothing is lost; rows never repeat."""
    import random

    rng = random.Random(42)
    config = ChunkingConfig(
        max_section_chars=400, protected_max_chars=500, target_chars=200, overlap_max_chars=60
    )
    for _ in range(300):
        units = []
        header = "Col A | Col B"
        for i in range(rng.randint(1, 25)):
            kind = rng.choice(["prose", "item", "row"])
            size = rng.choice([20, 60, 150, 250])  # 250 > target: oversized atomic unit
            text = f"{kind}{i} " + "x" * size
            if kind == "row":
                text = f"r{i} | " + "y" * size
            units.append(_Unit(text, rng.choice(["\n", " ", "\n\n"]), kind,
                               header if kind == "row" else None))  # fmt: skip
        pieces = pack_units(units, config)
        for piece in pieces:
            if len(piece) > config.target_chars:
                body = piece.removeprefix(header + "\n")
                assert body in {u.text for u in units}, piece[:60]
        joined = "\n".join(pieces)
        for unit in units:
            assert unit.text in joined
            if unit.kind in ("row", "item"):
                assert joined.count(unit.text) == 1


# ======================================================================== helpers


def test_split_sentences_respects_abbreviations():
    text = "Monitor INR (e.g. Weekly). Dr. Smith agreed. Approx. 10 mg daily. J. Doe too."
    assert split_sentences(text) == [
        "Monitor INR (e.g. Weekly).",
        "Dr. Smith agreed.",
        "Approx. 10 mg daily.",
        "J. Doe too.",
    ]


def test_hard_wrap_bounds_every_piece():
    text = "word " * 50 + "x" * 120
    pieces = hard_wrap(text.strip(), 40)
    assert all(len(p) <= 40 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dailymed-warfarin.md", ("dailymed", "warfarin")),
        ("medlineplus-sertraline.md", ("medlineplus", "sertraline")),
        ("GLOSSARY.md", ("glossary", None)),
        ("dailymed-insulin-glargine.md", ("dailymed", "insulin glargine")),
        ("random.txt", ("other", "random")),
    ],
)
def test_parse_file_name(name, expected):
    assert parse_file_name(name) == expected


def test_slugify():
    assert (
        slugify("7.3 Drugs that Increase Bleeding Risk") == "7.3-drugs-that-increase-bleeding-risk"
    )
    assert slugify("WARNING: BLEEDING RISK") == "warning-bleeding-risk"
    assert len(slugify("word " * 30)) <= 48


def test_normalize_superscript_variants():
    assert normalize("eGFR 30 mL/min/1.73 m 2 [see") == "eGFR 30 mL/min/1.73 m² [see"
    assert normalize("1.73 m 2is not") == "1.73 m² is not"
    assert normalize("Coumadin<sup>®</sup>") == "Coumadin®"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_chars": 3000, "max_section_chars": 2000},
        {"max_section_chars": 4000, "protected_max_chars": 3500},
        {"overlap_max_chars": 1200},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)

# ======================================================= new-version regressions


def test_markdown_table_separator_and_rows():
    text = "# Drug\n\n## 7 INTERACTIONS\n\n| Drug | Effect |\n| --- | --- |\n" + "\n".join(
        f"| Drug{i} | Effect{i} |" for i in range(15)
    )
    _, _, chunks = chunk_document("dailymed-test.md", text, SMALL)
    assert len(chunks) > 1
    assert all(c.text.startswith("| Drug | Effect |\n| --- | --- |") for c in chunks)
    for i in range(15):
        assert sum(f"| Drug{i} | Effect{i} |" in c.text for c in chunks) == 1


def test_fenced_code_heading_is_not_a_section():
    text = "# Example\n\n## 1 NOTES\n\n```python\n# comment\nprint('hello')\n```\n\n## 2 OTHER\n\nBody."
    _, sections, _ = chunk_document("notes.md", text)
    assert [s.title for s in sections] == ["1 NOTES", "2 OTHER"]
    assert "# comment" in sections[0].text


def test_dictionary_api_compatibility():
    documents = [{"source": "dailymed-test.md", "content": "# Test\n\n## 4 CONTRAINDICATIONS\n\nNo use in pregnancy."}]
    parents, children = chunk_documents(documents)
    assert parents and children
    assert parents[0]["section_id"] == children[0]["section_id"]
    assert parents[0]["content"] == "No use in pregnancy."
    assert "Drug: Test" in children[0]["embed_text"]
