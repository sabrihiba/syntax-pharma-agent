"""Hierarchical, structure-aware chunking for the pharmacology corpus.

Public APIs:
    load_corpus(corpus_dir, config=None) -> Corpus
    chunk_document(file_name, content, config=None)
        -> (DocumentInfo, list[Section], list[Chunk])
    chunk_documents(documents, soft_limit=2200, hard_limit=3500)
        -> (parents, children)

Sections contain their complete subtree, including descendant headings and text.
Chunks index only each heading's own body, avoiding duplicate embeddings.

Size limits apply to chunk text, excluding the contextual embedding header.
Atomic table rows and list items may exceed target_chars.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
SOURCE_LINE = re.compile(r"^Source:\s*(\S+)\s*$", re.IGNORECASE)
SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\s+")

PROTECTED = re.compile(r"contraindicat|warning|precaution", re.IGNORECASE)
BOXED_WARNING = re.compile(r"^(?:boxed\s+)?warning\b", re.IGNORECASE)

RECORD_ROW = re.compile(r"^\s*[A-Za-z][\w ()/,-]{0,40}:\s*\|")
LIST_ITEM = re.compile(r"^\s*(?:[-*•■▪●]|\d+[.)])\s+")
FENCE_OPEN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")

SENTENCE_BOUNDARY = re.compile(
    r"""(?<=[.!?])\s+(?=["“(\[]?[A-Z0-9])"""
)
ABBREVIATIONS = frozenset({
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "dr.", "mr.", "mrs.",
    "ms.", "no.", "fig.", "inc.", "st.", "al.", "ca.", "min.", "max.",
    "ref.", "resp.", "incl.", "u.s.", "u.s.p.", "mg.", "ml.", "hr.",
    "hrs.", "wk.", "mo.", "yr.",
})
SINGLE_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")

SUPERSCRIPT_M2_GLUED = re.compile(r"(\d+\.\d+)\s*m\s*2(?=[A-Za-z])")
SUPERSCRIPT_M2 = re.compile(r"(\d+\.\d+)\s*m\s*2\b")
INLINE_HTML = re.compile(r"</?(?:sup|sub)>", re.IGNORECASE)

SOURCE_LABELS = {
    "dailymed": "DailyMed FDA label",
    "medlineplus": "MedlinePlus patient monograph",
    "glossary": "Pharmacology glossary",
    "other": "Document",
}
CORPUS_SUFFIXES = {".md", ".txt"}

UnitKind = Literal["prose", "item", "row", "header"]


@dataclass(frozen=True)
class ChunkingConfig:
    max_section_chars: int = 2000
    protected_max_chars: int = 3500
    target_chars: int = 1200
    overlap_max_chars: int = 300

    def __post_init__(self) -> None:
        if not (
            0 < self.target_chars
            <= self.max_section_chars
            <= self.protected_max_chars
        ):
            raise ValueError(
                "need 0 < target_chars <= max_section_chars "
                "<= protected_max_chars"
            )
        if not 0 <= self.overlap_max_chars < self.target_chars:
            raise ValueError(
                "need 0 <= overlap_max_chars < target_chars"
            )


@dataclass(frozen=True)
class DocumentInfo:
    file: str
    source: str
    drug: str | None
    title: str
    url: str | None

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, SOURCE_LABELS["other"])


@dataclass(frozen=True)
class Section:
    """Complete parent section, including all descendant sections."""

    id: str
    doc: DocumentInfo
    path: tuple[str, ...]
    number: str | None
    text: str
    protected: bool
    boxed_warning: bool
    parent_id: str | None = None

    @property
    def title(self) -> str:
        return self.path[-1]

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)


@dataclass(frozen=True)
class Citation:
    id: str
    file: str
    source_label: str
    url: str | None
    section_number: str | None
    section_path: str

    @property
    def label(self) -> str:
        leaf = self.section_path.split(" > ")[-1]
        if self.section_number:
            return f"{self.file}, §{leaf}"
        return f"{self.file}, {leaf}"


@dataclass(frozen=True)
class Chunk:
    """Retrieval child containing a heading's own body."""

    id: str
    section_id: str
    part: int
    n_parts: int
    doc: DocumentInfo
    section_path: str
    section_number: str | None
    protected: bool
    text: str
    embed_text: str
    has_descendants: bool = False

    @property
    def is_partial(self) -> bool:
        return self.n_parts > 1 or self.has_descendants

    @property
    def citation(self) -> Citation:
        return Citation(
            id=self.section_id,
            file=self.doc.file,
            source_label=self.doc.source_label,
            url=self.doc.url,
            section_number=self.section_number,
            section_path=self.section_path,
        )


@dataclass(frozen=True)
class _Unit:
    text: str
    sep: str
    kind: UnitKind
    header: str | None = None


@dataclass
class _ParsedSection:
    level: int
    path: tuple[str, ...]
    parent_index: int | None
    lines: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    documents: list[DocumentInfo]
    sections: list[Section]
    chunks: list[Chunk]
    _sections_by_id: dict[str, Section] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sections_by_id = {}
        for section in self.sections:
            if section.id in self._sections_by_id:
                raise ValueError(f"duplicate section id: {section.id}")
            self._sections_by_id[section.id] = section

    def section(self, section_id: str) -> Section | None:
        return self._sections_by_id.get(section_id)

    def parent(self, chunk: Chunk) -> Section:
        return self._sections_by_id[chunk.section_id]

    @property
    def drugs(self) -> list[str]:
        return sorted({doc.drug for doc in self.documents if doc.drug})


def load_corpus(
    corpus_dir: Path,
    config: ChunkingConfig | None = None,
) -> Corpus:
    config = config or ChunkingConfig()
    files = sorted(
        path
        for path in Path(corpus_dir).iterdir()
        if path.is_file() and path.suffix.lower() in CORPUS_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"no .md/.txt files found in {corpus_dir}"
        )

    documents: list[DocumentInfo] = []
    sections: list[Section] = []
    chunks: list[Chunk] = []

    for path in files:
        doc, doc_sections, doc_chunks = chunk_document(
            path.name,
            path.read_text(encoding="utf-8-sig"),
            config,
        )
        documents.append(doc)
        sections.extend(doc_sections)
        chunks.extend(doc_chunks)

    return Corpus(documents, sections, chunks)


def chunk_document(
    file_name: str,
    content: str,
    config: ChunkingConfig | None = None,
) -> tuple[DocumentInfo, list[Section], list[Chunk]]:
    config = config or ChunkingConfig()
    title, url, nodes = _parse_sections(normalize(content))
    source, drug = parse_file_name(file_name)

    doc = DocumentInfo(
        file=file_name,
        source=source,
        drug=drug,
        title=title or Path(file_name).stem,
        url=url,
    )

    # Include the suffix to distinguish label.md from label.txt.
    namespace = Path(file_name).name.lower()
    used_ids: set[str] = set()
    ids = [
        _unique(f"{namespace}#{slugify(node.path[-1])}", used_ids)
        for node in nodes
    ]

    children: list[list[int]] = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        if node.parent_index is not None:
            children[node.parent_index].append(index)

    # Build complete subtree text from the leaves upward.
    full_text = [""] * len(nodes)
    for index in range(len(nodes) - 1, -1, -1):
        node = nodes[index]
        parts = ["\n".join(node.lines).strip()]
        for child_index in children[index]:
            child = nodes[child_index]
            heading = f"{'#' * child.level} {child.path[-1]}"
            parts.append(
                f"{heading}\n\n{full_text[child_index]}".rstrip()
            )
        full_text[index] = "\n\n".join(
            part for part in parts if part
        )

    sections: list[Section] = []
    chunks: list[Chunk] = []

    for index, node in enumerate(nodes):
        leaf = node.path[-1]
        match = SECTION_NUMBER.match(leaf)
        path_text = " > ".join(node.path)

        section = Section(
            id=ids[index],
            doc=doc,
            path=node.path,
            number=match.group(1) if match else None,
            text=full_text[index],
            protected=bool(PROTECTED.search(path_text)),
            boxed_warning=(
                source == "dailymed"
                and any(BOXED_WARNING.match(name) for name in node.path)
            ),
            parent_id=(
                ids[node.parent_index]
                if node.parent_index is not None
                else None
            ),
        )
        sections.append(section)

        # Embed only direct content. Descendants have their own chunks.
        body = "\n".join(node.lines).strip()
        if not body:
            continue

        pieces = _split_body(body, section.protected, config)
        if section.protected and len(pieces) > 1:
            logger.warning(
                "protected section %s split into %d chunks",
                section.id,
                len(pieces),
            )

        header = context_header(section)
        for part, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{section.id}#{part}",
                    section_id=section.id,
                    part=part,
                    n_parts=len(pieces),
                    doc=doc,
                    section_path=section.path_str,
                    section_number=section.number,
                    protected=section.protected,
                    text=piece,
                    embed_text=f"{header}\n{piece}",
                    has_descendants=bool(children[index]),
                )
            )

    return doc, sections, chunks


def chunk_documents(
    documents,
    soft_limit: int = 2200,
    hard_limit: int = 3500,
):
    """Compatibility API for [{"source": ..., "content": ...}, ...].

    Parents contain complete sections.
    Children retain the original fields and add retrieval metadata.
    """
    config = ChunkingConfig(
        max_section_chars=soft_limit,
        protected_max_chars=hard_limit,
        target_chars=min(1200, soft_limit),
        overlap_max_chars=min(300, min(1200, soft_limit) - 1),
    )
    parents = []
    children = []

    for document in documents:
        _, sections, chunks = chunk_document(
            document["source"],
            document["content"],
            config,
        )
        parent_ids = {
            section.id: len(parents) + index
            for index, section in enumerate(sections)
        }

        for section in sections:
            parents.append({
                "parent_id": parent_ids[section.id],
                "section_id": section.id,
                "ancestor_parent_id": (
                    parent_ids[section.parent_id]
                    if section.parent_id is not None
                    else None
                ),
                "source": section.doc.file,
                "drug": section.doc.drug,
                "section": section.path_str,
                "content": section.text,
                "protected": section.protected,
                "boxed_warning": section.boxed_warning,
                "url": section.doc.url,
            })

        for chunk in chunks:
            children.append({
                "chunk_id": len(children),
                "stable_id": chunk.id,
                "parent_id": parent_ids[chunk.section_id],
                "section_id": chunk.section_id,
                "part": chunk.part,
                "n_parts": chunk.n_parts,
                "is_partial": chunk.is_partial,
                "source": chunk.doc.file,
                "drug": chunk.doc.drug,
                "section": chunk.section_path,
                "content": chunk.text,
                "embed_text": chunk.embed_text,
                "protected": chunk.protected,
                "url": chunk.doc.url,
            })

    return parents, children


def normalize(text: str) -> str:
    """Normalize known export artifacts; preserve fenced code verbatim."""
    text = text.removeprefix("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    result: list[str] = []
    fence: str | None = None

    for line in text.split("\n"):
        if fence is not None:
            result.append(line)
            if _closes_fence(line, fence):
                fence = None
            continue

        opening = FENCE_OPEN.match(line)
        if opening:
            fence = opening.group(1)
            result.append(line)
            continue

        line = INLINE_HTML.sub("", line)
        line = SUPERSCRIPT_M2_GLUED.sub(r"\1 m² ", line)
        line = SUPERSCRIPT_M2.sub(r"\1 m²", line)
        result.append(line.rstrip())

    return "\n".join(result)


def parse_file_name(file_name: str) -> tuple[str, str | None]:
    stem = Path(file_name).stem.lower()
    if stem == "glossary":
        return "glossary", None

    prefix, _, rest = stem.partition("-")
    if prefix in SOURCE_LABELS and rest:
        return prefix, rest.replace("-", " ")

    return "other", stem.replace("-", " ")


def _closes_fence(line: str, fence: str) -> bool:
    pattern = (
        r"^[ \t]{0,3}"
        + re.escape(fence[0])
        + "{"
        + str(len(fence))
        + r",}[ \t]*$"
    )
    return bool(re.match(pattern, line))


def _parse_sections(
    text: str,
) -> tuple[str | None, str | None, list[_ParsedSection]]:
    title = None
    url = None
    nodes: list[_ParsedSection] = []
    stack: list[int] = []
    preamble: list[str] = []
    fence: str | None = None

    def append_line(line: str) -> None:
        if stack:
            nodes[stack[-1]].lines.append(line)
        else:
            preamble.append(line)

    def flush_preamble() -> None:
        if "\n".join(preamble).strip():
            nodes.append(
                _ParsedSection(
                    level=0,
                    path=("Overview",),
                    parent_index=None,
                    lines=list(preamble),
                )
            )
        preamble.clear()

    for line in text.split("\n"):
        if fence is not None:
            append_line(line)
            if _closes_fence(line, fence):
                fence = None
            continue

        opening = FENCE_OPEN.match(line)
        if opening:
            fence = opening.group(1)
            append_line(line)
            continue

        heading = HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            name = heading.group(2).strip()
            name = re.sub(r"[ \t]+#+[ \t]*$", "", name).strip()

            # Only the initial H1 is document metadata.
            if (
                level == 1
                and title is None
                and not nodes
                and not stack
            ):
                title = name
                continue

            if not nodes and not stack:
                flush_preamble()

            while stack and nodes[stack[-1]].level >= level:
                stack.pop()

            parent_index = stack[-1] if stack else None
            parent_path = (
                nodes[parent_index].path
                if parent_index is not None
                else ()
            )
            nodes.append(
                _ParsedSection(
                    level=level,
                    path=(*parent_path, name),
                    parent_index=parent_index,
                )
            )
            stack.append(len(nodes) - 1)
            continue

        source = SOURCE_LINE.match(line)
        if source and url is None and not nodes and not stack:
            url = source.group(1)
            continue

        append_line(line)

    if not nodes:
        flush_preamble()

    return title, url, nodes


def split_into_sections(
    text: str,
) -> tuple[
    str | None,
    str | None,
    list[tuple[tuple[str, ...], str]],
]:
    """Return direct section bodies, retaining headings with empty bodies."""
    title, url, nodes = _parse_sections(text)
    return title, url, [
        (node.path, "\n".join(node.lines).strip())
        for node in nodes
    ]


def _split_body(
    text: str,
    protected: bool,
    config: ChunkingConfig,
) -> list[str]:
    if not text:
        return []

    limit = (
        config.protected_max_chars
        if protected
        else config.max_section_chars
    )
    if len(text) <= limit:
        return [text]

    return pack_units(to_units(text, config), config)


def split_section(
    section: Section,
    config: ChunkingConfig,
) -> list[str]:
    """Split a complete section when explicitly requested."""
    return _split_body(section.text, section.protected, config)


def _blocks(text: str) -> list[tuple[bool, list[str]]]:
    """Split on blank lines while keeping fenced blocks intact."""
    blocks: list[tuple[bool, list[str]]] = []
    current: list[str] = []
    fence: str | None = None

    def flush() -> None:
        if current:
            blocks.append((False, list(current)))
            current.clear()

    for line in text.split("\n"):
        if fence is not None:
            current.append(line)
            if _closes_fence(line, fence):
                blocks.append((True, list(current)))
                current.clear()
                fence = None
            continue

        opening = FENCE_OPEN.match(line)
        if opening:
            flush()
            fence = opening.group(1)
            current.append(line)
        elif not line.strip():
            flush()
        else:
            current.append(line)

    if current:
        blocks.append((fence is not None, list(current)))

    return blocks


def to_units(text: str, config: ChunkingConfig) -> list[_Unit]:
    units: list[_Unit] = []

    for fenced, lines in _blocks(text):
        if fenced:
            # Code blocks stay atomic and receive no prose overlap.
            block_units = [_Unit("\n".join(lines), "\n\n", "row")]
        elif _is_table_block(lines):
            block_units = _table_units(lines, config)
        else:
            block_units = []
            pending_item: list[str] = []

            def flush_item() -> None:
                if pending_item:
                    block_units.extend(
                        _line_units(
                            "\n".join(pending_item),
                            config,
                            first_sep="\n",
                        )
                    )
                    pending_item.clear()

            for line in lines:
                if LIST_ITEM.match(line):
                    flush_item()
                    pending_item.append(line)
                elif pending_item and line[:1].isspace():
                    pending_item.append(line)
                else:
                    flush_item()
                    block_units.extend(
                        _line_units(line, config, first_sep="\n")
                    )

            flush_item()

        if block_units:
            first = block_units[0]
            block_units[0] = _Unit(
                first.text, "\n\n", first.kind, first.header
            )
            units.extend(block_units)

    return units


def _line_units(
    line: str,
    config: ChunkingConfig,
    first_sep: str,
) -> list[_Unit]:
    is_item = bool(LIST_ITEM.match(line))
    kind: UnitKind = "item" if is_item else "prose"
    limit = (
        config.max_section_chars if is_item else config.target_chars
    )

    if len(line) <= limit:
        return [_Unit(line, first_sep, kind)]

    units: list[_Unit] = []
    for sentence in split_sentences(line):
        for piece in hard_wrap(sentence, config.target_chars):
            units.append(
                _Unit(
                    piece,
                    first_sep if not units else " ",
                    kind,
                )
            )

    return units


def _is_table_row(line: str) -> bool:
    return " | " in line or line.lstrip().startswith("|")


def _is_separator_row(line: str) -> bool:
    if "|" not in line:
        return False

    cells = line.strip().strip("|").split("|")
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.strip())
        for cell in cells
    )


def _is_table_block(lines: list[str]) -> bool:
    return (
        sum(_is_table_row(line) for line in lines) >= 2
        or any(RECORD_ROW.match(line) for line in lines)
    )


def _table_units(
    lines: list[str],
    config: ChunkingConfig,
) -> list[_Unit]:
    units: list[_Unit] = []

    if any(RECORD_ROW.match(line) for line in lines):
        groups: list[list[str]] = []
        for line in lines:
            if not _is_table_row(line) or not groups:
                groups.append([line])
            else:
                groups[-1].append(line)

        for group in groups:
            record = "\n".join(group)
            if len(record) <= config.target_chars or len(group) == 1:
                units.append(_Unit(record, "\n", "row"))
                continue

            # Do not mistake an unlabeled record's first row for a title.
            if _is_table_row(group[0]):
                units.extend(
                    _Unit(row, "\n", "row") for row in group
                )
                continue

            title = group[0]
            units.extend(
                _Unit(row, "\n", "row", header=title)
                for row in group[1:]
            )

        return units

    first_row = next(
        index
        for index, line in enumerate(lines)
        if _is_table_row(line)
    )
    for line in lines[:first_row]:
        units.extend(_line_units(line, config, first_sep="\n"))

    header_lines = [lines[first_row]]
    data_start = first_row + 1

    if (
        data_start < len(lines)
        and _is_separator_row(lines[data_start])
    ):
        # Standard Markdown: one header row plus its separator.
        header_lines.append(lines[data_start])
        data_start += 1
    elif not lines[first_row].lstrip().startswith("|"):
        # Exported tables can have continuation header lines.
        while (
            data_start < len(lines)
            and lines[data_start].lstrip().startswith("|")
            and not _is_separator_row(lines[data_start])
        ):
            header_lines.append(lines[data_start])
            data_start += 1

    base_header = "\n".join(header_lines)
    active_header = base_header
    pending_notes: list[str] = []
    emitted_row = False

    for line in lines[data_start:]:
        if _is_separator_row(line):
            continue

        if not _is_table_row(line):
            if LIST_ITEM.match(line):
                # Footnotes are content, not repeatable table headers.
                for note in pending_notes:
                    units.extend(
                        _line_units(note, config, first_sep="\n")
                    )
                pending_notes.clear()

                units.extend(
                    _line_units(line, config, first_sep="\n")
                )
                active_header = base_header
            else:
                # Preserve unmarked applicability notes as row context.
                pending_notes.append(line)

            continue

        if pending_notes:
            active_header = (
                base_header + "\n" + "\n".join(pending_notes)
            )
            pending_notes.clear()

        units.append(
            _Unit(line, "\n", "row", header=active_header)
        )
        emitted_row = True

    if not emitted_row:
        units.append(_Unit(base_header, "\n", "header"))

    # Preserve trailing footnotes as ordinary text.
    for line in pending_notes:
        units.extend(_line_units(line, config, first_sep="\n"))

    return units


def pack_units(
    units: list[_Unit],
    config: ChunkingConfig,
) -> list[str]:
    pieces: list[str] = []
    current: list[_Unit] = []
    carried = False

    for unit in units:
        has_new_content = len(current) > int(carried)
        if (
            has_new_content
            and len(_render([*current, unit])) > config.target_chars
        ):
            pieces.append(_render(current))
            overlap = _overlap(current[-1], unit, config)
            carried = (
                overlap is not None
                and len(_render([overlap, unit]))
                <= config.target_chars
            )
            current = [overlap] if carried else []

        current.append(unit)

    if len(current) > int(carried):
        pieces.append(_render(current))

    return pieces


def _render(units: list[_Unit]) -> str:
    """Insert each table's header before its rows within this piece."""
    parts: list[str] = []
    active_header: str | None = None

    for unit in units:
        if unit.header and unit.header != active_header:
            if parts:
                parts.append("\n\n")
            parts.append(unit.header)
            parts.append("\n")
            parts.append(unit.text)
            active_header = unit.header
            continue

        if parts:
            parts.append(unit.sep)
        parts.append(unit.text)

        if not unit.header:
            active_header = (
                unit.text if unit.kind == "header" else None
            )

    return "".join(parts)


def _overlap(
    last: _Unit,
    following: _Unit,
    config: ChunkingConfig,
) -> _Unit | None:
    if (
        config.overlap_max_chars == 0
        or last.kind != "prose"
        or following.kind != "prose"
    ):
        return None

    sentences = split_sentences(last.text)
    if not sentences:
        return None

    sentence = sentences[-1]
    if len(sentence) > config.overlap_max_chars:
        return None

    return _Unit(sentence, "", "prose")


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_BOUNDARY.split(text.strip())
    sentences: list[str] = []

    for part in parts:
        if not part:
            continue
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)

    return sentences


def _ends_with_abbreviation(text: str) -> bool:
    words = text.rsplit(maxsplit=1)
    if not words:
        return False

    last_word = words[-1].lower().lstrip('(["“')
    return (
        last_word in ABBREVIATIONS
        or bool(SINGLE_INITIAL.search(text))
    )


def hard_wrap(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current = ""

    for word in text.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]

        if not word:
            continue

        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word

    if current:
        pieces.append(current)

    return pieces


def context_header(section: Section) -> str:
    parts: list[str] = []
    if section.doc.drug:
        parts.append(f"Drug: {section.doc.drug.title()}")
    parts.append(f"Source: {section.doc.source_label}")
    parts.append(f"Section: {section.path_str}")
    if section.boxed_warning:
        parts.append("BOXED WARNING")
    return " | ".join(parts)


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-.")
    if len(slug) > max_len:
        shortened = slug[:max_len]
        slug = shortened.rsplit("-", 1)[0] or shortened
    return slug or "section"


def _unique(candidate: str, used: set[str]) -> str:
    unique = candidate
    number = 2

    while unique in used:
        unique = f"{candidate}-{number}"
        number += 1

    used.add(unique)
    return unique