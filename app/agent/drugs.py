"""Resolve generic names and aliases explicitly documented in the corpus.

This helper does not guess brand names or infer complete product ingredients.
An alias mentioned in a monograph does not prove product equivalence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from app.rag.chunker import Corpus


ResolutionStatus = Literal[
    "resolved",
    "ambiguous",
    "combination",
    "not_found",
]

AliasKind = Literal[
    "generic",
    "brand",
    "other",
]

BULLET = re.compile(r"^\s*[-*•]\s+(.+)$")
MARKERS = re.compile(r"[®™¶†‡]")
COMBINATION = re.compile(
    r"\b(?:combination|containing)\b",
    re.IGNORECASE,
)


def normalize_drug_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("drug name must be a string")

    normalized = unicodedata.normalize(
        "NFKC",
        name,
    ).casefold()

    normalized = MARKERS.sub("", normalized)

    return " ".join(
        re.findall(r"[^\W_]+", normalized)
    )


@dataclass(frozen=True)
class AliasRecord:
    alias: str
    drug: str
    kind: AliasKind
    file: str
    section_id: str | None
    original_text: str
    is_combination: bool = False


@dataclass(frozen=True)
class DrugResolution:
    query: str
    status: ResolutionStatus
    drug: str | None
    candidates: tuple[str, ...]
    matches: tuple[AliasRecord, ...]
    message: str

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def is_combination(self) -> bool:
        return any(
            record.is_combination
            for record in self.matches
        )


class DrugResolver:
    def __init__(self, corpus: Corpus) -> None:
        self._drugs = tuple(corpus.drugs)
        self._aliases: dict[
            str,
            list[AliasRecord],
        ] = {}

        # Generic names come from document metadata.
        for doc in corpus.documents:
            if doc.drug:
                self._add(
                    AliasRecord(
                        alias=doc.drug,
                        drug=doc.drug,
                        kind="generic",
                        file=doc.file,
                        section_id=None,
                        original_text=doc.drug,
                    )
                )

        # Only explicit alias sections contribute alternative names.
        for section in corpus.sections:
            if (
                section.doc.source != "medlineplus"
                or not section.doc.drug
            ):
                continue

            title = section.title.strip().casefold()

            if title not in {
                "brand names",
                "brand names of combination products",
                "other names",
            }:
                continue

            kind: AliasKind = (
                "other"
                if title == "other names"
                else "brand"
            )
            combination_heading = "combination" in title

            for line in section.text.splitlines():
                match = BULLET.match(line)

                if match is None:
                    continue

                original = match.group(1).strip()

                # Parentheses describe formulation or ingredients.
                alias = re.split(
                    r"\s*\(",
                    original,
                    maxsplit=1,
                )[0]

                alias = MARKERS.sub("", alias).strip()

                if not normalize_drug_name(alias):
                    continue

                self._add(
                    AliasRecord(
                        alias=alias,
                        drug=section.doc.drug,
                        kind=kind,
                        file=section.doc.file,
                        section_id=section.id,
                        original_text=original,
                        is_combination=(
                            combination_heading
                            or bool(
                                COMBINATION.search(original)
                            )
                        ),
                    )
                )

    @property
    def drugs(self) -> tuple[str, ...]:
        return self._drugs

    def _add(self, record: AliasRecord) -> None:
        key = normalize_drug_name(record.alias)

        if key:
            records = self._aliases.setdefault(
                key,
                [],
            )

            if record not in records:
                records.append(record)

    def resolve_drug(self, name: str) -> DrugResolution:
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 200
        ):
            raise ValueError(
                "drug name must contain 1 to 200 characters"
            )

        key = normalize_drug_name(name)

        if not key:
            raise ValueError(
                "drug name must contain letters or numbers"
            )

        matches = tuple(
            sorted(
                self._aliases.get(key, ()),
                key=lambda record: (
                    record.drug,
                    record.file,
                    record.section_id or "",
                    record.alias,
                ),
            )
        )

        candidates = tuple(
            sorted({
                record.drug
                for record in matches
            })
        )

        if not matches:
            return DrugResolution(
                query=name,
                status="not_found",
                drug=None,
                candidates=(),
                matches=(),
                message=(
                    "Name not recognized from the corpus's generic "
                    "or alias entries. This does not establish that "
                    "the drug is medically equivalent to another name "
                    "or that no interaction exists."
                ),
            )

        if any(
            record.is_combination
            for record in matches
        ):
            return DrugResolution(
                query=name,
                status="combination",
                drug=None,
                candidates=candidates,
                matches=matches,
                message=(
                    "This name is documented as a combination product. "
                    "Candidates identify corpus monographs that mention "
                    "it, not a complete ingredient list. Do not substitute "
                    "a single-drug label for the full product; clarify "
                    "its ingredients and formulation."
                ),
            )

        if len(candidates) != 1:
            return DrugResolution(
                query=name,
                status="ambiguous",
                drug=None,
                candidates=candidates,
                matches=matches,
                message=(
                    "This name maps to multiple corpus drugs; "
                    "ask for clarification."
                ),
            )

        return DrugResolution(
            query=name,
            status="resolved",
            drug=candidates[0],
            candidates=candidates,
            matches=matches,
            message=(
                "Resolved from explicit corpus metadata "
                "or alias entries."
            ),
        )