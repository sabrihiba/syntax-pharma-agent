"""Minimum two-label evidence gate for explicit generic-drug interactions.

This checks retrieval coverage, not clinical interaction correctness.
Brand resolution and broader class coverage can be added separately.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations

from app.agent.consolidation import SectionEvidence


# Class aliases, not clinical conclusions or answer templates.
CLASS_TERMS = {
    "sertraline": (
        "selective serotonin reuptake inhibitor",
        "serotonin reuptake inhibitor",
        "ssri",
    ),
    "warfarin": ("anticoagulant",),
    "apixaban": ("anticoagulant",),
}

INTERACTION_INTENT = re.compile(
    r"\b(?:interact\w*|concomitant\w*|coadminist\w*|co-administ\w*|"
    r"together|combined|combination|prescrib\w*)\b|"
    r"\b(?:taking|take|used?|use|given)\b.{0,100}\bwith\b"
)


@dataclass(frozen=True)
class InteractionEvidenceCheck:
    source_drug: str
    target_drug: str
    section_ids: tuple[str, ...]
    match_kind: str

    @property
    def supported(self) -> bool:
        return bool(self.section_ids)

    @property
    def gap(self) -> str:
        terms = ", ".join(
            (
                self.target_drug,
                *CLASS_TERMS.get(self.target_drug, ()),
            )
        )
        return (
            f"Required interaction evidence: the {self.source_drug} DailyMed "
            f"label needs a complete warning or interaction section naming "
            f"{self.target_drug} or its mapped class. Search "
            f"{self.source_drug} drug interactions for {terms}; read the "
            "relevant parent section. Generic bleeding or monitoring warnings "
            "alone do not satisfy this two-label check. A measured INR or "
            "prothrombin-time effect is not required unless the user asks for "
            "it or it is necessary for an accurate answer."
        )


def _normalized(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", text).casefold(),
    ).strip()


def _mentions(
    text: str,
    term: str,
    *,
    plural: bool = False,
) -> bool:
    suffix = r"s?" if plural else ""

    return bool(
        re.search(
            r"(?<!\w)"
            + re.escape(_normalized(term))
            + suffix
            + r"(?!\w)",
            _normalized(text),
        )
    )


def _relevant(section: SectionEvidence) -> bool:
    number = section.citation.section_number or ""
    path = _normalized(section.citation.section_path)

    return (
        section.source.casefold() == "dailymed"
        and section.complete
        and bool(section.full_text)
        and (
            number.split(".")[0] in {"5", "7"}
            or bool(
                re.search(
                    r"interact|warning|precaution",
                    path,
                )
            )
        )
    )


def interaction_evidence_checks(
    question: str,
    sections: tuple[SectionEvidence, ...],
) -> tuple[InteractionEvidenceCheck, ...]:
    """Check counterpart mentions in complete relevant label sections.

    Only clear interaction wording and explicitly mentioned generic names
    activate this gate. Other questions still receive the normal model review.
    """
    normalized_question = _normalized(question)

    if not INTERACTION_INTENT.search(normalized_question):
        return ()

    known = set(CLASS_TERMS)
    known.update(
        _normalized(section.drug)
        for section in sections
        if section.drug
    )

    drugs = sorted(
        drug
        for drug in known
        if _mentions(normalized_question, drug)
    )

    if len(drugs) < 2:
        return ()

    checks = []

    for first, second in combinations(drugs, 2):
        for source_drug, target_drug in (
            (first, second),
            (second, first),
        ):
            candidates = [
                section
                for section in sections
                if section.drug
                and _normalized(section.drug) == source_drug
                and _relevant(section)
            ]

            direct = tuple(
                section.section_id
                for section in candidates
                if _mentions(
                    section.full_text,
                    target_drug,
                )
            )

            class_matches = ()

            if not direct:
                # A shared class word might describe the source drug itself.
                # Do not count that alone as evidence about the counterpart.
                shared_classes = set(
                    CLASS_TERMS.get(source_drug, ())
                )
                target_classes = [
                    term
                    for term in CLASS_TERMS.get(target_drug, ())
                    if term not in shared_classes
                ]

                class_matches = tuple(
                    section.section_id
                    for section in candidates
                    if any(
                        _mentions(
                            section.full_text,
                            term,
                            plural=True,
                        )
                        for term in target_classes
                    )
                )

            checks.append(
                InteractionEvidenceCheck(
                    source_drug=source_drug,
                    target_drug=target_drug,
                    section_ids=direct or class_matches,
                    match_kind=(
                        "drug"
                        if direct
                        else "class"
                        if class_matches
                        else "missing"
                    ),
                )
            )

    return tuple(checks)