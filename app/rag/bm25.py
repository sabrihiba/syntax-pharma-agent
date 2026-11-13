"""Local BM25 keyword retrieval over contextual chunk text.

No API calls or additional dependencies. Scores are positive BM25 scores,
not probabilities or cosine similarities. No stemming or synonym expansion.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from numbers import Real

from app.rag.chunker import Chunk, Citation, Corpus, Section


# Preserve decimals; split punctuation, hyphens, underscores, and unit slashes.
TOKEN = re.compile(r"\d+(?:\.\d+)+|[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN.findall(text)


def _finite_real(value) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if not _finite_real(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be positive and finite")

        if not _finite_real(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")


@dataclass(frozen=True)
class BM25Result:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> Citation:
        return self.chunk.citation


def _filter_values(
    values: Collection[str] | None,
    name: str,
) -> set[str] | None:
    if values is None:
        return None

    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{name} must be a collection of strings"
        )

    result = set()

    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{name} must contain nonempty strings"
            )

        result.add(value.strip().casefold())

    return result


class BM25Index:
    """Index Chunk.embed_text so drug names and headings remain searchable.

    Corpus statistics are calculated globally. Metadata filters restrict the
    candidate chunks without changing the meaning of their BM25 scores.
    """

    def __init__(
        self,
        corpus: Corpus,
        config: BM25Config | None = None,
    ) -> None:
        self.config = config or BM25Config()

        self._corpus = Corpus(
            documents=list(corpus.documents),
            sections=list(corpus.sections),
            chunks=list(corpus.chunks),
        )
        self._chunks = tuple(self._corpus.chunks)
        self._validate_chunks()

        self._postings: dict[
            str, list[tuple[int, int]]
        ] = defaultdict(list)

        lengths: list[int] = []

        for index, chunk in enumerate(self._chunks):
            terms = tokenize(chunk.embed_text)
            lengths.append(len(terms))

            for term, frequency in Counter(terms).items():
                self._postings[term].append(
                    (index, frequency)
                )

        self._lengths = tuple(lengths)
        self._average_length = (
            sum(lengths) / len(lengths)
            if lengths
            else 0.0
        )

        count = len(self._chunks)

        self._idf = {
            term: math.log1p(
                (count - len(postings) + 0.5)
                / (len(postings) + 0.5)
            )
            for term, postings in self._postings.items()
        }

    @classmethod
    def build(
        cls,
        corpus: Corpus,
        config: BM25Config | None = None,
    ) -> BM25Index:
        return cls(corpus, config)

    def _validate_chunks(self) -> None:
        ids: set[str] = set()

        for chunk in self._chunks:
            if chunk.id in ids:
                raise ValueError(
                    f"duplicate chunk id: {chunk.id}"
                )

            ids.add(chunk.id)
            parent = self._corpus.section(chunk.section_id)

            if parent is None:
                raise ValueError(
                    f"missing parent section: {chunk.section_id}"
                )

            if parent.doc != chunk.doc:
                raise ValueError(
                    f"parent document mismatch: {chunk.id}"
                )

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def average_length(self) -> float:
        return self._average_length

    def get_section(self, section_id: str) -> Section | None:
        return self._corpus.section(section_id)

    def search(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
        drugs: Collection[str] | None = None,
        sources: Collection[str] | None = None,
    ) -> list[BM25Result]:
        """Return matching chunks, ordered by descending BM25 score.

        Repeated query terms count once. Chunks with no term matches are
        excluded. Source filters use "dailymed", "medlineplus", etc.

        Scores are not bounded by 1 and are not clinical confidence scores.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "question must be a nonempty string"
            )

        if type(top_k) is not int or top_k <= 0:
            raise ValueError(
                "top_k must be a positive integer"
            )

        if min_score is not None and (
            not _finite_real(min_score)
            or min_score < 0
        ):
            raise ValueError(
                "min_score must be nonnegative and finite"
            )

        drug_set = _filter_values(drugs, "drugs")
        source_set = _filter_values(sources, "sources")

        candidates = {
            index
            for index, chunk in enumerate(self._chunks)
            if (
                drug_set is None
                or (chunk.doc.drug or "").casefold() in drug_set
            )
            and (
                source_set is None
                or chunk.doc.source.casefold() in source_set
            )
        }

        terms = sorted(set(tokenize(question)))

        if (
            not candidates
            or not terms
            or self._average_length == 0
        ):
            return []

        scores: dict[int, float] = defaultdict(float)
        k1, b = self.config.k1, self.config.b

        for term in terms:
            for index, frequency in self._postings.get(term, ()):
                if index not in candidates:
                    continue

                length_ratio = (
                    self._lengths[index]
                    / self._average_length
                )

                denominator = frequency + k1 * (
                    1 - b + b * length_ratio
                )

                scores[index] += (
                    self._idf[term]
                    * frequency
                    * (k1 + 1)
                    / denominator
                )

        # Equal scores retain original corpus order.
        ranked = sorted(
            (
                (index, score)
                for index, score in scores.items()
                if score > 0
                and (
                    min_score is None
                    or score >= min_score
                )
            ),
            key=lambda item: (-item[1], item[0]),
        )

        return [
            BM25Result(
                chunk=self._chunks[index],
                score=float(score),
            )
            for index, score in ranked[:top_k]
        ]