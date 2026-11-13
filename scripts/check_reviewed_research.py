"""Live check of research, consolidation, and evidence review."""

import argparse
from pathlib import Path

from app.agent.research import AgentConfig
from app.agent.reviewed_research import (
    ReviewedResearchAgent,
    ReviewConfig,
)
from app.agent.tools import AgentTools
from app.rag.cache import CachedEmbedder
from app.rag.chunker import load_corpus
from app.rag.hybrid import HybridRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        nargs="?",
        default=(
            "A patient currently on warfarin is prescribed sertraline. "
            "What is the risk of bleeding and what monitoring is recommended?"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    corpus = load_corpus(root / "corpus")

    print("Loading cached retrieval index...")

    retriever = HybridRetriever.build(
        corpus,
        CachedEmbedder(),
    )

    agent = ReviewedResearchAgent(
        AgentTools(retriever),
        AgentConfig(max_turns=12),
        review_config=ReviewConfig(max_reviews=3),
    )

    result = agent.collect(args.question)

    print("\nQUESTION:", result.question)

    print("\nPLAN:")
    for index, step in enumerate(result.plan, start=1):
        print(f"{index}. {step}")

    print("\nTOOL TRACE:")
    for index, entry in enumerate(result.trace, start=1):
        print(
            f"{index}. {entry.name} "
            f"ok={entry.ok} delivered={entry.delivered}"
        )
        print("   Arguments:", entry.arguments)

    print("\nCONSOLIDATED EVIDENCE:")
    for section in result.sections:
        print(
            section.section_id,
            f"complete={section.complete}",
            f"missing_ranges={section.missing_ranges}",
        )

    print("\nCOVERAGE REVIEWS:")
    for index, review in enumerate(result.reviews, start=1):
        print(f"Review {index}: sufficient={review.sufficient}")

        for check in review.checks:
            print(
                f"  Step {check.step_index}: "
                f"supported={check.supported}"
            )
            print("   Sections:", check.section_ids)
            print("   Assessment:", check.explanation)

    print("\nFINAL STATUS:", result.status)
    print("LAST COLLECTION STATUS:", result.collection_status)
    print("NOTES:", result.notes)

    if result.reviews and not result.reviews[-1].sufficient:
        print("\nUNRESOLVED GAPS:")
        for gap in result.reviews[-1].gaps:
            print("-", gap)


if __name__ == "__main__":
    main()