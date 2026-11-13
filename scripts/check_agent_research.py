"""Manually inspect the model-driven evidence collection workflow."""

import argparse
from pathlib import Path

from app.agent.research import ResearchAgent
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
            "Compare the contraindications of warfarin and apixaban. "
            "What conditions are shared and which are unique to each drug?"
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

    agent = ResearchAgent(
        AgentTools(retriever)
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

    print("\nCOLLECTED EVIDENCE:")
    for record in result.evidence:
        output = record["output"]

        if record["tool"] == "get_section":
            print(
                "SECTION:",
                output["section_id"],
                f"range={output['start_char']}:{output['end_char']}",
                f"complete={output['section_complete']}",
                f"has_more={output['has_more']}",
            )
        else:
            for passage in output["results"]:
                print(
                    "SEARCH HIT:",
                    passage["section_id"],
                    f"complete={passage['section_complete']}",
                )

    print("\nSTATUS:", result.status)
    print("\nUNREVIEWED COLLECTION NOTES:")
    print(result.notes)


if __name__ == "__main__":
    main()