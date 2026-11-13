# Instructions for coding assistants

## Status

This file was added during project finalization on 2026-11-13.
It was not present during the initial development, which was guided
through conversation instructions.

## Project

This is a corpus-grounded pharmacology research assistant.

- Python 3.11 or newer.
- FastAPI provides the HTTP API.
- LangGraph orchestrates research, evidence review, answer generation,
  grounding review, and bounded revisions.
- Retrieval combines vector similarity and BM25 using Reciprocal Rank Fusion.
- The application answers from the supplied local corpus.

## Repository structure

- `app/agent/`: planning, tools, evidence review, and answer workflows.
- `app/rag/`: corpus parsing, chunking, embeddings, caching, and retrieval.
- `app/static/`: frontend.
- `corpus/`: supplied source documents.
- `tests/`: automated tests.
- `scripts/`: manual diagnostic scripts.

## Working agreement

- Follow the user's requested scope.
- By default, provide code and commands for the user to copy.
- Do not modify files, commit, or push without explicit authorization.
- Inspect relevant code before proposing changes.
- Explain what changed and how it was verified.
- Never claim that a test or deployment passed unless it was observed.

## Evidence and answer quality

- Preserve the supplied corpus.
- Do not hardcode production answers or tool sequences for evaluation questions.
- Let the model choose among the available tools.
- Preserve complete source sections when completeness is required.
- Use pagination for large sections.
- Preserve qualifications, exceptions, numeric operators, and units.
- Do not treat warnings as contraindications.
- Missing evidence does not establish safety or absence of an interaction.
- Release answers only after the required reviews pass.
- Retain source citations and the educational-use disclaimer.

## Validation

Run commands from the repository root:

```bash
python -m pip check
python -m pytest -q