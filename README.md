# Pharmacology Research Agent

A corpus-grounded pharmacology assistant built with Python, FastAPI,
and LangGraph.

The agent plans evidence gathering, chooses retrieval tools, reviews
evidence coverage, drafts a cited answer, and checks its grounding before
releasing it.

Answers are for educational purposes only and are not medical advice.

## Requirements

- Python 3.11 or newer for local execution.
- Docker for container execution.
- API credentials for an OpenAI-compatible provider.
- A chat model supporting tool calling.
- An embedding model supported by the configured provider.
- Network access to the configured provider.

The supplied corpus contains 21 Markdown documents covering ten drugs
and a glossary. It is a local snapshot, not a live prescribing-information
service.

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Provider API key |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible API endpoint |
| `OPENAI_MODEL` | Chat model supporting tool calling |
| `OPENAI_EMBEDDING_MODEL` | Embedding model |

Use the exact endpoint and model identifiers supplied by your provider.

The chat and embedding models must be explicitly configured. There is
no silent model fallback.

Never commit `.env` or API credentials.

## Run with Docker

From the repository root:

```bash
docker build -t agent .
docker run -p 8000:8000 --env-file .env agent
```

Open:

- Frontend: http://localhost:8000
- API documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/health

The first startup may take several minutes because the application
loads the corpus and requests embeddings before becoming ready.

### Docker Compose

Alternatively:

```bash
docker compose up --build -d --wait --wait-timeout 600
docker compose ps
```

Compose binds the application to `127.0.0.1:8000` and stores embedding
cache files in a named volume.

Inspect logs:

```bash
docker compose logs -f pharma
```

Stop the application:

```bash
docker compose down
```

The named embedding-cache volume is retained by this command.

Stop the Compose service before starting a separate container on the
same port.

## Run locally

Run all commands from the repository root.

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Configure `.env` before starting the application.

## API

### POST /ask

Request:

```json
{
  "question": "What are the main contraindications for warfarin?"
}
```

The response contains:

- The original question and workflow status.
- A reviewed answer when available.
- Structured source citations.
- A research trace containing the plan, tool calls, and review summaries.
- An educational-use disclaimer.

An answer is released only when its status is `answer_reviewed`.
Incomplete research or unsuccessful answer review does not expose an
unapproved answer through the API.

The frontend renders the answer, links citation markers to source
cards, and provides an expandable research trace.

### GET /health

Returns application readiness:

```json
{
  "status": "ok",
  "ready": true
}
```

### Request handling

The application uses one shared retrieval index and accepts one active
question at a time per process. Concurrent requests may receive HTTP 503.

Invalid requests receive HTTP 422. Provider failures and timeouts are
returned as HTTP errors.

## Architecture

### Retrieval

1. Parse Markdown documents into hierarchical sections and chunks.
2. Generate embeddings for chunk retrieval.
3. Build an in-memory vector index and a BM25 index.
4. Merge their rankings using Reciprocal Rank Fusion.
5. Retrieve full parent sections when complete context is needed.

Search results can contain partial previews. The agent can retrieve
complete sections through pagination rather than treating a preview
as a complete medical list.

### LangGraph workflow

The research graph:

1. Generates an evidence-gathering plan.
2. Lets the model choose retrieval tools.
3. Executes validated tool calls.
4. Consolidates delivered evidence.
5. Reviews coverage and requests additional evidence when necessary.

The answer graph:

1. Runs the reviewed research workflow.
2. Generates a draft with source references.
3. Reviews source grounding and answer coverage.
4. Revises the draft when required, within a bounded revision limit.
5. Releases a reviewed answer or returns an incomplete status.

The available agent tools are:

- `search_corpus`: hybrid retrieval with optional drug/source filters.
- `get_section`: section retrieval with pagination.

Tool selection is model-driven. Execution limits bound turns, tool
calls, rejected-batch retries, evidence reviews, and answer revisions.

The displayed research trace contains execution summaries, not private
model reasoning.

## Embedding cache

Chunk and query embeddings are cached under `.cache/embeddings`.

Cache identity includes the configured model, endpoint, revision, and
exact embedding input. The retrieval index is rebuilt in memory at
startup while compatible cached embeddings are reused.

Docker Compose persists this cache through a named volume. No external
vector database is required.

## Project structure

```text
app/
  agent/              Planning, tools, reviews, and LangGraph workflows
  rag/                Chunking, embeddings, cache, and hybrid retrieval
  static/             Frontend
  config.py           Environment configuration
  main.py             FastAPI endpoints
  service.py          Application initialization and shared workflow

corpus/               Supplied Markdown corpus
tests/                Automated pytest tests
scripts/              Manual diagnostic checks

Dockerfile
compose.yaml
requirements.txt
pytest.ini
.env.example
```

## Automated tests

Install dependencies and run from the repository root:

```bash
python -m pytest -q
```

The default suite uses local or mocked checks. Live quality evaluations
are opt-in.

`pytest.ini` configures imports and a dedicated temporary directory at
`.cache/pytest-tmp`, which pytest clears between runs.

Observed default-suite result:

```text
744 passed, 20 skipped
```

The skipped cases comprise four public-question agent evaluations,
nine retrieval evaluations, and seven additional-question evaluations.

This result does not establish clinical correctness or guarantee
performance on unseen questions.

### Live public-question evaluation

These tests make API requests and may incur provider costs.

```bash
python -c "import os, pytest; os.environ['RUN_AGENT_QUALITY']='1'; raise SystemExit(pytest.main(['tests/test_agent_quality.py', '-v', '-s', '--tb=short']))"
```

### Live additional-question evaluation

These are practice generalization scenarios, not the assignment's
actual hidden questions.

```bash
python -c "import os, pytest; os.environ['RUN_UNSEEN_QUALITY']='1'; raise SystemExit(pytest.main(['tests/test_unseen_quality.py', '-v', '-s', '--tb=short']))"
```

The tests cover brand wording, additional drugs and interactions,
two-drug evidence, citation provenance, and withholding an unsupported
answer.

Evaluation rules are regression checks. They do not replace source
review or clinical assessment.

## Manual diagnostics

Run scripts as modules from the repository root:

```bash
python -m scripts.check_llm
python -m scripts.check_agent_research
python -m scripts.check_reviewed_research
python -m scripts.check_agent_answer "What are the main contraindications for warfarin?"
```

These scripts inspect tool calling, evidence collection, evidence
review, and the complete answer workflow. Live checks make API requests.

## Limitations

- Answers are limited to evidence available in the supplied corpus.
- The corpus may omit relevant drugs, conditions, or newer information.
- Missing evidence does not establish safety or absence of an interaction.
- Model-based evidence and grounding reviews can make mistakes.
- Additional-question tests cover selected scenarios, not every drug,
  brand name, interaction, or possible question.
- When sufficient evidence cannot be established, the workflow may stop
  without releasing a reviewed answer.
- Answers must not be used as individualized medical advice.
## AI-assisted development

See [AI_ASSISTANCE.md] for the assistance used and
the configuration timeline.

[AGENTS.md] contains coding-assistant instructions added
during project finalization.