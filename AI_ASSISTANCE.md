```markdown
# AI assistance and coding configuration

## Development assistance

Codex was used during development for:

- Architecture discussion and implementation proposals.
- Retrieval and LangGraph workflow code.
- Automated tests and evaluation rules.
- Diagnosis of test failures.
- API, frontend, Docker, and documentation guidance.
- Read-only inspection of the project and assignment.

The developer copied proposed changes into the project and ran local tests,
live evaluations, and Docker checks.

Some proposed corrections were also checked by the assistant in memory
before being supplied for copying. These checks are distinct from tests
run against the saved project files.

## Configuration timeline

During initial development, project instructions were supplied through
the conversation. No repository-level `AGENTS.md` existed at that time.

`AGENTS.md` was added during finalization on 2026-11-13. It records the
project's working instructions for subsequent coding-assistant sessions.
It is not presented as a configuration that existed throughout development.

## Main working instructions

The developer requested:

- Code to copy rather than direct file modifications.
- Incremental implementation and verification.
- LangGraph orchestration.
- Corpus-grounded answers with citations.
- Complete evidence when needed, including section pagination.
- Bounded retries and execution limits.
- General improvements rather than hardcoded answers to public questions.
- Protection of API credentials and local environment files.

## Skills and MCP configuration

No custom project-specific skill or MCP server configuration was created
for the initial development.

The assistant environment supplied tools and instructions for activities
such as code inspection, documentation lookup, and PDF review. These are
environment-managed capabilities, not dependencies of the submitted
application.

No machine-wide assistant configuration, credentials, or conversation
transcripts are included in this repository.

The application itself requires neither Codex nor an MCP server to run.
Its runtime configuration is documented in `.env.example` and `README.md`.

## Responsibility and limitations

Automated tests and model-based critics do not establish clinical
correctness or guarantee performance on hidden questions.