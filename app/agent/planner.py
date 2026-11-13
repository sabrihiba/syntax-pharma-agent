"""Create a concise evidence-gathering plan."""

from __future__ import annotations

import json


PLAN_PROMPT = """Return only a JSON object with one key: "steps".
Its value is a list of concise evidence-gathering subquestions, usually 2 to 4
steps and never more than 6.

The ORIGINAL USER QUESTION defines required scope. The plan is a guide,
not a new user request. Additional information may be explored when useful for
accuracy, ambiguity, context, or an important qualification. Stop once the
required evidence is sufficient to answer the original question accurately.
Do not turn optional background into a mandatory research task. A detail is
required only when explicitly requested or necessary to avoid a materially
incorrect, incomplete, or misleading answer. Important exceptions, relevant
numeric conditions, and necessary qualifications remain required.
For a risk-and-monitoring question, do not require a measured direction,
magnitude, or absence of a laboratory effect solely because the plan mentions
it. Such measurements become required if the user requests them or they are
necessary to answer accurately. If not established, do not invent them.

Write steps around required answer needs. Useful optional exploration may occur
during research without adding a mandatory plan step. Avoid broad phrases such
as "all pharmacokinetic effects" when the question only asks risk and monitoring.
For comparisons retrieve each requested drug's relevant section, then compare.
For interactions cover the named drugs, the requested risk and monitoring.
Do not automatically add mechanisms, measured lab effects, counseling, dosing,
or thresholds unless requested or necessary for an accurate answer.
Preserve important source qualifications and exceptions. Treat the corpus as a
snapshot; do not call it current prescribing information. Do not answer the
medical question, provide private reasoning, or invent facts."""


class AgentProtocolError(RuntimeError):
    pass


def serialized_size(value) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def validate_question(question: str) -> None:
    if (
        not isinstance(question, str)
        or not question.strip()
        or len(question) > 4000
    ):
        raise ValueError(
            "question must contain 1 to 4000 characters"
        )


def create_plan(
    question: str,
    *,
    client,
    model: str,
    max_context_bytes: int = 48000,
    max_output_tokens: int = 2000,
) -> tuple[str, ...]:
    validate_question(question)

    request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": PLAN_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "max_completion_tokens": max_output_tokens,
    }

    if serialized_size(request) > max_context_bytes:
        raise ValueError(
            "planning request exceeds the context byte budget"
        )

    response = client.chat.completions.create(**request)

    if not response.choices:
        raise AgentProtocolError(
            "Planner returned no choices"
        )

    choice = response.choices[0]

    if choice.finish_reason == "length":
        raise AgentProtocolError(
            "Planner reached its output token limit"
        )

    content = choice.message.content

    if not isinstance(content, str) or len(content) > 8000:
        raise AgentProtocolError(
            "Planner returned invalid or oversized content"
        )

    try:
        payload = json.loads(content)
    except ValueError as exc:
        raise AgentProtocolError(
            "Planner did not return valid JSON"
        ) from exc

    if (
        not isinstance(payload, dict)
        or set(payload) != {"steps"}
    ):
        raise AgentProtocolError(
            'Planner must return {"steps": [...]}'
        )

    steps = payload["steps"]

    if (
        not isinstance(steps, list)
        or not 1 <= len(steps) <= 6
        or any(
            not isinstance(step, str)
            or not step.strip()
            or len(step) > 500
            for step in steps
        )
    ):
        raise AgentProtocolError(
            "Planner must return 1 to 6 nonempty short steps"
        )

    return tuple(step.strip() for step in steps)