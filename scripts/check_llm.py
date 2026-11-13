"""Check tool calling through the configured OpenAI-compatible endpoint.

Normally makes two chat API requests:
1. Ask the model to call a harmless local calculator.
2. Return the tool result and check that the model uses it.

No corpus loading or embedding requests.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from app import config as settings


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers using a local calculator.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]


def check_tool_calling() -> None:
    if not settings.API_KEY:
        raise ValueError("OPENAI_API_KEY is missing")

    if not settings.MODEL:
        raise ValueError("OPENAI_MODEL is missing")

    messages = [
        {
            "role": "system",
            "content": (
                "You are testing a local tool integration. "
                "When asked to use a tool, call it and use its result."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use add_numbers exactly once to calculate 19 + 23. "
                "After receiving the tool result, report the sum."
            ),
        },
    ]

    print(f"Checking model: {settings.MODEL}")

    with OpenAI(
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL or None,
        timeout=30.0,
        max_retries=1,
    ) as client:
        response = client.chat.completions.create(
            model=settings.MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        calls = message.tool_calls or []

        if len(calls) != 1:
            raise RuntimeError(
                f"Expected one tool call, received {len(calls)}. "
                "This run did not verify tool calling; inspect the "
                "model and proxy configuration before proceeding."
            )

        call = calls[0]

        if call.type != "function":
            raise RuntimeError(
                f"Unexpected tool-call type: {call.type}"
            )

        if call.function.name != "add_numbers":
            raise RuntimeError(
                f"Unexpected tool: {call.function.name}"
            )

        arguments = json.loads(call.function.arguments)

        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")

        if set(arguments) != {"a", "b"}:
            raise ValueError("Expected exactly the arguments a and b")

        if any(
            type(arguments[name]) is not int
            for name in ("a", "b")
        ):
            raise ValueError("Tool arguments must be integers")

        if sorted(arguments.values()) != [19, 23]:
            raise ValueError(
                "The model did not pass the requested numbers"
            )

        # Execute the local tool ourselves.
        result = arguments["a"] + arguments["b"]

        print("PASS: model returned the correct tool and arguments")
        print(f"PASS: local tool executed; sum={result}")

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        })

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({"sum": result}),
        })

        final_response = client.chat.completions.create(
            model=settings.MODEL,
            messages=messages,
        )

        answer = final_response.choices[0].message.content or ""

        if not re.search(r"\b42\b", answer):
            raise RuntimeError(
                "The model received the tool result but did not "
                "report the expected sum of 42."
            )

        print("PASS: model accepted the tool result and reported 42")
        print("\nTool-calling round trip passed.")


if __name__ == "__main__":
    try:
        check_tool_calling()
    except Exception as exc:
        print(f"\nCHECK FAILED: {type(exc).__name__}: {exc}")
        print(
            "A failure may indicate configuration, schema, or proxy "
            "compatibility issues; it does not by itself prove that "
            "the model lacks tool support."
        )
        raise SystemExit(1)