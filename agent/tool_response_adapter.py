"""Adapt text-based tool calls into structured OpenAI-format tool_calls."""

import json
import os
from types import SimpleNamespace
from uuid import uuid4

from agent.tool_call_parser import content_after_tool_calls, has_tool_calls, parse_tool_calls


def should_adapt(response, model: str) -> bool:
    msg = response.choices[0].message
    has_existing = getattr(msg, "tool_calls", None)
    if has_existing:
        return False
    content = getattr(msg, "content", None) or ""
    if not has_tool_calls(content):
        return False
    return model.startswith("local/") or bool(os.environ.get("HERMES_FORCE_TOOL_INJECTION"))


def adapt_response(response, tools: list[dict]):
    msg = response.choices[0].message
    content = getattr(msg, "content", None) or ""

    parsed = parse_tool_calls(content)
    if not parsed:
        return response

    valid_names = {t["function"]["name"] for t in tools if "function" in t}

    mock_calls = []
    for tc in parsed:
        if tc.name not in valid_names:
            continue
        mock_calls.append(SimpleNamespace(
            id=f"call_{uuid4().hex[:8]}",
            type="function",
            function=SimpleNamespace(
                name=tc.name,
                arguments=json.dumps(tc.arguments),
            ),
        ))

    if not mock_calls:
        return response

    msg.tool_calls = mock_calls
    msg.content = content_after_tool_calls(content) or None
    response.choices[0].finish_reason = "tool_calls"
    return response


def format_tool_results_as_content(tool_results: list[dict]) -> str:
    parts = []
    for r in tool_results:
        payload = json.dumps({"name": r.get("name", ""), "content": r.get("content", "")})
        parts.append(f"<tool_response>\n{payload}\n</tool_response>")
    return "\n".join(parts)
