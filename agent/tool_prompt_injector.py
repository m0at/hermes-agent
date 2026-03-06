"""Inject tool definitions into system prompts for models without structured tool calling."""

import json
import os


def format_tools_for_prompt(tools: list[dict]) -> str:
    tool_defs = []
    for tool in tools:
        func = tool.get("function", {})
        tool_defs.append(json.dumps({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        }))

    tools_block = "\n".join(tool_defs)

    return (
        "You have access to the following tools. To call a tool, output a tool_call XML block:\n"
        "\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"param": "value"}}\n'
        "</tool_call>\n"
        "\n"
        "Available tools:\n"
        "<tools>\n"
        f"{tools_block}\n"
        "</tools>\n"
        "\n"
        "After each tool call, you will receive the result in a <tool_response> block. "
        "You may call multiple tools. When you have the final answer, respond normally without tool_call tags."
    )


def inject_tools_into_system_prompt(system_prompt: str, tools: list[dict]) -> str:
    return system_prompt + "\n\n" + format_tools_for_prompt(tools)


def needs_tool_injection(model: str) -> bool:
    if os.environ.get("HERMES_FORCE_TOOL_INJECTION"):
        return True
    if model.startswith("local/"):
        return True
    if "qwen" in model.lower() and "openrouter" not in model.lower():
        return True
    return False


def format_tool_response(tool_call_id: str, name: str, result: str) -> str:
    return "<tool_response>" + json.dumps({"name": name, "result": result}) + "</tool_response>"
