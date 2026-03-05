#!/usr/bin/env python3
"""Thin wrapper around AIAgent for testbed use.

Provides a sandboxed, configurable agent with sensible defaults for
experimentation: cheap model, limited toolsets, quiet output, trajectory
capture, and structured results.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_agent import AIAgent


def _check_api_key(base_url: str | None = None):
    """Warn early if no API key is configured."""
    if base_url and base_url.startswith("http://localhost"):
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        env_file = Path.home() / ".hermes" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return  # will be loaded by dotenv
        print("WARNING: No OPENROUTER_API_KEY found.")
        print("  Set it via: export OPENROUTER_API_KEY=sk-or-...")
        print(f"  Or add it to {env_file}\n")


# Fast and cheap default for testbed use
DEFAULT_MODEL = "google/gemini-2.0-flash"
DEFAULT_TOOLSETS = ["file"]
DEFAULT_MAX_ITERATIONS = 15


class TestbedAgent:
    """Managed agent instance for testbed experiments."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        toolsets: list[str] | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        system_prompt: str | None = None,
        verbose: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.toolsets = toolsets or list(DEFAULT_TOOLSETS)
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.base_url = base_url
        self.api_key = api_key
        self.history: list[dict] = []
        self._agent: AIAgent | None = None

    @classmethod
    def local(cls, port=8787, **kwargs):
        return cls(
            model="mistral-7b-local",
            base_url=f"http://localhost:{port}/v1",
            api_key="local",
            **kwargs,
        )

    def _ensure_agent(self) -> AIAgent:
        if self._agent is None:
            _check_api_key(self.base_url)
            self._agent = AIAgent(
                model=self.model,
                enabled_toolsets=self.toolsets,
                max_iterations=self.max_iterations,
                ephemeral_system_prompt=self.system_prompt,
                quiet_mode=not self.verbose,
                verbose_logging=self.verbose,
                skip_context_files=True,
                skip_memory=True,
                tool_delay=0.0,
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._agent

    def ask(self, message: str, conversation_history: list[dict] | None = None) -> dict:
        """Send a message and return structured result.

        Returns dict with keys:
            response: str - final text response
            tool_calls: list - tools that were invoked
            turns: int - number of API calls made
            elapsed: float - wall time in seconds
            completed: bool
        """
        agent = self._ensure_agent()
        t0 = time.monotonic()
        raw = agent.run_conversation(
            user_message=message,
            conversation_history=conversation_history,
        )
        elapsed = time.monotonic() - t0

        # Extract tool calls from message history
        tool_calls = []
        for msg in raw.get("messages", []):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_calls.append({
                        "tool": fn.get("name"),
                        "args": fn.get("arguments"),
                    })

        result = {
            "response": raw.get("final_response") or "",
            "tool_calls": tool_calls,
            "turns": raw.get("api_calls", 0),
            "elapsed": round(elapsed, 2),
            "completed": raw.get("completed", False),
            "messages": raw.get("messages", []),
        }
        self.history.append({"query": message, "result": result})
        return result

    def chat(self, message: str) -> str:
        """Simple string-in, string-out interface."""
        return self.ask(message)["response"]

    def multi_turn(self, messages: list[str]) -> list[dict]:
        """Run a multi-turn conversation, carrying history forward."""
        conversation = []
        results = []
        for msg in messages:
            result = self.ask(msg, conversation_history=conversation)
            conversation = result["messages"]
            results.append(result)
        return results

    def reset(self):
        """Discard agent state and history."""
        self._agent = None
        self.history = []

    def dump_history(self, path: str | None = None) -> str:
        """Dump run history to JSON. Returns the JSON string."""
        data = json.dumps(self.history, indent=2, default=str)
        if path:
            Path(path).write_text(data)
        return data
