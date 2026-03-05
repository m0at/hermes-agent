#!/usr/bin/env python3
"""Run eval tasks from tasks.yaml and report results.

Usage:
    python3 testbed/eval_runner.py                   # run all tasks
    python3 testbed/eval_runner.py --tasks simple_chat,arithmetic
    python3 testbed/eval_runner.py --model openai/gpt-4o
    python3 testbed/eval_runner.py --output results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TestbedAgent, DEFAULT_MODEL


TASKS_FILE = Path(__file__).parent / "tasks.yaml"


def load_tasks(filter_ids: list[str] | None = None) -> list[dict]:
    tasks = yaml.safe_load(TASKS_FILE.read_text())
    if filter_ids:
        tasks = [t for t in tasks if t["id"] in filter_ids]
    return tasks


def check_result(task: dict, result: dict) -> tuple[bool, str]:
    """Check if a result passes the task's criteria. Returns (passed, reason)."""
    response = result.get("response", "")
    method = task.get("check", "contains")
    expected = task.get("expected", "")

    if method == "contains":
        if expected.lower() in response.lower():
            return True, f"found '{expected}'"
        return False, f"'{expected}' not in response"

    if method == "responds":
        if response and len(response.strip()) > 10:
            return True, f"got {len(response)} chars"
        return False, "empty or trivial response"

    return False, f"unknown check method: {method}"


def run_eval(
    model: str,
    filter_ids: list[str] | None = None,
    verbose: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    tasks = load_tasks(filter_ids)
    if not tasks:
        print("No tasks to run.")
        return []

    print(f"Running {len(tasks)} eval task(s) with {model}\n")
    print(f"{'ID':<25} {'Status':<8} {'Time':>6}  {'Detail'}")
    print("-" * 70)

    results = []
    passed = 0
    for task in tasks:
        agent_kwargs = dict(
            model=model,
            toolsets=task.get("toolsets", ["file"]),
            max_iterations=task.get("max_iterations", 15),
            verbose=verbose,
        )
        if base_url:
            agent_kwargs["base_url"] = base_url
            agent_kwargs["api_key"] = api_key or "local"
        agent = TestbedAgent(**agent_kwargs)

        try:
            result = agent.ask(task["query"])
            ok, reason = check_result(task, result)
        except Exception as e:
            result = {"response": "", "tool_calls": [], "turns": 0, "elapsed": 0, "completed": False}
            ok, reason = False, f"error: {e}"

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        print(f"{task['id']:<25} {status:<8} {result['elapsed']:>5.1f}s  {reason}")
        if verbose and not ok:
            preview = (result.get("response") or "")[:200]
            print(f"  response: {preview}")

        results.append({
            "id": task["id"],
            "passed": ok,
            "reason": reason,
            "elapsed": result["elapsed"],
            "turns": result["turns"],
            "tool_calls": len(result["tool_calls"]),
            "response_length": len(result.get("response", "")),
        })

    print("-" * 70)
    print(f"Result: {passed}/{len(tasks)} passed\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Hermes Testbed Eval Runner")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", "-t", help="Comma-separated task IDs to run")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--local", action="store_true", help="Use a local model server")
    parser.add_argument("--base-url", help="Custom OpenAI-compatible base URL")
    parser.add_argument("--port", type=int, default=8787, help="Port for --local (default: 8787)")
    args = parser.parse_args()

    filter_ids = args.tasks.split(",") if args.tasks else None

    base_url = args.base_url
    api_key = None
    if args.local:
        base_url = f"http://localhost:{args.port}/v1"
        api_key = "local"
    elif args.base_url:
        api_key = "local"

    results = run_eval(args.model, filter_ids, verbose=args.verbose, base_url=base_url, api_key=api_key)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
