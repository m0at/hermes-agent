#!/usr/bin/env python3
"""Interactive REPL for experimenting with the Hermes agent.

Usage:
    python3 testbed/repl.py                          # interactive REPL
    python3 testbed/repl.py --query "hello"           # single query
    python3 testbed/repl.py --toolsets terminal,file   # enable tools
    python3 testbed/repl.py --model openai/gpt-4o     # pick model
    python3 testbed/repl.py --unsafe                   # all tools
    python3 testbed/repl.py --local                    # local model at localhost:8787
    python3 testbed/repl.py --local --port 9000        # local model, custom port
    python3 testbed/repl.py --base-url http://x/v1     # custom endpoint
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TestbedAgent, DEFAULT_MODEL, DEFAULT_TOOLSETS


def _fetch_stats(port: int) -> dict | None:
    """Grab latest inference stats from the proxy."""
    try:
        import urllib.request
        r = urllib.request.urlopen(f"http://localhost:{port}/stats", timeout=1)
        return json.loads(r.read())
    except Exception:
        return None


def _format_stats(stats: dict) -> str:
    """One-line dim summary of inference perf."""
    parts = []
    if "decode_tps" in stats:
        parts.append(f"{stats['decode_tps']:.1f} tok/s")
    if "ms_per_token" in stats:
        parts.append(f"{stats['ms_per_token']:.0f}ms/tok")
    if "total_ms" in stats:
        secs = stats["total_ms"] / 1000
        parts.append(f"{secs:.1f}s total")
    if "backend" in stats:
        parts.append(stats["backend"])
    return " · ".join(parts)


def print_result(result: dict, verbose: bool = False, port: int = 8787):
    """Pretty-print an agent result."""
    _d = lambda t: f"\033[2m{t}\033[0m"
    _m = lambda t: f"\033[35m{t}\033[0m"
    _p = lambda t: f"\033[38;5;141m{t}\033[0m"
    if result["tool_calls"]:
        n = len(result["tool_calls"])
        info = f"{n} tool{'s' if n != 1 else ''} · {result['turns']} turn{'s' if result['turns'] != 1 else ''} · {result['elapsed']}s"
        print(f"\n  {_d(info)}")
        if verbose:
            for tc in result["tool_calls"]:
                args_preview = tc["args"][:80] if isinstance(tc["args"], str) else json.dumps(tc["args"])[:80]
                print(f"    {_m('→')} {_m(tc['tool'])}{_d('(' + args_preview + ')')}")
    print()
    print(f"  {_p('◇')} {result['response']}")
    stats = _fetch_stats(port)
    if stats:
        print(f"  {_d(_format_stats(stats))}")
    print()


def _c(code: int, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def _dim(t: str) -> str: return _c(2, t)
def _bold(t: str) -> str: return _c(1, t)
def _cyan(t: str) -> str: return _c(36, t)
def _magenta(t: str) -> str: return _c(35, t)
def _yellow(t: str) -> str: return _c(33, t)
def _green(t: str) -> str: return _c(32, t)
def _pink(t: str) -> str: return f"\033[38;5;213m{t}\033[0m"
def _purple(t: str) -> str: return f"\033[38;5;141m{t}\033[0m"
def _sky(t: str) -> str: return f"\033[38;5;117m{t}\033[0m"


def print_banner(agent: TestbedAgent, endpoint_info: str | None = None):
    """Print a kawaii banner."""
    bar = _dim("─" * 42)
    print()
    print(f"  {bar}")
    print(f"    {_purple('✧')} {_bold(_pink('H E R M E S'))}  {_bold(_purple('T E S T B E D'))} {_purple('✧')}")
    print(f"  {bar}")
    print()
    print(f"  {_sky('◈')} model    {_bold(_cyan(agent.model))}")
    if endpoint_info:
        print(f"  {_sky('◈')} endpoint {_green(endpoint_info)}")
    print(f"  {_sky('◈')} tools    {_magenta(', '.join(agent.toolsets))}")
    print(f"  {_sky('◈')} turns    {_yellow(str(agent.max_iterations))}")
    print()
    print(f"  {_dim('quit')} exit  {_dim('·')}  {_dim('reset')} clear  {_dim('·')}  {_dim('history')} dump")
    print()


def run_repl(agent: TestbedAgent, verbose: bool, endpoint_info: str | None = None, port: int = 8787):
    """Interactive read-eval-print loop."""
    print_banner(agent, endpoint_info)

    conversation = []
    while True:
        try:
            user_input = input(f"{_pink('✦')} {_bold('you')} {_dim('›')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_pink('✧')} {_dim('bye bye~')}\n")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            conversation = []
            print(f"  {_dim('✧ state cleared')}\n")
            continue
        if user_input.lower() == "history":
            print(agent.dump_history())
            continue

        try:
            result = agent.ask(user_input, conversation_history=conversation)
            conversation = result["messages"]
            print_result(result, verbose=verbose, port=port)
        except KeyboardInterrupt:
            print(f"\n  {_dim('~ interrupted ~')}\n")
        except Exception as e:
            print(f"\n  \033[31m✗\033[0m {_dim(str(e))}\n")


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent Testbed REPL")
    parser.add_argument("--query", "-q", help="Single query (non-interactive)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--toolsets", "-t", default=",".join(DEFAULT_TOOLSETS),
                        help=f"Comma-separated toolsets (default: {','.join(DEFAULT_TOOLSETS)})")
    parser.add_argument("--unsafe", action="store_true", help="Enable all toolsets")
    parser.add_argument("--max-iterations", type=int, default=15, help="Max tool-calling iterations")
    parser.add_argument("--system-prompt", "-s", help="Custom system prompt")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool call details")
    parser.add_argument("--local", action="store_true", help="Use local model at localhost (default port 8787)")
    parser.add_argument("--port", type=int, default=8787, help="Port for local model (default: 8787)")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--api-key", help="API key for custom endpoint")
    args = parser.parse_args()

    toolsets = None if args.unsafe else args.toolsets.split(",")
    endpoint_info = None

    if args.local:
        agent = TestbedAgent.local(
            port=args.port,
            toolsets=toolsets,
            max_iterations=args.max_iterations,
            system_prompt=args.system_prompt,
            verbose=args.verbose,
        )
        endpoint_info = f"Using local model at localhost:{args.port}"
    elif args.base_url:
        agent = TestbedAgent(
            model=args.model,
            toolsets=toolsets,
            max_iterations=args.max_iterations,
            system_prompt=args.system_prompt,
            verbose=args.verbose,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        endpoint_info = args.base_url
    else:
        agent = TestbedAgent(
            model=args.model,
            toolsets=toolsets,
            max_iterations=args.max_iterations,
            system_prompt=args.system_prompt,
            verbose=args.verbose,
        )

    if args.query:
        result = agent.ask(args.query)
        print_result(result, verbose=args.verbose, port=args.port)
    else:
        run_repl(agent, verbose=args.verbose, endpoint_info=endpoint_info, port=args.port)


if __name__ == "__main__":
    main()
