<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent (Fork)

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**A personal fork of [Nous Research's Hermes Agent](https://github.com/NousResearch/hermes-agent)** with local Qwen3.5-9B inference on Apple Silicon, CLI improvements, Rust-accelerated prompt scanning, and RL research tooling.

---

## What Changed from Upstream

### Local Qwen3.5-9B on Apple Silicon

Run Qwen3.5-9B locally via MLX-VLM — no cloud provider needed. Vision + text, 4-bit quantized, on GPU.

```bash
hermes --provider local --model local/qwen3.5-9b
```

The server auto-starts on port 8800 — no manual setup. Polls up to 60s while the model loads, shows stderr on crash. Serves OpenAI-compatible `/v1/chat/completions`.

**New files**: `local_models/serve.py`
**Modified**: `hermes_cli/runtime_provider.py`, `agent/model_metadata.py`

### CLI Terminal Improvements

- **Dynamic terminal resize** — horizontal rules and box borders recompute width at render time instead of hardcoded `'─' * 200`
- **Think block styling** — `<think>...</think>` tags render as dim italic gray with a `~ thinking ~` header, so local chain-of-thought models look clean
- **Image paste (Ctrl+I)** — detects macOS clipboard images, saves to `~/.hermes/images/`, converts to OpenAI vision format (base64 data URIs) for VLM
- **Color scheme picker** — choose between "cyber" (green/blue) and "synthwave" (pink/purple) on first launch

**Modified**: `cli.py`, `agent/display.py`
**New file**: `hermes_cli/color_scheme.py`

### Rust-Accelerated Prompt Scanning

`hermes_rs` — a PyO3 native module that replaces the Python regex injection scanner with a compiled Rust `RegexSet`. **17x faster** on real context files. Falls back to pure Python if not installed.

```bash
cd hermes_rs && maturin develop --release
```

**New directory**: `hermes_rs/`
**Modified**: `agent/prompt_builder.py`

### Context Compression Improvements

- Fallback client chain when primary auxiliary model fails (reads `OPENAI_BASE_URL`)
- Provider-aware `max_tokens` vs `max_completion_tokens` handling
- Better token estimation for compression preflight checks

**Modified**: `agent/context_compressor.py`

### Prompt Builder Hardening

- Injection detection for context files (`AGENTS.md`, `.cursorrules`, `SOUL.md`) with 10 threat patterns
- Head/tail truncation strategy (70% head, 20% tail) for oversized context files
- Skill index built from `~/.hermes/skills/` grouped by category

**Modified**: `agent/prompt_builder.py`

---

## Project Structure

```
hermes-agent/
├── agent/                  # Core agent modules
│   ├── auxiliary_client.py #   Shared LLM client for side tasks
│   ├── context_compressor.py # Auto context window compression
│   ├── display.py          #   Spinner, kawaii faces, formatting
│   ├── model_metadata.py   #   Token estimation, model context lengths
│   ├── prompt_builder.py   #   System prompt assembly + injection detection
│   ├── prompt_caching.py   #   Anthropic prompt caching
│   ├── redact.py           #   Secret redaction for logs
│   ├── skill_commands.py   #   Skill slash commands
│   └── trajectory.py       #   ShareGPT trajectory export
├── hermes_cli/             # CLI application
│   ├── main.py             #   Entry point + subcommand dispatch
│   ├── auth.py             #   Multi-provider OAuth + API key auth
│   ├── config.py           #   ~/.hermes/ configuration management
│   ├── setup.py            #   Interactive setup wizard
│   ├── gateway.py          #   Messaging gateway management
│   ├── runtime_provider.py #   Provider resolution + local model auto-start
│   ├── doctor.py           #   Diagnostic checks
│   ├── skills_hub.py       #   Skill search/install/manage
│   ├── color_scheme.py     #   Color theme picker
│   └── ...                 #   banner, callbacks, cron, models, etc.
├── hermes_rs/              # Rust-accelerated modules (PyO3)
│   └── src/                #   prompt_scanner.rs, token_estimate.rs
├── tools/                  # 40+ agent tools
├── skills/                 # 22 skill categories
├── gateway/                # Telegram, Discord, Slack, WhatsApp
├── environments/           # RL training environments
├── local_models/           # Qwen3.5-9B model server (MLX-VLM)
├── cron/                   # Job scheduler
├── cli.py                  # Interactive REPL (prompt_toolkit TUI)
├── run_agent.py            # Main AIAgent orchestrator
├── batch_runner.py         # Trajectory generation for training
├── trajectory_compressor.py # Trajectory compression for training
├── testbed/                # Local evaluation harness
└── tinker-atropos/         # RL training framework (Atropos/GRPO)
```

---

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Works on Linux, macOS, and WSL2. Handles Python, Node.js, dependencies, and the `hermes` command.

```bash
source ~/.bashrc    # reload shell
hermes setup        # configure your LLM provider
hermes              # start chatting
```

### Local Qwen3.5-9B (Apple Silicon)

After install, the local model deps are included by default. Just select local provider:

```bash
hermes model        # select "Local models" → Qwen3.5-9B
hermes              # server auto-starts, model loads (~60s first time)
```

Or start the server manually:

```bash
python3 -m local_models.serve qwen
hermes --provider local --model local/qwen3.5-9b
```

### Rust Module (Optional)

For 17x faster prompt injection scanning:

```bash
cd hermes_rs
uv pip install maturin
maturin develop --release
```

---

## Usage

```bash
hermes              # Interactive CLI
hermes model        # Switch provider or model
hermes setup        # Re-run setup wizard
hermes gateway      # Start messaging gateway
hermes status       # Show component status
hermes doctor       # Diagnose issues
hermes update       # Update to latest
hermes skills       # Search/install skills
hermes cron         # Manage scheduled jobs
hermes tools        # Configure tool access per platform
```

---

## RL Research

This fork includes a complete pipeline for reinforcement learning research with tool-calling agents. See [RL_RESEARCH_WITH_HERMES.md](RL_RESEARCH_WITH_HERMES.md) for the full guide.

```
Dataset -> Batch Runner -> Raw Trajectories -> Compressor -> GRPO Training -> Fine-tuned Model
```

**Quick start (SFT data, no GPU)**:
```bash
python3 batch_runner.py --dataset prompts.jsonl --run-name my-run --workers 4
python3 trajectory_compressor.py --input data/my-run/trajectories.jsonl \
  --output data/my-run/compressed.jsonl --max-tokens 15000
```

---

## Documentation

Full upstream docs at **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**.

| Section | What's Covered |
|---------|---------------|
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | Install, setup, first conversation |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | Commands, keybindings, sessions |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | Providers, models, all options |
| [Messaging](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp |
| [Tools](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ tools, toolset system |
| [Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Procedural memory, Skills Hub |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | Project structure, agent loop |

---

## Contributing

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
python3 -m pytest tests/ -q
```

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com). Fork maintained by Andy.
