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

**A personal fork of [Nous Research's Hermes Agent](https://github.com/NousResearch/hermes-agent)** — adds local Qwen3.5-9B inference on Apple Silicon, browser-based WebGPU inference, clipboard image paste, Rust-accelerated prompt scanning, an evaluation testbed, and RL research tooling.

> **109 commits** ahead of upstream | **+4,900 lines** | **15+ new files** | Full upstream compatibility preserved

---

## What's Different from Upstream

This fork extends the upstream Hermes Agent with features focused on **local/private inference**, **advanced CLI interaction**, and **research tooling**. Everything below is unique to this fork — none of it exists in the upstream repo.

### 1. Local Qwen3.5-9B on Apple Silicon

Run Qwen3.5-9B locally via [MLX-VLM](https://github.com/ml-explore/mlx-vlm) — no cloud provider, no API keys, vision + text, 4-bit quantized, runs on unified GPU memory.

```bash
hermes --provider local --model local/qwen3.5-9b
```

- Server auto-starts on port 8800 when you select the local provider
- Polls up to 60s while the model loads, shows stderr on crash
- Serves OpenAI-compatible `/v1/chat/completions`
- Server process is tied to the hermes session — dies when hermes exits (atexit + SIGTERM cleanup)
- Manual start: `python3 -m local_models.serve qwen`

| | |
|---|---|
| **New files** | `local_models/serve.py`, `__init__.py`, `__main__.py` |
| **Modified** | `hermes_cli/runtime_provider.py`, `agent/model_metadata.py`, `pyproject.toml` |

### 2. WebGPU Client-Side Inference (Browser)

Run models entirely on the user's GPU via their browser — no server-side compute, no API keys, nothing leaves the machine. Uses [WebLLM](https://github.com/mlc-ai/web-llm) (MLC) to load quantized models through WebGPU.

```bash
hermes --provider webgpu
```

Auto-starts a bridge server on port 8801 and opens the browser UI. The user picks a model, WebLLM downloads and loads it on their GPU, then Hermes sends chat requests through the bridge.

**Architecture:**

```
Hermes CLI ──HTTP POST──> Bridge Server (port 8801) ──WebSocket──> Browser (WebGPU/WebLLM)
              /v1/chat/completions        <── inference results ──┘
```

The bridge (`web_client/bridge.py`) is a Starlette app that:
- Serves the web UI at `http://127.0.0.1:8801/`
- Accepts a WebSocket connection from the browser (the inference engine)
- Exposes an OpenAI-compatible `/v1/chat/completions` endpoint for Hermes
- Forwards requests to the browser, streams SSE responses back
- Tracks request/token/throughput stats at `/health`

**Available models** (all q4f16 quantized for browser VRAM):

| Model | VRAM | Context |
|-------|------|---------|
| Qwen3 4B | ~2.5 GB | 8K |
| Qwen2.5 3B Instruct | ~1.8 GB | 8K |
| Llama 3.1 8B Instruct | ~4.5 GB | 8K |
| Mistral 7B v0.3 | ~4 GB | 8K |
| SmolLM2 1.7B | ~1 GB | 4K |

**Requirements:** Chrome 113+ or Edge 113+ (WebGPU support). Works on any OS.

**Manual start:**

```bash
python3 -m web_client.bridge                  # start bridge + open browser
python3 -m web_client.bridge --no-browser     # headless (don't auto-open)
python3 -m web_client.bridge --port 9000      # custom port
```

| | |
|---|---|
| **New files** | `web_client/bridge.py`, `web_client/index.html`, `__init__.py`, `__main__.py` |
| **Modified** | `hermes_cli/runtime_provider.py`, `hermes_cli/auth.py`, `agent/model_metadata.py` |

### 3. Clipboard Image Paste (Cmd+V)

Paste screenshots and images directly into the chat — just like Claude Code. Press Cmd+V (macOS) or Ctrl+V (Linux) and the image appears as `[Image #N]` above the input prompt.

**How it works:**

1. Screenshot to clipboard (Cmd+Shift+Ctrl+4 on macOS)
2. Press Cmd+V in hermes — the `[Image #N] 12KB (up to select)` widget appears above the input
3. Type your question and press Enter — image is base64-encoded and sent as an OpenAI vision `image_url` content part
4. If you paste without typing, the default prompt is "What do you see in this image?"

**Extraction methods** (tried in order on macOS):
1. `pngpaste` — fastest if installed (`brew install pngpaste`)
2. PyObjC `NSPasteboard` in subprocess — reliable, no extra deps. Runs in a separate process to avoid prompt_toolkit's asyncio loop starving AppKit's CFRunLoop (which causes `NSPasteboard` to silently return nil in-process)
3. `osascript` fallback — AppleScript `clipboard info` + `«class PNGf»` extraction

**Linux:** Uses `xclip -selection clipboard -t image/png -o`

Large text pastes (>20 lines) are auto-collapsed to a temp file reference and expanded on submit.

| | |
|---|---|
| **Modified** | `cli.py` (keybindings, image widget, base64 encoding), `run_agent.py` (multipart content support) |

### 4. CLI Terminal Improvements

- **Dynamic terminal resize** — horizontal rules and box borders recompute width at render time via `shutil.get_terminal_size()` instead of hardcoded `'─' * 200`
- **Think block styling** — `<think>...</think>` tags from local CoT models render as dim italic gray with a `~ thinking ~` header
- **Color scheme picker** — choose between "cyber" (green/blue) and "synthwave" (pink/purple) on first launch, saved to config
- **Themed banner** — caduceus logo and welcome banner use the selected color scheme
- **Provider hot-swap** — `/provider openrouter|local|webgpu|custom` switches inference provider mid-session without restarting
- **`/copycode` command** — imports Claude Code skills from `~/.claude/commands/`, local `SKILL.md` files, and the [Anthropic skills repo](https://github.com/anthropics/skills) into hermes skill format

| | |
|---|---|
| **New files** | `hermes_cli/color_scheme.py` |
| **Modified** | `cli.py`, `agent/display.py`, `hermes_cli/banner.py` |

### 5. Rust-Accelerated Prompt Scanning

`hermes_rs` — a PyO3 native module that replaces the Python regex injection scanner with a compiled Rust `RegexSet`. **17x faster** on real context files. Falls back to pure Python if not installed.

**What it scans for** (10 threat patterns):
- Prompt injection / instruction override
- Deception hiding / ignore previous instructions
- System prompt extraction attempts
- HTML comment injection / hidden divs
- Translate-and-execute attacks
- Curl exfiltration / secret reading
- Invisible unicode characters (U+200B, U+200C, U+200D, U+2060, U+FEFF, U+202A-E)

Also includes `truncate_content()` — smart head/tail truncation (70% head, 20% tail) that preserves UTF-8 boundaries.

```bash
cd hermes_rs && maturin develop --release
```

| | |
|---|---|
| **New files** | `hermes_rs/` (Cargo.toml, src/lib.rs, prompt_scanner.rs, token_estimate.rs) |
| **Modified** | `agent/prompt_builder.py` |

### 6. Context Compression Improvements

- Fallback client chain when primary auxiliary model fails (reads `OPENAI_BASE_URL`)
- Provider-aware `max_tokens` vs `max_completion_tokens` handling
- Better token estimation for compression preflight checks

| | |
|---|---|
| **Modified** | `agent/context_compressor.py` |

### 7. Evaluation Testbed

Self-contained harness for programmatic agent interaction and eval.

```bash
python3 -m testbed.repl --query "list files in /tmp"
python3 -m testbed.eval_runner                        # run eval suite
```

- `testbed/repl.py` — interactive REPL or single-query runner (`--query`, `--toolsets`, `--model`, `--unsafe`, `--verbose`)
- `testbed/eval_runner.py` — runs eval tasks from `tasks.yaml`, scores results
- `testbed/harness.py` — thin wrapper around AIAgent for programmatic use
- `testbed/tasks.yaml` — 102 lines of eval task definitions across categories
- Default model: `google/gemini-2.0-flash` (cheap). File toolset only by default; `--unsafe` enables all tools.

| | |
|---|---|
| **New files** | `testbed/repl.py`, `eval_runner.py`, `harness.py`, `tasks.yaml`, `__init__.py` |

### 8. RL Research Pipeline

Complete pipeline for reinforcement learning research with tool-calling agents. See [RL_RESEARCH_WITH_HERMES.md](RL_RESEARCH_WITH_HERMES.md) for the full guide.

```
Dataset -> Batch Runner -> Raw Trajectories -> Compressor -> GRPO Training -> Fine-tuned Model
```

**Quick start (SFT data, no GPU):**

```bash
python3 batch_runner.py --dataset prompts.jsonl --run-name my-run --workers 4
python3 trajectory_compressor.py --input data/my-run/trajectories.jsonl \
  --output data/my-run/compressed.jsonl --max-tokens 15000
```

**Research directions covered:**
1. Tool selection and chaining
2. Error recovery strategies
3. Long-horizon planning
4. Mixture of specialists via toolset distributions
5. Self-improving skill creation
6. Rich reward function design
7. Cross-environment transfer

| | |
|---|---|
| **New/modified** | `batch_runner.py`, `trajectory_compressor.py`, `RL_RESEARCH_WITH_HERMES.md` |

---

## Providers

This fork supports 5 inference providers:

| Provider | Flag | How It Works |
|----------|------|-------------|
| **OpenRouter** | `--provider openrouter` | Cloud API via OpenRouter (default) |
| **Nous** | `--provider nous` | Nous Research portal with OAuth |
| **OpenAI Codex** | `--provider openai-codex` | OpenAI Codex Responses API |
| **Local (MLX)** | `--provider local` | Qwen3.5-9B on Apple Silicon GPU |
| **WebGPU** | `--provider webgpu` | Browser-side inference via WebLLM |

Switch mid-session with `/provider <name>` or set permanently:

```bash
hermes model    # interactive provider/model picker
```

**Environment variables:**

| Variable | Purpose |
|----------|---------|
| `HERMES_INFERENCE_PROVIDER` | Override provider selection |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `OPENAI_API_KEY` | OpenAI/custom API key |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible endpoint |
| `HERMES_WEBGPU_PORT` | Bridge server port (default: 8801) |
| `HERMES_NOUS_MIN_KEY_TTL_SECONDS` | Nous credential cache TTL (default: 1800) |

---

## Project Structure

```
hermes-agent/
├── agent/                  # Core agent modules
│   ├── auxiliary_client.py #   Shared LLM client for side tasks
│   ├── context_compressor.py # Auto context window compression
│   ├── display.py          #   Spinner, kawaii faces, think-block formatting
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
│   ├── color_scheme.py     #   Color theme picker (cyber/synthwave)
│   ├── banner.py           #   Themed welcome banner + caduceus
│   └── ...                 #   callbacks, cron, models, etc.
├── hermes_rs/              # Rust-accelerated modules (PyO3)
│   └── src/                #   prompt_scanner.rs, token_estimate.rs
├── tools/                  # 40+ agent tools
├── skills/                 # 22 skill categories
├── gateway/                # Telegram, Discord, Slack, WhatsApp
├── environments/           # RL training environments
├── local_models/           # Qwen3.5-9B model server (MLX-VLM)
│   └── serve.py            #   OpenAI-compatible /v1/chat/completions
├── web_client/             # WebGPU browser inference bridge
│   ├── bridge.py           #   Starlette bridge (HTTP ↔ WebSocket ↔ browser)
│   └── index.html          #   Browser UI (WebLLM model loader + inference)
├── testbed/                # Evaluation harness
│   ├── repl.py             #   Interactive REPL / single-query runner
│   ├── eval_runner.py      #   Task-based eval scoring
│   ├── harness.py          #   Programmatic AIAgent wrapper
│   └── tasks.yaml          #   Eval task definitions
├── cron/                   # Job scheduler
├── cli.py                  # Interactive REPL (prompt_toolkit TUI)
├── run_agent.py            # Main AIAgent orchestrator
├── batch_runner.py         # Trajectory generation for RL training
├── trajectory_compressor.py # Trajectory compression for training
├── mini-swe-agent/         # SWE-Agent submodule
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

After install, the local model deps are included by default:

```bash
hermes model        # select "Local models" → Qwen3.5-9B
hermes              # server auto-starts, model loads (~60s first time)
```

Or start the server manually:

```bash
python3 -m local_models.serve qwen
hermes --provider local --model local/qwen3.5-9b
```

### WebGPU Browser Inference (Any OS)

No extra deps needed — just a WebGPU-capable browser (Chrome 113+).

```bash
hermes --provider webgpu    # auto-starts bridge, opens browser
```

1. Browser opens → pick a model → click **Load Model** (downloads weights once, cached in browser)
2. Click **Connect to Hermes** → bridge links browser to CLI
3. Chat normally — inference runs on your GPU through the browser

### Rust Module (Optional)

For 17x faster prompt injection scanning:

```bash
cd hermes_rs
uv pip install maturin
maturin develop --release
```

### Image Paste (Optional)

For fastest clipboard image extraction, install pngpaste:

```bash
brew install pngpaste    # macOS only, optional — PyObjC fallback works without it
```

---

## Usage

```bash
hermes                              # Interactive CLI (default provider)
hermes --provider local             # Local MLX inference (Apple Silicon)
hermes --provider webgpu            # Browser WebGPU inference (any OS)
hermes model                        # Switch provider or model
hermes setup                        # Re-run setup wizard
hermes gateway                      # Start messaging gateway
hermes status                       # Show component status
hermes doctor                       # Diagnose issues
hermes update                       # Update to latest
hermes skills                       # Search/install skills
hermes cron                         # Manage scheduled jobs
hermes tools                        # Configure tool access per platform
```

**In-session commands:**

| Command | Description |
|---------|-------------|
| `/provider <name>` | Switch inference provider mid-session |
| `/copycode` | Import Claude Code skills into hermes |
| `/model <id>` | Switch model |
| `/tools` | List available tools |
| `/skills` | List installed skills |
| `/help` | Show all commands |

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| Cmd+V / Ctrl+V | Paste clipboard image |
| Enter | Send message |
| Alt+Enter / Ctrl+J | Insert newline (multiline input) |
| Ctrl+C | Interrupt agent |
| Ctrl+D | Exit |

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
git clone --recurse-submodules https://github.com/m0at/hermes-agent.git
cd hermes-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
uv pip install -e ./mini-swe-agent
python3 -m pytest tests/ -q
```

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com). Fork maintained by Andy.
