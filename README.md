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

**A personal fork of [Nous Research's Hermes Agent](https://github.com/NousResearch/hermes-agent)** — local Qwen3.5-9B vision inference on Apple Silicon, browser-based WebGPU inference, clipboard image paste for VLMs, Rust-accelerated prompt scanning, evaluation testbed, and RL research tooling.

> **109 commits** ahead of upstream | **+4,900 lines** | **15+ new files** | Full upstream compatibility preserved

---

## Fork vs Upstream — What's New

Everything below is **unique to this fork**. The upstream repo has none of it.

| Feature | Upstream | This Fork |
|---------|----------|-----------|
| Local inference | Cloud-only (OpenRouter, Nous) | Qwen3.5-9B on Apple Silicon GPU via MLX-VLM |
| Browser inference | No | WebGPU client-side via WebLLM bridge |
| Image input | No | Cmd+V clipboard paste with VLM vision support |
| Prompt security | Python regex | Rust `RegexSet` (17x faster, PyO3) |
| Think blocks | Raw `<think>` tags shown | Styled dim italic with `~ thinking ~` header |
| Terminal rendering | Hardcoded 200-char rules | Dynamic `shutil.get_terminal_size()` |
| Color themes | Fixed colors | Cyber (green/blue) or Synthwave (pink/purple) |
| Provider switching | Restart required | `/provider` hot-swap mid-session |
| Skill import | Manual | `/copycode` imports from Claude Code + Anthropic repo |
| Server lifecycle | Orphaned processes | Server tied to hermes session (atexit + SIGTERM) |
| Evaluation | No | Testbed with REPL, eval runner, task definitions |
| RL training | No | Full pipeline: batch runner → trajectory compressor → GRPO |
| Context compression | Basic | Fallback client chain, provider-aware token handling |

---

## 1. Local Qwen3.5-9B on Apple Silicon

Run Qwen3.5-9B locally via [MLX-VLM](https://github.com/ml-explore/mlx-vlm) — no cloud, no API keys. Vision + text, 4-bit quantized, unified GPU memory.

```bash
hermes --provider local --model local/qwen3.5-9b
```

- Auto-starts server on port 8800, polls up to 60s for model load
- Serves OpenAI-compatible `/v1/chat/completions` with full vision support
- Server dies when hermes exits — atexit hook + SIGTERM handler, no orphan processes
- Manual start: `python3 -m local_models.serve qwen`

**New:** `local_models/serve.py` | **Modified:** `hermes_cli/runtime_provider.py`, `agent/model_metadata.py`

---

## 2. WebGPU Client-Side Inference (Browser)

Run models on the user's GPU through their browser — zero server compute, nothing leaves the machine. Uses [WebLLM](https://github.com/mlc-ai/web-llm) (MLC) for quantized model loading via WebGPU.

```bash
hermes --provider webgpu
```

```
Hermes CLI ──HTTP POST──> Bridge (port 8801) ──WebSocket──> Browser (WebGPU/WebLLM)
              /v1/chat/completions       <── inference results ──┘
```

The bridge (`web_client/bridge.py`) is a Starlette app: serves the web UI, accepts a WebSocket from the browser, exposes OpenAI-compatible `/v1/chat/completions` for Hermes, streams SSE responses back.

| Model | VRAM | Context |
|-------|------|---------|
| Qwen3 4B | ~2.5 GB | 8K |
| Qwen2.5 3B Instruct | ~1.8 GB | 8K |
| Llama 3.1 8B Instruct | ~4.5 GB | 8K |
| Mistral 7B v0.3 | ~4 GB | 8K |
| SmolLM2 1.7B | ~1 GB | 4K |

Requires Chrome 113+ or Edge 113+. Works on macOS, Windows, Linux.

**New:** `web_client/bridge.py`, `web_client/index.html` | **Modified:** `hermes_cli/runtime_provider.py`, `hermes_cli/auth.py`

---

## 3. Clipboard Image Paste for VLMs

Paste screenshots directly into the chat. Cmd+V on macOS, Ctrl+V on Linux. The image shows as `[Image #N]` above the input, gets base64-encoded and sent as an OpenAI `image_url` content part. Works with any vision model — local Qwen3.5-9B, OpenRouter multimodal models, etc.

```
● what word is this?
  📎 attached clip_20260305_171559_1.png (7KB)
```

### Why this was hard

Terminal apps can't natively receive image data from the clipboard. When you press Cmd+V:

1. **The terminal emulator intercepts it**, not your app. It reads the clipboard, extracts any **text**, and sends it to stdin wrapped in bracketed paste escape sequences (`\e[200~...text...\e[201~`).
2. **Terminals only paste text.** If the clipboard has an image with no text, the terminal sends an empty bracketed paste — your app gets `data=''` and has no idea an image exists.
3. **You can't just call the clipboard API.** PyObjC's `NSPasteboard` requires a running CFRunLoop for XPC communication with the macOS pasteboard server. But prompt_toolkit's asyncio event loop uses kqueue, not CFRunLoop. So `NSPasteboard.generalPasteboard().dataForType_()` **silently returns nil** when called in-process — no error, no exception, just nothing. This is the bug that made us think the clipboard was empty when it wasn't.

### How we solved it

The clipboard extraction runs in a **separate Python subprocess**. A fresh process gets its own AppKit runtime with a working CFRunLoop, so `NSPasteboard` works correctly. The subprocess writes the image to disk, the main process picks it up.

**Extraction chain** (tried in order on macOS):
1. `pngpaste` — native Obj-C binary, fastest (`brew install pngpaste`)
2. PyObjC `NSPasteboard` **in subprocess** — reliable, no extra deps
3. `osascript` fallback — AppleScript `clipboard info` + `«class PNGf»`

**Linux:** `xclip -selection clipboard -t image/png -o`

The BracketedPaste handler in prompt_toolkit fires on Cmd+V (even with empty data), triggers the subprocess clipboard check, and attaches any found image. Images survive through the interrupt queue, multipart content assembly, and the full `run_conversation` pipeline to the API call.

**Modified:** `cli.py`, `run_agent.py` (multipart content support through the entire agent pipeline)

---

## 4. Rust-Accelerated Prompt Scanning

`hermes_rs` — PyO3 native module. Compiled Rust `RegexSet` replaces Python regex for injection detection. **17x faster** on real context files. Falls back to pure Python if not installed.

Scans for 10 threat patterns: prompt injection, instruction override, system prompt extraction, HTML comment injection, hidden divs, translate-and-execute, curl exfiltration, secret reading, invisible unicode (U+200B/C/D, U+2060, U+FEFF, U+202A-E).

Includes `truncate_content()` — smart 70% head / 20% tail truncation preserving UTF-8 boundaries.

```bash
cd hermes_rs && maturin develop --release
```

**New:** `hermes_rs/` (Cargo.toml, src/lib.rs, prompt_scanner.rs, token_estimate.rs) | **Modified:** `agent/prompt_builder.py`

---

## 5. CLI Improvements

- **Dynamic terminal width** — rules and boxes use `shutil.get_terminal_size()`, no more overflow in VS Code
- **Think block styling** — `<think>` tags render as dim italic gray with `~ thinking ~` header, toggleable with `/thinkon` `/thinkoff`
- **Color schemes** — "cyber" (green/blue) or "synthwave" (pink/purple), chosen on first launch
- **`/provider` hot-swap** — switch `openrouter|local|webgpu|custom` mid-session
- **`/copycode`** — imports skills from `~/.claude/commands/`, local `SKILL.md` files, and [anthropics/skills](https://github.com/anthropics/skills) repo
- **Large paste collapse** — pastes >20 lines saved to temp file, expanded on submit
- **Server lifecycle** — local model server is a child process, killed on exit

**New:** `hermes_cli/color_scheme.py` | **Modified:** `cli.py`, `agent/display.py`, `hermes_cli/banner.py`

---

## 6. Evaluation Testbed

```bash
python3 -m testbed.repl --query "list files in /tmp"     # single query
python3 -m testbed.eval_runner                             # run eval suite
```

- `testbed/repl.py` — REPL or single-query (`--query`, `--toolsets`, `--model`, `--unsafe`)
- `testbed/eval_runner.py` — runs tasks from `tasks.yaml`, scores results
- `testbed/harness.py` — programmatic AIAgent wrapper
- Default: `google/gemini-2.0-flash`, file toolset only. `--unsafe` enables all tools.

---

## 7. RL Research Pipeline

Full pipeline for reinforcement learning with tool-calling agents. See [RL_RESEARCH_WITH_HERMES.md](RL_RESEARCH_WITH_HERMES.md).

```
Dataset → Batch Runner → Raw Trajectories → Compressor → GRPO Training → Fine-tuned Model
```

```bash
python3 batch_runner.py --dataset prompts.jsonl --run-name my-run --workers 4
python3 trajectory_compressor.py --input data/my-run/trajectories.jsonl \
  --output data/my-run/compressed.jsonl --max-tokens 15000
```

---

## Providers

| Provider | Flag | How It Works |
|----------|------|-------------|
| **OpenRouter** | `--provider openrouter` | Cloud API (default) |
| **Nous** | `--provider nous` | Nous Research portal with OAuth |
| **OpenAI Codex** | `--provider openai-codex` | Codex Responses API |
| **Local (MLX)** | `--provider local` | Qwen3.5-9B on Apple Silicon GPU |
| **WebGPU** | `--provider webgpu` | Browser-side inference via WebLLM |

Switch mid-session: `/provider <name>` | Set permanently: `hermes model`

| Variable | Purpose |
|----------|---------|
| `HERMES_INFERENCE_PROVIDER` | Override provider |
| `OPENROUTER_API_KEY` | OpenRouter key |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | Custom endpoint |
| `HERMES_WEBGPU_PORT` | Bridge port (default: 8801) |

---

## Project Structure

```
hermes-agent/
├── agent/                  # Core agent modules
│   ├── context_compressor.py # Auto context window compression
│   ├── display.py          #   Spinner, think-block formatting
│   ├── model_metadata.py   #   Token estimation, context lengths
│   ├── prompt_builder.py   #   System prompt + injection detection
│   └── ...                 #   caching, redaction, skills, trajectory
├── hermes_cli/             # CLI application
│   ├── runtime_provider.py #   Provider resolution + auto-start
│   ├── color_scheme.py     #   Theme picker (cyber/synthwave)
│   └── ...                 #   auth, config, setup, gateway, etc.
├── hermes_rs/              # Rust prompt scanner (PyO3)
├── local_models/           # Qwen3.5-9B MLX-VLM server
├── web_client/             # WebGPU browser inference bridge
│   ├── bridge.py           #   Starlette HTTP ↔ WebSocket ↔ browser
│   └── index.html          #   WebLLM model loader UI
├── testbed/                # Evaluation harness
├── cli.py                  # Interactive REPL (prompt_toolkit)
├── run_agent.py            # AIAgent orchestrator
├── batch_runner.py         # RL trajectory generation
├── trajectory_compressor.py # Trajectory compression
├── mini-swe-agent/         # SWE-Agent submodule
└── tinker-atropos/         # RL training (Atropos/GRPO)
```

---

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc && hermes setup
```

**Local model:** `hermes model` → select Local → Qwen3.5-9B. Or: `hermes --provider local`

**WebGPU:** `hermes --provider webgpu` (needs Chrome 113+)

**Rust scanner:** `cd hermes_rs && maturin develop --release`

**Faster image paste:** `brew install pngpaste` (optional, PyObjC fallback works)

---

## Usage

```bash
hermes                              # Interactive CLI
hermes --provider local             # Local Qwen3.5-9B (Apple Silicon)
hermes --provider webgpu            # Browser WebGPU inference
hermes model                        # Switch provider/model
hermes setup / doctor / status      # Configuration & diagnostics
hermes gateway                      # Telegram, Discord, Slack, WhatsApp
hermes skills / tools / cron        # Skills, tools, scheduled jobs
```

| Shortcut | Action |
|----------|--------|
| Cmd+V / Ctrl+V | Paste clipboard image |
| Enter | Send message |
| Alt+Enter / Ctrl+J | Newline (multiline) |
| Ctrl+C | Interrupt agent |
| `/provider <name>` | Switch provider |
| `/copycode` | Import Claude Code skills |
| `/thinkon` `/thinkoff` | Toggle think blocks |

---

## Documentation

Full upstream docs: **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**

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
