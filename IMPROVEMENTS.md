# Hermes Agent Improvements

Running list of improvements made to the cloned Nous Research hermes-agent repo.
These are candidates for upstream contribution.

---

## CLI Terminal Interface (`cli.py`)

### Dynamic Terminal Resize Handling
- Replaced hardcoded `'─' * 200` horizontal rules with dynamic-width callables using `shutil.get_terminal_size()`
- All box borders (clarify, sudo, approval, response) now compute width at render time
- Input height calculation uses fresh terminal dimensions instead of cached `console.width`
- **Before**: Resizing terminal garbled the entire display with overflowing box-drawing chars
- **After**: Rules and boxes redraw correctly at any terminal width

### Think Block Styling
- Added `_format_think_blocks()` — renders `<think>...</think>` tags as dim italic gray ANSI text (`\033[2;3;37m`) with a `~ thinking ~` header
- Applied to all response output so local models with chain-of-thought look clean

### Image Paste Support (Ctrl+I)
- Detects macOS clipboard image data via `osascript` (checks for `«class PNGf»` / `«class TIFF»`)
- Saves clipboard image to `~/.hermes/images/clip_YYYYMMDD_HHMMSS.png`
- Inserts `[image: /path/to/file.png]` reference tag into input buffer
- On submit, extracts image references and converts to OpenAI vision multimodal content format (base64 data URIs)
- Enables Qwen3.5-9B to receive images from clipboard

---

## Local Model Server (`local_models/`)

### Qwen3.5-9B Server (`local_models/serve.py`)
- MLX-VLM backend with monkey-patched model alias rewriting and Starlette `/v1` route mounting
- Exposed via OpenAI-compatible `/v1/chat/completions` endpoint on port 8800

### Auto-Start Local Server (`hermes_cli/runtime_provider.py`)
- Socket probe checks if model server is alive on expected port
- If dead, auto-spawns `python3 -m local_models.serve qwen` in background
- Polls up to 60s for health check (model loading takes time)
- Shows stderr tail on crash for debugging
- User never has to manually start the server — `hermes --provider local --model local/qwen3.5-9b` just works

### Model Metadata (`agent/model_metadata.py`)
- Added context length entry for Qwen3.5-9B: 32768

---

## Runtime Provider Resolution (`hermes_cli/runtime_provider.py`)

### Local Provider Support
- `--provider local` resolves model → port mapping
- Reads model config from `~/.hermes/config.yaml`
- Falls back to `http://127.0.0.1:8800/v1` when no explicit base_url
- Auto-start integration (see above)

---

## Rust Acceleration (`hermes_rs/`)

### PyO3 Native Module
- `scan_context_content()` — regex-based prompt injection detection, **17x faster** than Python
- Uses compiled `RegexSet` (all 10 threat patterns matched in a single pass)
- Falls back to pure Python if Rust module not installed
- Build: `cd hermes_rs && maturin develop --release`

---

## Dependencies Added
- `mlx-vlm` — Apple Silicon VLM inference (Qwen3.5-9B)
- `torch`, `torchvision` — Required by Qwen3.5 VLM processor
- `hermes_rs` (optional) — Rust-accelerated prompt scanning via PyO3
