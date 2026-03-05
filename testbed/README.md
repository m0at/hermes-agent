# Hermes Agent Testbed

A small, self-contained harness for interacting with the Hermes agent
programmatically. Useful for experimentation, evaluation, and development.

## Quick Start

```bash
# Interactive REPL (cheapest model, safe toolset)
python3 testbed/repl.py

# Single query
python3 testbed/repl.py --query "What is 2+2?"

# With specific toolsets
python3 testbed/repl.py --toolsets terminal,file

# Run evaluation suite
python3 testbed/eval_runner.py

# Run with a specific model
python3 testbed/repl.py --model google/gemini-2.0-flash

# Verbose mode (see tool calls)
python3 testbed/repl.py --verbose
```

## Components

| File | Purpose |
|------|---------|
| `repl.py` | Interactive REPL or single-query runner |
| `eval_runner.py` | Run eval tasks and score results |
| `tasks.yaml` | Eval task definitions with expected outcomes |
| `harness.py` | Thin wrapper around AIAgent for testbed use |

## Configuration

Set `OPENROUTER_API_KEY` in your environment or in `~/.hermes/.env`.

Default model is `google/gemini-2.0-flash` (fast and cheap for testing).
Override with `--model`.

## Safety

By default the testbed runs with `file` toolset only (read/write/search).
Pass `--toolsets terminal,file,web` to enable more tools.
The `--unsafe` flag enables all tools.

## Local Model (No API Key)

Run the testbed entirely offline using a local Mistral 7B model. A small
Python proxy (`local_proxy.py`) wraps the native binary in an
OpenAI-compatible `/v1/chat/completions` endpoint so the existing agent
stack works without code changes.

### One-command start

```bash
bash testbed/run_local.sh
```

This launches the proxy on port 8787, waits for it to become healthy, then
starts the REPL pointed at `http://localhost:8787/v1`.

### Manual start

Run the proxy and REPL in separate terminals:

```bash
# Terminal 1 – start the proxy
python3 testbed/local_proxy.py

# Terminal 2 – point the REPL at it
OPENAI_BASE_URL=http://localhost:8787/v1 \
OPENAI_API_KEY=local \
  python3 testbed/repl.py --model mistral
```

### Architecture

```
REPL/eval_runner → TestbedAgent → AIAgent → OpenAI client
                                                 ↓
                                           localhost:8787
                                                 ↓
                                           local_proxy.py
                                                 ↓
                                           mistral binary (Metal GPU)
                                                 ↓
                                           mistral-7b-instruct-v0.2.Q4_0.gguf
```

### Testing the proxy

```bash
python3 testbed/test_proxy.py
```

### Configuration

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `MISTRAL_BINARY` | `./mistral` | Path to the compiled Mistral inference binary |
| `MISTRAL_MODEL` | `mistral-7b-instruct-v0.2.Q4_0.gguf` | Path to the GGUF model file |

### Limitations

- **No tool calling** – the local model does not support function/tool-call
  schemas; tool-dependent eval tasks will fail.
- **No streaming** – responses are returned in a single block once
  generation completes.
- **Single-shot subprocess per request** – the proxy spawns the mistral
  binary for each completion request, so throughput is limited to one
  request at a time.
