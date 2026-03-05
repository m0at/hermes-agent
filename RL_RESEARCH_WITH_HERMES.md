# Reinforcement Learning Research with Hermes Agent

## Why RL for AI Agents Matters

The dominant paradigm for building capable AI agents today is prompting: you write a system prompt, give the model tools, and hope it figures out the right sequence of actions. This works surprisingly well for simple tasks, but breaks down on complex, multi-step workflows where the agent needs to plan ahead, recover from mistakes, and make non-obvious tool choices.

Reinforcement learning changes this fundamentally. Instead of relying on the model's pre-trained intuitions about tool use, you let it practice on real tasks and learn from outcomes. An RL-trained agent doesn't just know *what* tools exist — it learns *when* to use them, *how* to chain them effectively, and *which strategies* lead to successful task completion.

Hermes Agent is built from the ground up to support this workflow: generate trajectories at scale, compress them for efficient training, and run full GRPO training loops with reward signals from real task execution.

---

## What Hermes Provides for RL Research

### The Full Pipeline

```
Dataset (prompts)
    |
    v
Batch Runner (parallel trajectory generation)
    |
    v
Raw Trajectories (ShareGPT format with tool calls)
    |
    v
Trajectory Compressor (fit within token budgets)
    |
    v
Atropos + Tinker (GRPO training with LoRA)
    |
    v
Fine-tuned Model (better tool use, planning, recovery)
```

Each stage is a standalone component. You can use just the batch runner for SFT data generation, or run the full loop for online RL training.

### Trajectory Generation

The batch runner (`batch_runner.py`) runs the agent across hundreds or thousands of prompts in parallel, producing training-ready trajectories. Each trajectory captures:

- The full conversation including system prompt with tool definitions
- Reasoning traces in `<think>` blocks
- Tool calls and their results in structured XML
- Tool usage statistics (which tools were called, success/failure rates)
- Metadata for filtering and analysis

A key feature is **toolset distributions** — you can specify probability distributions over different tool combinations, so the model encounters diverse scenarios during training. For example:

```yaml
toolset_distribution:
  - toolsets: [terminal, file, web]
    weight: 0.5
  - toolsets: [terminal, file]
    weight: 0.3
  - toolsets: [terminal, browser, file]
    weight: 0.2
```

This prevents the model from overfitting to a single tool configuration and encourages generalization.

### Trajectory Compression

Real agent trajectories can be extremely long — a 30-turn conversation with terminal output and file contents easily exceeds 30K tokens. The trajectory compressor intelligently shrinks these while preserving training signal:

1. **Protected regions**: The first few turns (problem setup) and last few turns (solution) are never touched
2. **Middle compression**: Intermediate tool calls and outputs are summarized by a fast model (Gemini Flash via OpenRouter)
3. **Token budget**: You set a target max length, and the compressor removes just enough middle content to fit

This is critical for practical training. Without compression, you either need enormous context windows (expensive, slow) or you lose the most important parts of the trajectory.

### Training with GRPO

Hermes uses **Group Relative Policy Optimization (GRPO)** for RL training, implemented through the Atropos/Tinker framework:

- **Atropos**: Coordinates rollouts, manages groups, computes advantages
- **Tinker**: Handles model weights, LoRA training, inference serving
- **Environments**: Define tasks, scoring, and reward functions

The training loop:
1. Environment provides a task (e.g., "fix this bug in the codebase")
2. Model generates a rollout — a full multi-turn agent conversation
3. Reward function scores the outcome (e.g., do the tests pass?)
4. GRPO computes advantages relative to other rollouts in the same group
5. Model weights are updated via importance sampling loss on LoRA adapters

---

## Research Directions Worth Pursuing

### 1. Tool Selection and Chaining

**The problem**: Given 38+ tools, models often use suboptimal tool sequences. They might use `web_search` when the answer is in a local file, or make five separate `read_file` calls when `search_files` would find what they need in one step.

**Why RL helps**: Reward functions can penalize unnecessary tool calls (efficiency), reward successful task completion, and bonus for using fewer steps. Over training, the model learns which tools to reach for first and how to chain them effectively.

**Concrete experiment**: Use HumanEvalPack or SWE-bench tasks. Compare a base model's tool usage patterns before and after GRPO training. Measure: task completion rate, average tool calls per task, time to completion.

### 2. Error Recovery and Robustness

**The problem**: When a tool call fails (command returns non-zero exit code, file not found, API error), base models often either give up or repeat the exact same failing action.

**Why RL helps**: By training on trajectories where the model encounters failures and must recover, the model learns recovery strategies: reading error messages carefully, trying alternative approaches, debugging incrementally.

**Concrete experiment**: Deliberately introduce failure modes in the environment (flaky network, missing dependencies, permission errors). Train with GRPO where only successful recovery gets reward. Measure recovery rate before/after training.

### 3. Long-Horizon Planning

**The problem**: Complex tasks require the model to maintain a plan across 20-30 turns. Base models lose track of their original goal, get sidetracked by intermediate results, or forget constraints stated earlier.

**Why RL helps**: The reward signal comes at the end of the full trajectory, forcing the model to learn to maintain coherence across the entire conversation. Trajectory compression helps by keeping trajectories within trainable lengths.

**Concrete experiment**: Design tasks that require 15+ coordinated tool calls (e.g., "set up a complete CI pipeline for this repo"). Reward only if the final state passes all validation checks. Compare planning coherence before/after GRPO.

### 4. Mixture of Specialists via Toolset Distributions

**The problem**: A model trained with all tools available learns a "jack of all trades" policy. But in deployment, you might want it to excel at specific toolsets (terminal-heavy DevOps tasks vs. browser-heavy research tasks).

**Why RL helps**: Toolset distributions let you control the training mix. You can train specialist models by weighting certain toolset combinations, or train a generalist that performs well across all combinations.

**Concrete experiment**: Train three models — one with 80% terminal-heavy tasks, one with 80% browser-heavy tasks, one with uniform distribution. Evaluate each on both task types. Does specialization improve performance without catastrophic forgetting?

### 5. Self-Improving Skill Creation

**The problem**: Hermes has a skills system where the agent can create and save procedural knowledge. But base models create low-quality skills or fail to reuse existing ones effectively.

**Why RL helps**: You can reward the model for creating skills that are actually reused (and succeed) in future tasks. This creates a self-improving loop: better skills → faster task completion → higher reward → better skill creation.

**Concrete experiment**: Multi-episode training where the model faces repeated similar tasks. Track whether it creates reusable skills and whether skill quality improves over episodes. Compare with/without skill creation tools enabled.

### 6. Reward Function Design

**The problem**: Most RL for agents uses binary rewards (task succeeded or not). This creates sparse signal that's hard to learn from, especially for long trajectories.

**Why RL helps** (meta-research): Hermes's ToolContext gives reward functions access to the full execution environment — you can run verification commands, check file contents, inspect process state. This enables rich, dense reward signals.

**Concrete experiments**:
- **Partial credit**: Reward proportional to how many test cases pass (not just all-or-nothing)
- **Progress rewards**: Reward for completing sub-goals (file created, dependency installed, test framework set up)
- **Style rewards**: Penalize destructive operations (rm -rf), reward idempotent approaches
- **Efficiency bonuses**: Extra reward for completing tasks in fewer turns

Compare convergence speed and final performance across reward designs.

### 7. Cross-Environment Transfer

**The problem**: Models trained in one execution environment (e.g., Docker with Ubuntu) may not transfer to others (e.g., local macOS, SSH to a server).

**Why RL helps**: Hermes supports 5 terminal backends. You can train across multiple backends and test transfer.

**Concrete experiment**: Train on Docker (Ubuntu), evaluate on local macOS and SSH. Does the model learn environment-agnostic strategies? What about training on mixed backends vs. single backend?

---

## How to Get Started

### Generating SFT Data (No GPU Required)

The simplest entry point — generate high-quality trajectories using a frontier model, then fine-tune a smaller model on them:

```bash
# Create a dataset of prompts
cat > prompts.jsonl << 'EOF'
{"prompt": "Create a Python script that scrapes the top HN stories and saves them as JSON"}
{"prompt": "Find and fix the bug in /workspace/app.py that causes the tests to fail"}
{"prompt": "Set up a basic Flask API with SQLite storage and write tests for it"}
EOF

# Generate trajectories
python3 batch_runner.py \
  --dataset prompts.jsonl \
  --run-name my-sft-run \
  --workers 4

# Compress trajectories to fit training context
python3 trajectory_compressor.py \
  --input data/my-sft-run/trajectories.jsonl \
  --output data/my-sft-run/compressed.jsonl \
  --max-tokens 15000
```

The output is ShareGPT-format JSONL ready for fine-tuning with any standard SFT framework (axolotl, torchtune, etc.).

### Running Full RL Training

Requires a GPU for training. The agent itself can set this up:

```
hermes> Use the RL training tools to discover available environments,
        select the SWE environment, and start a training run.
```

Or programmatically via the RL tools:
1. `rl_list_environments()` — discover what's available
2. `rl_select_environment("hermes_swe_env")` — load config
3. `rl_edit_config("total_steps", 500)` — adjust parameters
4. `rl_start_training()` — launches Atropos + Tinker + Environment
5. `rl_check_status(run_id)` — monitor via WandB

### Creating Custom Environments

Subclass `HermesAgentBaseEnv` and implement four methods:

```python
class MyEnv(HermesAgentBaseEnv):
    async def setup(self):
        # Load your dataset
        self.dataset = load_dataset("my-org/my-tasks")

    async def get_next_item(self):
        # Return next task for the model to attempt
        item = self.dataset[self.current_idx]
        return {"prompt": item["instruction"], "test": item["test_code"]}

    def format_prompt(self, item):
        return item["prompt"]

    async def compute_reward(self, item, result, ctx: ToolContext):
        # Use ctx to verify the model's work
        output = ctx.terminal(f'python3 -c "{item["test"]}"')
        return 1.0 if output["exit_code"] == 0 else 0.0
```

The `ToolContext` in `compute_reward` gives you access to the same terminal session the model used — you can inspect files, run tests, check state, anything the agent could do.

---

## Key Technical Details

### Trajectory Format

Hermes uses ShareGPT format with tool-call extensions:

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a helpful assistant.\n\n## Tools\n<tools>\n[{\"type\": \"function\", ...}]\n</tools>"
    },
    {
      "from": "human",
      "value": "Fix the failing tests in the repo"
    },
    {
      "from": "gpt",
      "value": "<think>Let me start by understanding what tests exist and why they fail.</think>\n<tool_call>\n{\"name\": \"terminal\", \"arguments\": {\"command\": \"cd /workspace && python3 -m pytest --tb=short 2>&1 | head -50\"}}\n</tool_call>"
    },
    {
      "from": "tool",
      "value": "<tool_response>\nFAILED tests/test_auth.py::test_login - AssertionError: expected 200, got 401\n</tool_response>"
    }
  ],
  "metadata": {
    "model": "anthropic/claude-opus-4.6",
    "toolsets": ["terminal", "file"],
    "session_id": "abc123"
  },
  "tool_stats": {
    "terminal": {"calls": 8, "successes": 6, "failures": 2},
    "read_file": {"calls": 3, "successes": 3, "failures": 0}
  }
}
```

### GRPO Training Details

- **Algorithm**: Group Relative Policy Optimization
- **Adapter**: LoRA (rank 32 default)
- **Base model**: Qwen3-8B (default, configurable)
- **Context length**: 8192 tokens (default)
- **Learning rate**: 4e-5
- **Group size**: 4 rollouts per task (advantages computed within group)
- **Rollout budget**: 10-30 turns per trajectory depending on environment

### Two-Phase Operation

Hermes supports two modes of trajectory generation within Atropos:

1. **Phase 1 (OpenAI servers)**: Uses standard `chat_completion()` with native tool parsing. Best for SFT data generation with frontier models.

2. **Phase 2 (VLLM ManagedServer)**: Uses `/generate` endpoint with client-side tool parsing. Required for actual RL training where gradients flow through the model. Supports Hermes, Mistral, Llama3, Qwen, DeepSeek, and other tool-call formats.

### Supported Tool-Call Parsers

For Phase 2 (training your own models), client-side parsers extract structured tool calls from raw model output:

- Hermes format (Nous models)
- Mistral format
- Llama 3 format
- Qwen format
- DeepSeek v3 / v3.1 format
- Kimi K2 format
- GLM format
- Longcat format

This means you can RL-train models in any of these tool-calling conventions.

---

## What Makes This Different

Most RL-for-agents research uses toy environments — text games, simple APIs with 3-4 tools, synthetic tasks. Hermes is different because:

1. **Real tools**: 38+ production tools including terminal execution, browser automation, file manipulation, web search. The model interacts with actual systems, not simulations.

2. **Real sandboxing**: Modal/Docker/Singularity backends provide genuine isolation. Reward functions can safely verify outcomes without contaminating the training environment.

3. **Scale-ready**: Batch runner handles parallelization, checkpointing, and fault tolerance. Trajectory compressor manages token budgets. The pipeline handles thousands of trajectories without manual intervention.

4. **Open ecosystem**: Everything is MIT-licensed. Trajectories are in standard ShareGPT format. Models train with standard LoRA. No proprietary lock-in at any stage.

5. **End-to-end**: From "I have a dataset of tasks" to "I have a fine-tuned model that's better at those tasks" in a single framework. No stitching together 5 different repos.

The gap between "agent that works in demos" and "agent that reliably works in production" is where RL training provides the most value. Hermes gives you the infrastructure to close that gap systematically.
