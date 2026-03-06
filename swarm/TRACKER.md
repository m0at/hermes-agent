# Swarm Implementation Tracker

Master board for all swarm work. Each section tracks status.

## P1 Bug Fixes (Prerequisites)

| # | Task | Status | Notes |
|---|------|--------|-------|
| B1 | Lazy import firecrawl in web_tools.py | DONE | Moved import into _get_firecrawl_client() |
| B2 | SQLite session double-write dedup | DONE | Added _session_db_flushed counter |
| B3 | Background job cleanup safety | DONE | cleanup_vm checks active processes before teardown |
| B4 | Nudge text in training trajectories | DONE | Use original_user_message in _save_trajectory |
| B5 | Delegate tool stdout/stderr isolation | DONE | Thread-local stream wrapper, no more global redirect |

## Swarm Core Infrastructure

| # | Task | Status | File | Notes |
|---|------|--------|------|-------|
| S1 | Core types + exceptions | DONE | types.py, exceptions.py | TaskState, WorkerState, SwarmTask, SwarmWorker, SwarmConfig, SwarmResult, ArtifactRef |
| S2 | DAG task scheduler | DONE | scheduler.py | Thread-safe, deps, retries, cancel_downstream, checkpoint |
| S3 | Worker pool manager | DONE | worker.py | Local/Modal/SSH backends, scale up/down, assign/release |
| S4 | Artifact bus | DONE | artifacts.py | SHA-256 provenance, store/get/export manifest |
| S5 | Model router | DONE | router.py | Budget-aware, role-based, cost tracking |
| S9 | Agent roles | DONE | roles.py | Planner, Executor, Critic, Verifier, Merger, Researcher |

## Swarm — Next Wave

| # | Task | Status | Notes |
|---|------|--------|-------|
| S6 | Agent-to-agent messaging bus | PENDING | |
| S7 | Git worktree orchestration | PENDING | Branch-per-agent |
| S8 | Telemetry + dashboard | PENDING | Cost, tokens, failures |
| S10 | Semantic conflict resolver | PENDING | Multi-agent file edits |
| I1 | CLI /swarm command | PENDING | Launch, status, cancel |
| I2 | Modal worker backend (real) | PENDING | First cloud backend |
| I3 | Human approval gates | PENDING | Risk-tier based |
| S11 | Swarm orchestrator (main loop) | PENDING | Ties scheduler + workers + router together |
| S12 | Task decomposition (planner) | PENDING | Auto-split plans into parallel tasks |
| S13 | Verifier agents | PENDING | Score patches/results before merge |
