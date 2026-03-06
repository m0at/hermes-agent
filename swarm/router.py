from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swarm.types import SwarmConfig
from swarm.exceptions import BudgetExceededError


# USD per 1M tokens (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus":      (15.00, 75.00),
    "claude-sonnet":    (3.00,  15.00),
    "claude-haiku":     (0.25,  1.25),
    "gpt-4o":           (2.50,  10.00),
    "gpt-4o-mini":      (0.15,  0.60),
    "gemini-flash":     (0.10,  0.40),
    "qwen3.5-9b":       (0.00,  0.00),   # local
}

# role -> ordered preference list of models
_DEFAULT_ROUTING: dict[str, list[str]] = {
    "planner":    ["claude-opus", "gpt-4o"],
    "critic":     ["claude-opus", "gpt-4o"],
    "executor":   ["gemini-flash", "gpt-4o-mini"],
    "verifier":   ["claude-sonnet"],
    "merger":     ["claude-sonnet", "claude-opus"],
    "researcher": ["gemini-flash", "gpt-4o-mini"],
    "local":      ["qwen3.5-9b"],
}

# task_type overrides (applied before role fallback)
_TASK_TYPE_ROUTING: dict[str, list[str]] = {
    "code_generation": ["gemini-flash", "gpt-4o-mini"],
    "code_review":     ["claude-opus", "gpt-4o"],
    "summarization":   ["gemini-flash", "claude-haiku"],
    "planning":        ["claude-opus", "gpt-4o"],
    "verification":    ["claude-sonnet"],
    "local":           ["qwen3.5-9b"],
}


@dataclass
class _UsageRecord:
    model: str
    role: str
    task_type: str
    input_tokens: int
    output_tokens: int
    cost: float


class ModelRouter:
    def __init__(self, config: SwarmConfig) -> None:
        self._config = config
        self._budget_limit = config.budget_limit_usd
        self._routing = dict(_DEFAULT_ROUTING)
        self._task_routing = dict(_TASK_TYPE_ROUTING)
        self._pricing = dict(_MODEL_PRICING)
        self._usage: list[_UsageRecord] = []
        self._total_spend: float = 0.0

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def select_model(
        self,
        role: str,
        task_type: str | None = None,
        budget_remaining: float | None = None,
    ) -> str:
        budget = budget_remaining if budget_remaining is not None else self._remaining_budget()

        # Build candidate list: task_type first, then role fallback
        candidates: list[str] = []
        if task_type and task_type in self._task_routing:
            candidates.extend(self._task_routing[task_type])
        if role in self._routing:
            for m in self._routing[role]:
                if m not in candidates:
                    candidates.append(m)
        if not candidates:
            candidates = ["gemini-flash"]

        # If budget is constrained, filter out models that are too expensive
        # (rough estimate: 2k input + 1k output as a probe)
        if budget is not None and budget > 0:
            affordable = [
                m for m in candidates
                if self.estimate_cost(m, 2000, 1000) <= budget
            ]
            if affordable:
                candidates = affordable

        return candidates[0]

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = self._pricing.get(model, (5.0, 15.0))
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def log_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        role: str = "",
        task_type: str = "",
    ) -> None:
        self._usage.append(_UsageRecord(
            model=model,
            role=role,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        ))
        self._total_spend += cost

    def get_spend_summary(self) -> dict[str, Any]:
        by_model: dict[str, float] = {}
        by_role: dict[str, float] = {}
        by_task_type: dict[str, float] = {}
        total_input = 0
        total_output = 0

        for r in self._usage:
            by_model[r.model] = by_model.get(r.model, 0.0) + r.cost
            if r.role:
                by_role[r.role] = by_role.get(r.role, 0.0) + r.cost
            if r.task_type:
                by_task_type[r.task_type] = by_task_type.get(r.task_type, 0.0) + r.cost
            total_input += r.input_tokens
            total_output += r.output_tokens

        return {
            "total_cost_usd": self._total_spend,
            "budget_limit_usd": self._budget_limit,
            "budget_remaining_usd": self._remaining_budget(),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "num_requests": len(self._usage),
            "by_model": by_model,
            "by_role": by_role,
            "by_task_type": by_task_type,
        }

    def is_within_budget(self, estimated_cost: float) -> bool:
        if self._budget_limit <= 0:
            return True  # no limit set
        return (self._total_spend + estimated_cost) <= self._budget_limit

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remaining_budget(self) -> float | None:
        if self._budget_limit <= 0:
            return None
        return max(0.0, self._budget_limit - self._total_spend)
