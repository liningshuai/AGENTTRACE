from __future__ import annotations

import random

from .llm_baseline import LLMBaselineRunner, llm_baseline_placeholder
from .models import Scenario
from .ranker import AgentTraceRanker, GroupWeights


class BaselineSuite:
    def __init__(self, seed: int = 12345, max_depth: int = 10, weights: GroupWeights | None = None) -> None:
        self._rng = random.Random(seed)
        self._agenttrace = AgentTraceRanker(weights=weights, max_depth=max_depth)

    def methods(
        self,
        include_llm: bool = False,
        llm_runner: LLMBaselineRunner | None = None,
    ) -> dict[str, callable]:
        methods: dict[str, callable] = {
            "random": self.random_baseline,
            "first_node": self.first_node_baseline,
            "last_node": self.last_node_baseline,
            "agenttrace": self.agenttrace_baseline,
            "position_only": self.position_only_baseline,
        }
        if include_llm:
            methods["llm_analysis"] = llm_runner.rank if llm_runner is not None else llm_baseline_placeholder
        return methods

    def random_baseline(self, scenario: Scenario) -> list[int]:
        nodes = [step.step_id for step in scenario.steps]
        self._rng.shuffle(nodes)
        return nodes

    def first_node_baseline(self, scenario: Scenario) -> list[int]:
        return [step.step_id for step in scenario.steps]

    def last_node_baseline(self, scenario: Scenario) -> list[int]:
        error_node = scenario.ground_truth.error_node_id
        ordered = [max(1, error_node - 1)]
        ordered.extend(node_id for node_id in range(len(scenario.steps), 0, -1) if node_id not in ordered)
        return ordered

    def agenttrace_baseline(self, scenario: Scenario) -> list[int]:
        return self._agenttrace.rank(scenario).ordered_nodes

    def position_only_baseline(self, scenario: Scenario) -> list[int]:
        position_ranker = AgentTraceRanker(
            weights=GroupWeights(position=1.0, structure=0.0, content=0.0, flow=0.0, confidence=0.0),
            max_depth=self._agenttrace.max_depth,
        )
        return position_ranker.rank(scenario).ordered_nodes
