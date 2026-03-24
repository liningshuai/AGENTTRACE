from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import random
import time
from typing import Any

from .baselines import BaselineSuite
from .llm_baseline import LLMBaselineRunner
from .models import Scenario
from .ranker import AgentTraceRanker, GroupWeights, grid_search_weights, reciprocal_rank


def evaluate_methods(
    scenarios: list[Scenario],
    weights: GroupWeights | None = None,
    include_llm: bool = False,
    llm_runner: LLMBaselineRunner | None = None,
    llm_max_scenarios: int | None = None,
    max_depth: int = 10,
    seed: int = 12345,
) -> dict[str, dict[str, float]]:
    suite = BaselineSuite(seed=seed, max_depth=max_depth, weights=weights)
    results: dict[str, dict[str, float]] = {}
    for name, method in suite.methods(include_llm=include_llm, llm_runner=llm_runner).items():
        active_scenarios = scenarios
        if name == "llm_analysis" and llm_max_scenarios is not None and llm_max_scenarios < len(scenarios):
            rng = random.Random(seed)
            active_scenarios = rng.sample(scenarios, llm_max_scenarios)
        results[name] = _evaluate_method(active_scenarios, method)
    return results


def evaluate_agenttrace_breakdowns(
    scenarios: list[Scenario],
    weights: GroupWeights | None = None,
    max_depth: int = 10,
) -> dict[str, dict[str, dict[str, float]]]:
    ranker = AgentTraceRanker(weights=weights, max_depth=max_depth)
    return {
        "by_domain": _group_and_score(scenarios, ranker.rank, key=lambda scenario: scenario.domain),
        "by_domain_group": _group_and_score(scenarios, ranker.rank, key=lambda scenario: scenario.domain_group),
        "by_bug_type": _group_and_score(scenarios, ranker.rank, key=lambda scenario: scenario.ground_truth.bug_type),
        "by_bug_position": _group_and_score(scenarios, ranker.rank, key=lambda scenario: scenario.ground_truth.bug_position_bucket),
        "by_trace_length": _group_and_score(scenarios, ranker.rank, key=_trace_length_bucket),
    }


def build_agenttrace_trace_comparison(
    scenarios: list[Scenario],
    weights: GroupWeights | None = None,
    max_depth: int = 10,
) -> dict[str, Any]:
    ranker = AgentTraceRanker(weights=weights, max_depth=max_depth)
    traces: list[dict[str, Any]] = []

    for scenario in scenarios:
        result = ranker.rank(scenario)
        ranking = result.ordered_nodes
        truth = scenario.ground_truth.root_cause_node_id
        error_node = scenario.ground_truth.error_node_id
        truth_rank = ranking.index(truth) + 1 if truth in ranking else None
        predicted_top1 = ranking[0] if ranking else None
        top3 = ranking[:3]
        top5 = ranking[:5]

        traces.append(
            {
                "scenario_id": scenario.scenario_id,
                "domain": scenario.domain,
                "domain_group": scenario.domain_group,
                "interaction_pattern": scenario.interaction_pattern,
                "ground_truth": {
                    "root_cause_node_id": truth,
                    "error_node_id": error_node,
                    "bug_type": scenario.ground_truth.bug_type,
                    "bug_description": scenario.ground_truth.bug_description,
                    "bug_position_bucket": scenario.ground_truth.bug_position_bucket,
                },
                "predicted": {
                    "top1_node_id": predicted_top1,
                    "top3_node_ids": top3,
                    "top5_node_ids": top5,
                    "top1_match_root": predicted_top1 == truth,
                    "top3_match_root": truth in top3,
                    "top5_match_root": truth in top5,
                    "recall@1": float(predicted_top1 == truth),
                    "recall@3": float(truth in top3),
                    "recall@5": float(truth in top5),
                    "ground_truth_rank": truth_rank,
                },
                "candidate_root_cause_node_ids": result.candidate_nodes,
                "ranked_nodes": [
                    {
                        "rank": index,
                        "node_id": node_id,
                        "score": result.scores[node_id],
                        "group_scores": result.group_scores[node_id],
                        "is_candidate_ancestor": node_id in result.candidate_nodes,
                        "is_ground_truth_root_cause": node_id == truth,
                        "is_error_node": node_id == error_node,
                        "step": _serialize_scenario_step(scenario, node_id),
                    }
                    for index, node_id in enumerate(ranking, start=1)
                ],
            }
        )

    return {
        "dataset": "synthetic_agenttrace_reproduction",
        "method": "agenttrace",
        "trace_count": len(traces),
        "weights": ranker.weights.as_dict(),
        "max_depth": max_depth,
        "traces": traces,
    }


def learn_weights(
    validation_scenarios: list[Scenario],
    max_depth: int = 10,
    profile: str = "paper",
) -> dict[str, object]:
    weights, metrics = grid_search_weights(validation_scenarios, max_depth=max_depth, profile=profile)
    return {"weights": weights.as_dict(), "validation_metrics": metrics, "weight_profile": profile}


def _evaluate_method(scenarios: list[Scenario], method: Callable[[Scenario], list[int]]) -> dict[str, float]:
    hit1 = 0
    hit3 = 0
    hit5 = 0
    mrr_sum = 0.0
    total_runtime_ms = 0.0
    total = max(1, len(scenarios))

    for scenario in scenarios:
        started = time.perf_counter()
        ranking = method(scenario)
        total_runtime_ms += (time.perf_counter() - started) * 1000.0
        truth = scenario.ground_truth.root_cause_node_id
        if ranking and ranking[0] == truth:
            hit1 += 1
        if truth in ranking[:3]:
            hit3 += 1
        if truth in ranking[:5]:
            hit5 += 1
        mrr_sum += reciprocal_rank(ranking, truth)

    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "hit@5": hit5 / total,
        "mrr": mrr_sum / total,
        "mean_runtime_ms": total_runtime_ms / total,
        "evaluated_scenarios": total,
    }


def _group_and_score(
    scenarios: list[Scenario],
    rank_method: Callable[[Scenario], object],
    key: Callable[[Scenario], str],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[tuple[list[int], int]]] = defaultdict(list)
    for scenario in scenarios:
        ranking = rank_method(scenario).ordered_nodes
        truth = scenario.ground_truth.root_cause_node_id
        buckets[key(scenario)].append((ranking, truth))

    results: dict[str, dict[str, float]] = {}
    for bucket, items in buckets.items():
        count = len(items)
        hit1 = sum(1 for ranking, truth in items if ranking and ranking[0] == truth)
        hit3 = sum(1 for ranking, truth in items if truth in ranking[:3])
        hit5 = sum(1 for ranking, truth in items if truth in ranking[:5])
        mrr_sum = sum(reciprocal_rank(ranking, truth) for ranking, truth in items)
        results[bucket] = {
            "count": count,
            "hit@1": hit1 / max(1, count),
            "hit@3": hit3 / max(1, count),
            "hit@5": hit5 / max(1, count),
            "mrr": mrr_sum / max(1, count),
        }
    return dict(sorted(results.items()))


def _trace_length_bucket(scenario: Scenario) -> str:
    length = len(scenario.steps)
    if length <= 9:
        return "8-9"
    if length <= 11:
        return "10-11"
    if length <= 13:
        return "12-13"
    return "14-15"


def _serialize_scenario_step(scenario: Scenario, step_id: int) -> dict[str, Any]:
    step = scenario.steps[step_id - 1]
    return {
        "step_id": step.step_id,
        "agent": step.agent,
        "action_type": step.action_type,
        "timestamp": step.timestamp,
        "confidence": step.confidence,
        "tags": step.tags,
        "input_excerpt": step.input[:280],
        "output_excerpt": step.output[:280],
        "metadata": step.metadata,
    }
