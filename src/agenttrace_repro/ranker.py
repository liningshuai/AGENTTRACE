from __future__ import annotations

from dataclasses import dataclass
import itertools
from statistics import mean, pstdev
import time

from .config import DEFAULT_GROUP_WEIGHTS, PAPER_WEIGHT_GRID, ROBUST_WEIGHT_GRID, WIDE_WEIGHT_GRID
from .graph import (
    Graph,
    approximate_betweenness,
    backward_trace,
    build_graph,
    descendants,
    edge_type_count,
    incoming_cross_agent,
    outgoing_cross_agent,
    reverse_distances_to_error,
)
from .models import Scenario


ERROR_TOKENS = {"error", "failed", "invalid", "contradiction", "mismatch", "unsafe", "omitted", "corrupted"}
HEDGE_TOKENS = {"maybe", "probably", "appears", "roughly", "not", "assume", "unclear", "plausible"}


@dataclass(frozen=True, slots=True)
class GroupWeights:
    position: float
    structure: float
    content: float
    flow: float
    confidence: float

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> GroupWeights:
        return cls(
            position=payload["position"],
            structure=payload["structure"],
            content=payload["content"],
            flow=payload["flow"],
            confidence=payload["confidence"],
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "position": self.position,
            "structure": self.structure,
            "content": self.content,
            "flow": self.flow,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class RankingResult:
    ordered_nodes: list[int]
    scores: dict[int, float]
    group_scores: dict[int, dict[str, float]]
    runtime_ms: float
    candidate_nodes: list[int]


def default_weights() -> GroupWeights:
    return GroupWeights.from_dict(DEFAULT_GROUP_WEIGHTS)


def weight_grid(profile: str = "paper") -> dict[str, tuple[float, ...]]:
    if profile == "paper":
        return PAPER_WEIGHT_GRID
    if profile == "wide":
        return WIDE_WEIGHT_GRID
    if profile == "robust":
        return ROBUST_WEIGHT_GRID
    raise ValueError(f"Unknown weight profile: {profile}")


class AgentTraceRanker:
    def __init__(self, weights: GroupWeights | None = None, max_depth: int = 10) -> None:
        self.weights = weights or default_weights()
        self.max_depth = max_depth

    def rank(self, scenario: Scenario) -> RankingResult:
        started = time.perf_counter()
        graph = build_graph(scenario)
        group_scores = compute_group_scores(scenario, graph)
        candidate_set = backward_trace(graph, scenario.ground_truth.error_node_id, self.max_depth)
        scores = {
            node_id: (
                self.weights.position * groups["position"]
                + self.weights.structure * groups["structure"]
                + self.weights.content * groups["content"]
                + self.weights.flow * groups["flow"]
                + self.weights.confidence * groups["confidence"]
            )
            for node_id, groups in group_scores.items()
        }
        ordered_candidates = sorted(candidate_set, key=lambda node_id: (scores[node_id], -node_id), reverse=True)
        remainder = sorted((node_id for node_id in graph.nodes if node_id not in candidate_set), key=lambda node_id: (scores[node_id], -node_id), reverse=True)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        return RankingResult(
            ordered_nodes=ordered_candidates + remainder,
            scores=scores,
            group_scores=group_scores,
            runtime_ms=runtime_ms,
            candidate_nodes=ordered_candidates,
        )


def compute_group_scores(scenario: Scenario, graph: Graph) -> dict[int, dict[str, float]]:
    node_ids = sorted(graph.nodes)
    total_nodes = len(node_ids)
    reverse_distance = reverse_distances_to_error(graph, scenario.ground_truth.error_node_id)
    reachability = {node_id: len(descendants(graph, node_id)) for node_id in node_ids}
    betweenness = approximate_betweenness(graph)
    position_index = {node_id: index for index, node_id in enumerate(node_ids)}
    role_weights = {agent: 1.0 - (index * 0.15) for index, agent in enumerate(scenario.agents)}

    token_lengths = {
        node_id: max(1, len((graph.nodes[node_id].input + " " + graph.nodes[node_id].output).split()))
        for node_id in node_ids
    }
    avg_length = mean(token_lengths.values())
    length_std = pstdev(token_lengths.values()) if len(token_lengths) > 1 else 0.0

    raw = {
        "position_earlyness": {},
        "position_distance": {},
        "position_forward_index": {},
        "structure_out_degree": {},
        "structure_reachability": {},
        "structure_betweenness": {},
        "structure_in_degree": {},
        "content_error_keywords": {},
        "content_uncertainty": {},
        "content_length_anomaly": {},
        "content_keyword_density": {},
        "content_novelty": {},
        "content_log_anomaly": {},
        "flow_cross_agent": {},
        "flow_role_criticality": {},
        "flow_communication": {},
        "flow_role_mismatch": {},
        "confidence_inverse": {},
        "confidence_hedging": {},
        "confidence_drop": {},
        "confidence_validation_gap": {},
    }

    max_distance = max(reverse_distance.values(), default=1)
    max_out_degree = max((len(graph.children[node_id]) for node_id in node_ids), default=1)
    max_in_degree = max((len(graph.parents[node_id]) for node_id in node_ids), default=1)

    for node_id in node_ids:
        step = graph.nodes[node_id]
        pos_ratio = position_index[node_id] / max(1, total_nodes - 1)
        earlyness = 1.0 - pos_ratio
        distance_ratio = reverse_distance.get(node_id, 0) / max(1, max_distance)

        content = f"{step.input} {step.output}".lower()
        tokens = content.split()
        keyword_hits = sum(1 for token in tokens if any(keyword in token for keyword in ERROR_TOKENS))
        hedge_hits = sum(1 for token in tokens if any(keyword == token or keyword in token for keyword in HEDGE_TOKENS))
        length_anomaly = 0.0
        if length_std > 0:
            length_anomaly = min(1.0, abs(token_lengths[node_id] - avg_length) / (3 * length_std))
        parent_ids = sorted(graph.parents.get(node_id, set()))
        parent_confidences = [
            graph.nodes[parent_id].confidence
            for parent_id in parent_ids
            if graph.nodes[parent_id].confidence is not None
        ]
        parent_keyword_hits = []
        for parent_id in parent_ids:
            parent_content = f"{graph.nodes[parent_id].input} {graph.nodes[parent_id].output}".lower().split()
            parent_keyword_hits.append(sum(1 for token in parent_content if any(keyword in token for keyword in ERROR_TOKENS)))
        avg_parent_confidence = mean(parent_confidences) if parent_confidences else 0.5
        avg_parent_keyword_hits = mean(parent_keyword_hits) if parent_keyword_hits else 0.0
        confidence_value = step.confidence if step.confidence is not None else 0.5
        decision_consistency = float(step.metadata.get("decision_consistency", 1.0))
        handoff_completeness = float(step.metadata.get("handoff_completeness", 1.0))
        data_integrity = float(step.metadata.get("data_integrity", 1.0))
        role_boundary_score = float(step.metadata.get("role_boundary_score", 1.0))
        validation_recorded = bool(step.metadata.get("validation_recorded", True))

        raw["position_earlyness"][node_id] = earlyness
        raw["position_distance"][node_id] = distance_ratio
        raw["position_forward_index"][node_id] = pos_ratio
        raw["structure_out_degree"][node_id] = len(graph.children[node_id]) / max(1, max_out_degree)
        raw["structure_reachability"][node_id] = reachability[node_id] / max(1, total_nodes - 1)
        raw["structure_betweenness"][node_id] = betweenness[node_id]
        raw["structure_in_degree"][node_id] = len(graph.parents[node_id]) / max(1, max_in_degree)
        raw["content_error_keywords"][node_id] = float(keyword_hits)
        raw["content_uncertainty"][node_id] = float(hedge_hits)
        raw["content_length_anomaly"][node_id] = length_anomaly
        raw["content_keyword_density"][node_id] = keyword_hits / max(1, len(tokens))
        raw["content_novelty"][node_id] = max(0.0, float(keyword_hits) - avg_parent_keyword_hits)
        raw["content_log_anomaly"][node_id] = 1.0 - mean((decision_consistency, handoff_completeness, data_integrity))
        raw["flow_cross_agent"][node_id] = float(incoming_cross_agent(graph, node_id) or outgoing_cross_agent(graph, node_id))
        raw["flow_role_criticality"][node_id] = role_weights.get(step.agent, 0.5)
        raw["flow_communication"][node_id] = float(edge_type_count(graph, node_id, "communication") > 0 or step.message_to is not None)
        raw["flow_role_mismatch"][node_id] = 1.0 - role_boundary_score
        raw["confidence_inverse"][node_id] = 1.0 - confidence_value
        raw["confidence_hedging"][node_id] = min(1.0, hedge_hits / 4.0)
        raw["confidence_drop"][node_id] = max(0.0, avg_parent_confidence - confidence_value)
        raw["confidence_validation_gap"][node_id] = float(not validation_recorded)

    normalized = {name: _normalize(values) for name, values in raw.items()}
    group_scores: dict[int, dict[str, float]] = {}
    for node_id in node_ids:
        group_scores[node_id] = {
            "position": mean(normalized[name][node_id] for name in ("position_earlyness", "position_distance", "position_forward_index")),
            "structure": mean(
                normalized[name][node_id]
                for name in ("structure_out_degree", "structure_reachability", "structure_betweenness", "structure_in_degree")
            ),
            "content": mean(
                normalized[name][node_id]
                for name in (
                    "content_error_keywords",
                    "content_uncertainty",
                    "content_length_anomaly",
                    "content_keyword_density",
                    "content_novelty",
                    "content_log_anomaly",
                )
            ),
            "flow": mean(
                normalized[name][node_id]
                for name in ("flow_cross_agent", "flow_role_criticality", "flow_communication", "flow_role_mismatch")
            ),
            "confidence": mean(
                normalized[name][node_id]
                for name in ("confidence_inverse", "confidence_hedging", "confidence_drop", "confidence_validation_gap")
            ),
        }
    return group_scores


def _normalize(values: dict[int, float]) -> dict[int, float]:
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum - minimum <= 1e-8:
        return {key: 0.0 for key in values}
    return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}


def grid_search_weights(
    validation_scenarios: list[Scenario],
    max_depth: int = 10,
    profile: str = "paper",
) -> tuple[GroupWeights, dict[str, float]]:
    grid = weight_grid(profile)
    candidates: list[GroupWeights] = []
    for position, structure, content, flow, confidence in itertools.product(
        grid["position"],
        grid["structure"],
        grid["content"],
        grid["flow"],
        grid["confidence"],
    ):
        if abs((position + structure + content + flow + confidence) - 1.0) > 1e-9:
            continue
        candidates.append(GroupWeights(position, structure, content, flow, confidence))

    best_weights = default_weights()
    best_metrics = {"hit@1": -1.0, "hit@3": -1.0, "mrr": -1.0}
    for weights in candidates:
        ranker = AgentTraceRanker(weights=weights, max_depth=max_depth)
        hit1 = 0
        hit3 = 0
        mrr_sum = 0.0
        for scenario in validation_scenarios:
            ranking = ranker.rank(scenario).ordered_nodes
            truth = scenario.ground_truth.root_cause_node_id
            if ranking and ranking[0] == truth:
                hit1 += 1
            if truth in ranking[:3]:
                hit3 += 1
            mrr_sum += reciprocal_rank(ranking, truth)

        total = max(1, len(validation_scenarios))
        metrics = {"hit@1": hit1 / total, "hit@3": hit3 / total, "mrr": mrr_sum / total}
        if metrics["hit@1"] > best_metrics["hit@1"] or (
            metrics["hit@1"] == best_metrics["hit@1"] and metrics["mrr"] > best_metrics["mrr"]
        ):
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def reciprocal_rank(ranking: list[int], truth: int) -> float:
    try:
        return 1.0 / (ranking.index(truth) + 1)
    except ValueError:
        return 0.0
