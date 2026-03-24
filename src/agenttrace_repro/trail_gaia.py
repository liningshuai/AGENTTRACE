from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
from statistics import mean
import time
from typing import Any

from datasets import load_dataset

from .llm_baseline import LLMConfig, LLMBaselineRunner, load_config_from_env
from .models import save_json


WRAPPER_SPAN_NAMES = {
    "main",
    "get_examples_to_answer",
    "answer_single_question",
    "create_agent_hierarchy",
    "CodeAgent.run",
    "ToolCallingAgent.run",
}
KEYWORDS = (
    "error",
    "failed",
    "exception",
    "invalid",
    "not found",
    "hallucinated",
    "unable",
    "cannot",
    "incorrect",
    "wrong",
)


@dataclass(slots=True)
class GaiaSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    timestamp: str
    span_name: str
    status_code: str
    span_kind: str
    depth: int
    child_count: int
    step_number: int | None
    raw: dict[str, Any] = field(repr=False)


@dataclass(slots=True)
class GaiaTrace:
    trace_id: str
    candidate_spans: list[GaiaSpan]
    positive_span_ids: set[str]
    root_proxy_span_id: str | None
    scores: dict[str, Any]
    category_counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GaiaWeights:
    late: float = 0.10
    depth: float = 0.20
    llm: float = 0.20
    tool: float = 0.20
    keywords: float = 0.10
    text_len: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "late": self.late,
            "depth": self.depth,
            "llm": self.llm,
            "tool": self.tool,
            "keywords": self.keywords,
            "text_len": self.text_len,
        }


def parse_relaxed_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(cleaned)


def load_gaia_traces() -> list[GaiaTrace]:
    dataset = load_dataset("PatronusAI/TRAIL", split="gaia")
    traces: list[GaiaTrace] = []
    for row in dataset:
        trace = parse_relaxed_json(row["trace"])
        labels = parse_relaxed_json(row["labels"])
        traces.append(_build_gaia_trace(trace, labels))
    return traces


def _build_gaia_trace(trace: dict[str, Any], labels: dict[str, Any]) -> GaiaTrace:
    flattened = _flatten_spans(trace["trace_id"], trace.get("spans", []))
    by_id = {span.span_id: span for span in flattened}
    flattened = [
        GaiaSpan(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            timestamp=span.timestamp,
            span_name=span.span_name,
            status_code=span.status_code,
            span_kind=span.span_kind,
            depth=span.depth,
            child_count=span.child_count,
            step_number=span.step_number if span.step_number is not None else _nearest_step_number(span, by_id),
            raw=span.raw,
        )
        for span in flattened
    ]
    candidates = [span for span in flattened if _is_action_span(span)]
    candidates.sort(key=lambda span: (span.timestamp, span.span_id))

    unique_positive_ids = {error["location"] for error in labels.get("errors", [])}
    positive_ids = unique_positive_ids & {span.span_id for span in candidates}
    root_proxy_span_id = None
    for span in candidates:
        if span.span_id in positive_ids:
            root_proxy_span_id = span.span_id
            break

    categories = Counter(error.get("category", "Unknown") for error in labels.get("errors", []))
    return GaiaTrace(
        trace_id=trace["trace_id"],
        candidate_spans=candidates,
        positive_span_ids=positive_ids,
        root_proxy_span_id=root_proxy_span_id,
        scores=(labels.get("scores") or [{}])[0],
        category_counts=dict(categories),
        metadata={
            "total_flat_spans": len(flattened),
            "candidate_count": len(candidates),
            "positive_count": len(positive_ids),
        },
    )


def _nearest_step_number(span: GaiaSpan, by_id: dict[str, GaiaSpan]) -> int | None:
    current = span
    while current.parent_span_id is not None:
        parent = by_id.get(current.parent_span_id)
        if parent is None:
            return None
        if parent.step_number is not None:
            return parent.step_number
        current = parent
    return None


def _flatten_spans(trace_id: str, root_spans: list[dict[str, Any]]) -> list[GaiaSpan]:
    spans: list[GaiaSpan] = []

    def recurse(span: dict[str, Any], depth: int = 0) -> None:
        children = span.get("child_spans") or []
        span_name = str(span.get("span_name") or "")
        spans.append(
            GaiaSpan(
                trace_id=trace_id,
                span_id=str(span["span_id"]),
                parent_span_id=span.get("parent_span_id"),
                timestamp=str(span.get("timestamp") or ""),
                span_name=span_name,
                status_code=str(span.get("status_code") or "Unset"),
                span_kind=str(span.get("span_kind") or "Internal"),
                depth=depth,
                child_count=len(children),
                step_number=_extract_step_number(span_name),
                raw=span,
            )
        )
        for child in children:
            recurse(child, depth + 1)

    for span in root_spans:
        recurse(span)
    return spans


def _extract_step_number(span_name: str) -> int | None:
    if not span_name.startswith("Step "):
        return None
    try:
        return int(span_name.split()[1])
    except (IndexError, ValueError):
        return None


def _is_action_span(span: GaiaSpan) -> bool:
    if span.child_count != 0:
        return False
    if span.span_name.startswith("Step "):
        return False
    if span.span_name in WRAPPER_SPAN_NAMES:
        return False
    return True


def span_text(span: GaiaSpan) -> str:
    attrs = span.raw.get("span_attributes") or {}
    parts: list[str] = []
    for key in (
        "input.value",
        "output.value",
        "llm.input_messages.0.message.content",
        "llm.output_messages.0.message.content",
    ):
        value = attrs.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    for event in span.raw.get("events") or []:
        event_attrs = event.get("Attributes") or {}
        for key in ("exception.message", "exception.type"):
            value = event_attrs.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
    for log in span.raw.get("logs") or []:
        body = log.get("body")
        if body is not None:
            parts.append(str(body))
    text = "\n".join(parts)
    return text[:5000]


def rank_gaia_trace(trace: GaiaTrace, weights: GaiaWeights = GaiaWeights()) -> list[str]:
    scores = score_gaia_trace(trace, weights=weights)
    ranked_spans = sorted(trace.candidate_spans, key=lambda span: scores[span.span_id], reverse=True)
    return [span.span_id for span in ranked_spans]


def score_gaia_trace(trace: GaiaTrace, weights: GaiaWeights = GaiaWeights()) -> dict[str, float]:
    candidates = trace.candidate_spans
    if not candidates:
        return {}

    total = len(candidates)
    max_depth = max(span.depth for span in candidates) or 1
    raw = {
        "late": {},
        "depth": {},
        "llm": {},
        "tool": {},
        "keywords": {},
        "text_len": {},
    }

    for index, span in enumerate(candidates):
        text = span_text(span).lower()
        span_id = span.span_id
        raw["late"][span_id] = index / max(1, total - 1)
        raw["depth"][span_id] = span.depth / max_depth
        raw["llm"][span_id] = float("LiteLLMModel.__call__" in span.span_name)
        raw["tool"][span_id] = float(("Tool" in span.span_name) and ("LiteLLMModel" not in span.span_name))
        raw["keywords"][span_id] = sum(1 for keyword in KEYWORDS if keyword in text)
        raw["text_len"][span_id] = min(len(text) / 4000.0, 1.0)

    normalized = {name: _normalize(values) for name, values in raw.items()}
    scores: dict[str, float] = {}
    for span in candidates:
        span_id = span.span_id
        scores[span_id] = (
            weights.late * normalized["late"][span_id]
            + weights.depth * normalized["depth"][span_id]
            + weights.llm * normalized["llm"][span_id]
            + weights.tool * normalized["tool"][span_id]
            + weights.keywords * normalized["keywords"][span_id]
            + weights.text_len * normalized["text_len"][span_id]
        )
    return scores


def _normalize(values: dict[str, float]) -> dict[str, float]:
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum - minimum <= 1e-9:
        return {key: 0.0 for key in values}
    return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}


def evaluate_gaia_graph(traces: list[GaiaTrace], weights: GaiaWeights = GaiaWeights()) -> dict[str, Any]:
    scored = []
    runtimes = []
    for trace in traces:
        started = time.perf_counter()
        ranking = rank_gaia_trace(trace, weights=weights)
        runtimes.append((time.perf_counter() - started) * 1000.0)
        scored.append((trace, ranking))
    metrics = _evaluate_gaia_rankings(scored, name="agenttrace_gaia")
    metrics["mean_runtime_ms"] = mean(runtimes) if runtimes else 0.0
    return metrics


def evaluate_gaia_baselines(traces: list[GaiaTrace], seed: int = 12345) -> dict[str, dict[str, Any]]:
    rng = random.Random(seed)

    def random_rank(trace: GaiaTrace) -> list[str]:
        ids = [span.span_id for span in trace.candidate_spans]
        rng.shuffle(ids)
        return ids

    def first_rank(trace: GaiaTrace) -> list[str]:
        return [span.span_id for span in trace.candidate_spans]

    def last_rank(trace: GaiaTrace) -> list[str]:
        return [span.span_id for span in reversed(trace.candidate_spans)]

    baselines = {
        "random": random_rank,
        "first_action": first_rank,
        "last_action": last_rank,
    }
    results: dict[str, dict[str, Any]] = {}
    for name, ranker in baselines.items():
        started = []
        scored = []
        for trace in traces:
            begin = time.perf_counter()
            ranking = ranker(trace)
            started.append((time.perf_counter() - begin) * 1000.0)
            scored.append((trace, ranking))
        results[name] = _evaluate_gaia_rankings(scored, name=name)
        results[name]["mean_runtime_ms"] = mean(started) if started else 0.0
    return results


def _evaluate_gaia_rankings(scored: Any, name: str) -> dict[str, Any]:
    scored = list(scored)
    hit1_any = hit3_any = hit5_any = 0
    hit1_root = hit3_root = hit5_root = 0
    mrr_any = 0.0
    mrr_root = 0.0
    recall1 = 0.0
    recall3 = 0.0
    recall5 = 0.0
    evaluated = 0
    root_evaluated = 0

    for trace, ranking in scored:
        positives = trace.positive_span_ids
        if positives:
            evaluated += 1
            first_positive_rank = _first_relevant_rank(ranking, positives)
            hit1_any += int(first_positive_rank == 1)
            hit3_any += int(first_positive_rank is not None and first_positive_rank <= 3)
            hit5_any += int(first_positive_rank is not None and first_positive_rank <= 5)
            mrr_any += 0.0 if first_positive_rank is None else 1.0 / first_positive_rank
            recall1 += len(set(ranking[:1]) & positives) / len(positives)
            recall3 += len(set(ranking[:3]) & positives) / len(positives)
            recall5 += len(set(ranking[:5]) & positives) / len(positives)

        if trace.root_proxy_span_id is not None:
            root_evaluated += 1
            rank = _first_relevant_rank(ranking, {trace.root_proxy_span_id})
            hit1_root += int(rank == 1)
            hit3_root += int(rank is not None and rank <= 3)
            hit5_root += int(rank is not None and rank <= 5)
            mrr_root += 0.0 if rank is None else 1.0 / rank

    candidate_counts = [trace.metadata["candidate_count"] for trace, _ in scored]
    positive_counts = [trace.metadata["positive_count"] for trace, _ in scored]
    return {
        "name": name,
        "evaluated_traces_any": evaluated,
        "evaluated_traces_root_proxy": root_evaluated,
        "hit@1_any": hit1_any / max(1, evaluated),
        "hit@3_any": hit3_any / max(1, evaluated),
        "hit@5_any": hit5_any / max(1, evaluated),
        "mrr_any": mrr_any / max(1, evaluated),
        "recall@1": recall1 / max(1, evaluated),
        "recall@3": recall3 / max(1, evaluated),
        "recall@5": recall5 / max(1, evaluated),
        "hit@1_root_proxy": hit1_root / max(1, root_evaluated),
        "hit@3_root_proxy": hit3_root / max(1, root_evaluated),
        "hit@5_root_proxy": hit5_root / max(1, root_evaluated),
        "mrr_root_proxy": mrr_root / max(1, root_evaluated),
        "avg_candidate_count": mean(candidate_counts),
        "avg_positive_count": mean(positive_counts),
    }


def _first_relevant_rank(ranking: list[str], positives: set[str]) -> int | None:
    for index, span_id in enumerate(ranking, start=1):
        if span_id in positives:
            return index
    return None


class GaiaLLMRunner:
    def __init__(self, runner: LLMBaselineRunner) -> None:
        self.runner = runner

    def rank(self, trace: GaiaTrace) -> list[str]:
        prompt = build_gaia_llm_prompt(trace)
        response = self.runner._chat_completion(prompt)
        predicted_ids = _extract_span_ids(response, valid_ids={span.span_id for span in trace.candidate_spans})
        ranking = predicted_ids[:]
        remaining = [span.span_id for span in trace.candidate_spans if span.span_id not in predicted_ids]
        ranking.extend(remaining)
        return ranking


def build_gaia_llm_prompt(trace: GaiaTrace, max_chars_per_span: int = 280) -> str:
    lines = [
        "You are debugging an agent execution trace.",
        "Identify the action spans most likely to contain substantive execution mistakes or hallucinations.",
        "Return up to 5 span IDs ranked from most likely to least likely.",
        "Prefer the earliest causally important mistake when several spans look related.",
        "Output only a JSON array of span IDs, for example: [\"abc\", \"def\"]",
        "",
        "Candidate action spans:",
    ]
    for index, span in enumerate(trace.candidate_spans, start=1):
        text = span_text(span).replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        excerpt = text[:max_chars_per_span]
        step_number = span.step_number if span.step_number is not None else "NA"
        lines.append(
            f"{index}. span_id={span.span_id} | step={step_number} | name={span.span_name} | "
            f"status={span.status_code} | depth={span.depth} | text={excerpt}"
        )
    return "\n".join(lines)


def _extract_span_ids(response_text: str, valid_ids: set[str]) -> list[str]:
    candidates = re.findall(r"[0-9a-f]{16}", response_text)
    ranked: list[str] = []
    for candidate in candidates:
        if candidate in valid_ids and candidate not in ranked:
            ranked.append(candidate)
    return ranked


def evaluate_gaia_llm(
    traces: list[GaiaTrace],
    cache_path: Path,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    config = load_config_from_env(timeout_s=timeout_s, cache_path=cache_path)
    runner = GaiaLLMRunner(LLMBaselineRunner(config))
    scored = []
    for trace in traces:
        started = time.perf_counter()
        ranking = runner.rank(trace)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        scored.append((trace, ranking, runtime_ms))

    metrics = _evaluate_gaia_rankings(((trace, ranking) for trace, ranking, _ in scored), name="llm_gaia")
    metrics["mean_runtime_ms"] = mean(runtime_ms for _, _, runtime_ms in scored)
    return metrics


def summarize_gaia_dataset(traces: list[GaiaTrace]) -> dict[str, Any]:
    category_counts = Counter()
    for trace in traces:
        category_counts.update(trace.category_counts)
    return {
        "trace_count": len(traces),
        "traces_with_labels": sum(1 for trace in traces if trace.positive_span_ids),
        "traces_without_labels": sum(1 for trace in traces if not trace.positive_span_ids),
        "avg_candidate_count": mean(trace.metadata["candidate_count"] for trace in traces),
        "avg_positive_count": mean(trace.metadata["positive_count"] for trace in traces),
        "category_counts": dict(category_counts.most_common()),
    }


def run_gaia_pipeline(
    output_dir: Path,
    include_llm: bool = False,
    llm_cache: Path | None = None,
    llm_timeout: float = 60.0,
    traces: list[GaiaTrace] | None = None,
) -> dict[str, Any]:
    traces = traces or load_gaia_traces()
    dataset_summary = summarize_gaia_dataset(traces)
    graph_metrics = evaluate_gaia_graph(traces)
    baseline_metrics = evaluate_gaia_baselines(traces)
    report: dict[str, Any] = {
        "dataset": dataset_summary,
        "graph_method": graph_metrics,
        "baselines": baseline_metrics,
    }
    if include_llm:
        cache_path = llm_cache or (output_dir / "llm_cache_gaia.json")
        report["llm_method"] = evaluate_gaia_llm(traces, cache_path=cache_path, timeout_s=llm_timeout)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "gaia_report.json", report)
    return report


def build_gaia_trace_comparison(traces: list[GaiaTrace], weights: GaiaWeights = GaiaWeights()) -> dict[str, Any]:
    comparison_rows: list[dict[str, Any]] = []
    for trace in traces:
        scores = score_gaia_trace(trace, weights=weights)
        ranking = rank_gaia_trace(trace, weights=weights)
        positives = trace.positive_span_ids
        root_proxy = trace.root_proxy_span_id
        top1 = ranking[:1]
        top3 = ranking[:3]
        top5 = ranking[:5]
        by_id = {span.span_id: span for span in trace.candidate_spans}

        comparison_rows.append(
            {
                "trace_id": trace.trace_id,
                "notes": {
                    "has_official_human_root_cause": False,
                    "root_proxy_is_approximation": True,
                },
                "human_labels": {
                    "positive_span_ids": sorted(positives),
                    "positive_count": len(positives),
                    "root_proxy_span_id": root_proxy,
                    "root_proxy_span": _serialize_gaia_span(by_id.get(root_proxy), score=scores.get(root_proxy)) if root_proxy else None,
                },
                "predicted": {
                    "top1_span_id": top1[0] if top1 else None,
                    "top3_span_ids": top3,
                    "top5_span_ids": top5,
                    "top1_hit_any": bool(set(top1) & positives),
                    "top3_hit_any": bool(set(top3) & positives),
                    "top5_hit_any": bool(set(top5) & positives),
                    "top1_hit_root_proxy": bool(root_proxy is not None and root_proxy in top1),
                    "top3_hit_root_proxy": bool(root_proxy is not None and root_proxy in top3),
                    "top5_hit_root_proxy": bool(root_proxy is not None and root_proxy in top5),
                    "recall@1": len(set(top1) & positives) / len(positives) if positives else 0.0,
                    "recall@3": len(set(top3) & positives) / len(positives) if positives else 0.0,
                    "recall@5": len(set(top5) & positives) / len(positives) if positives else 0.0,
                },
                "positive_spans": [
                    _serialize_gaia_span(by_id[span_id], score=scores.get(span_id))
                    for span_id in sorted(positives)
                    if span_id in by_id
                ],
                "ranked_spans": [
                    {
                        "rank": index,
                        **_serialize_gaia_span(by_id[span_id], score=scores.get(span_id)),
                        "is_human_labeled_positive": span_id in positives,
                        "is_root_proxy": span_id == root_proxy,
                    }
                    for index, span_id in enumerate(ranking, start=1)
                    if span_id in by_id
                ],
            }
        )

    return {
        "dataset": "PatronusAI/TRAIL:gaia",
        "method": "agenttrace_gaia",
        "trace_count": len(comparison_rows),
        "weights": weights.as_dict(),
        "notes": {
            "positive_span_ids_are_human_labeled_bad_locations": True,
            "root_proxy_span_id_is_the_earliest_labeled_positive_among_candidate_action_spans": True,
            "gaia_does_not_provide_a_single_official_root_cause_label": True,
        },
        "traces": comparison_rows,
    }


def _serialize_gaia_span(span: GaiaSpan | None, score: float | None = None) -> dict[str, Any] | None:
    if span is None:
        return None
    return {
        "span_id": span.span_id,
        "step_number": span.step_number,
        "span_name": span.span_name,
        "status_code": span.status_code,
        "span_kind": span.span_kind,
        "depth": span.depth,
        "child_count": span.child_count,
        "timestamp": span.timestamp,
        "score": score,
        "text_excerpt": span_text(span)[:320],
    }
