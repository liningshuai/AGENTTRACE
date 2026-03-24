from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import evaluate_agenttrace_breakdowns, evaluate_methods, learn_weights
from .generator import ScenarioGenerator
from .llm_baseline import LLMBaselineRunner, load_config_from_env
from .models import load_scenarios, save_json, save_scenarios
from .ranker import GroupWeights, default_weights
from .trail_gaia import run_gaia_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return _handle_generate(args)
    if args.command == "evaluate":
        return _handle_evaluate(args)
    if args.command == "pipeline":
        return _handle_pipeline(args)
    if args.command == "gaia":
        return _handle_gaia(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic reproduction of the AgentTrace paper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate synthetic benchmark data.")
    generate.add_argument("--output", type=Path, required=True, help="Path to the benchmark JSONL file.")
    generate.add_argument("--validation-output", type=Path, default=None, help="Optional path for validation JSONL.")
    generate.add_argument("--validation-count", type=int, default=50, help="Validation scenarios to generate.")
    generate.add_argument("--seed", type=int, default=42, help="Random seed.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate baselines on a benchmark.")
    evaluate.add_argument("--input", type=Path, required=True, help="Path to a benchmark JSONL file.")
    evaluate.add_argument("--validation", type=Path, default=None, help="Optional validation split used to learn weights.")
    evaluate.add_argument("--report", type=Path, default=None, help="Optional path to write a JSON report.")
    evaluate.add_argument("--seed", type=int, default=12345, help="Random seed for baselines.")
    evaluate.add_argument("--max-depth", type=int, default=10, help="Maximum backward tracing depth.")
    evaluate.add_argument("--learn-weights", action="store_true", help="Fit group weights on the validation split.")
    evaluate.add_argument("--weight-profile", choices=("paper", "wide", "robust"), default="paper", help="Grid used for validation-time weight search.")
    evaluate.add_argument("--include-llm", action="store_true", help="Evaluate the real LLM baseline using LLM_* environment variables.")
    evaluate.add_argument("--llm-max-scenarios", type=int, default=None, help="Optional cap for LLM baseline evaluation to control cost.")
    evaluate.add_argument("--llm-cache", type=Path, default=None, help="Optional cache path for LLM responses.")
    evaluate.add_argument("--llm-timeout", type=float, default=60.0, help="HTTP timeout in seconds for LLM requests.")

    pipeline = subparsers.add_parser("pipeline", help="Generate data and evaluate in one run.")
    pipeline.add_argument("--output-dir", type=Path, required=True, help="Directory to write datasets and reports.")
    pipeline.add_argument("--seed", type=int, default=42, help="Random seed.")
    pipeline.add_argument("--validation-count", type=int, default=50, help="Validation scenarios to generate.")
    pipeline.add_argument("--max-depth", type=int, default=10, help="Maximum backward tracing depth.")
    pipeline.add_argument("--weight-profile", choices=("paper", "wide", "robust"), default="paper", help="Grid used for validation-time weight search.")
    pipeline.add_argument("--include-llm", action="store_true", help="Evaluate the real LLM baseline using LLM_* environment variables.")
    pipeline.add_argument("--llm-max-scenarios", type=int, default=None, help="Optional cap for LLM baseline evaluation to control cost.")
    pipeline.add_argument("--llm-cache", type=Path, default=None, help="Optional cache path for LLM responses.")
    pipeline.add_argument("--llm-timeout", type=float, default=60.0, help="HTTP timeout in seconds for LLM requests.")

    gaia = subparsers.add_parser("gaia", help="Run the TRAIL/GAIA adaptation pipeline.")
    gaia.add_argument("--output-dir", type=Path, required=True, help="Directory to write the GAIA report.")
    gaia.add_argument("--include-llm", action="store_true", help="Evaluate the real LLM baseline on all GAIA traces.")
    gaia.add_argument("--llm-cache", type=Path, default=None, help="Optional cache path for GAIA LLM responses.")
    gaia.add_argument("--llm-timeout", type=float, default=60.0, help="HTTP timeout in seconds for GAIA LLM requests.")

    return parser


def _handle_generate(args: argparse.Namespace) -> int:
    generator = ScenarioGenerator(seed=args.seed)
    bundle = generator.generate_paper_bundle(validation_count=args.validation_count)
    save_scenarios(args.output, bundle.benchmark)
    print(f"Wrote benchmark with {len(bundle.benchmark)} scenarios to {args.output}")
    if args.validation_output is not None:
        save_scenarios(args.validation_output, bundle.validation)
        print(f"Wrote validation split with {len(bundle.validation)} scenarios to {args.validation_output}")
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.input)
    weights = default_weights()
    learned = None

    if args.learn_weights:
        if args.validation is None:
            raise SystemExit("--learn-weights requires --validation")
        validation_scenarios = load_scenarios(args.validation)
        learned = learn_weights(validation_scenarios, max_depth=args.max_depth, profile=args.weight_profile)
        weights = GroupWeights.from_dict(learned["weights"])
        print(f"Learned weights: {learned['weights']}")
        print(f"Validation metrics: {learned['validation_metrics']}")

    llm_runner = _build_llm_runner(args)
    method_results = evaluate_methods(
        scenarios,
        weights=weights,
        include_llm=args.include_llm,
        llm_runner=llm_runner,
        llm_max_scenarios=args.llm_max_scenarios,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    breakdowns = evaluate_agenttrace_breakdowns(scenarios, weights=weights, max_depth=args.max_depth)
    report = {
        "dataset_size": len(scenarios),
        "weights": weights.as_dict(),
        "learned": learned,
        "weight_profile": args.weight_profile,
        "methods": method_results,
        "agenttrace_breakdowns": breakdowns,
    }
    _print_report(report)
    if args.report is not None:
        save_json(args.report, report)
        print(f"Saved report to {args.report}")
    return 0


def _handle_pipeline(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    benchmark_path = output_dir / "benchmark.jsonl"
    validation_path = output_dir / "validation.jsonl"
    report_path = output_dir / "report.json"

    generator = ScenarioGenerator(seed=args.seed)
    bundle = generator.generate_paper_bundle(validation_count=args.validation_count)
    save_scenarios(benchmark_path, bundle.benchmark)
    save_scenarios(validation_path, bundle.validation)

    learned = learn_weights(bundle.validation, max_depth=args.max_depth, profile=args.weight_profile)
    weights = GroupWeights.from_dict(learned["weights"])
    llm_runner = _build_llm_runner(args)
    method_results = evaluate_methods(
        bundle.benchmark,
        weights=weights,
        include_llm=args.include_llm,
        llm_runner=llm_runner,
        llm_max_scenarios=args.llm_max_scenarios,
        max_depth=args.max_depth,
    )
    breakdowns = evaluate_agenttrace_breakdowns(bundle.benchmark, weights=weights, max_depth=args.max_depth)
    report = {
        "dataset_size": len(bundle.benchmark),
        "weights": weights.as_dict(),
        "learned": learned,
        "weight_profile": args.weight_profile,
        "methods": method_results,
        "agenttrace_breakdowns": breakdowns,
    }
    save_json(report_path, report)
    _print_report(report)
    print(f"Artifacts written under {output_dir}")
    return 0


def _handle_gaia(args: argparse.Namespace) -> int:
    report = run_gaia_pipeline(
        output_dir=args.output_dir,
        include_llm=args.include_llm,
        llm_cache=args.llm_cache,
        llm_timeout=args.llm_timeout,
    )
    dataset = report["dataset"]
    graph = report["graph_method"]
    print(
        "GAIA dataset: "
        f"traces={dataset['trace_count']}, "
        f"with_labels={dataset['traces_with_labels']}, "
        f"without_labels={dataset['traces_without_labels']}, "
        f"avg_candidates={dataset['avg_candidate_count']:.2f}"
    )
    print(
        "Graph method: "
        f"hit@1_any={graph['hit@1_any']:.3f}, "
        f"hit@3_any={graph['hit@3_any']:.3f}, "
        f"mrr_any={graph['mrr_any']:.3f}, "
        f"hit@1_root_proxy={graph['hit@1_root_proxy']:.3f}, "
        f"mean_runtime_ms={graph['mean_runtime_ms']:.2f}"
    )
    if "llm_method" in report:
        llm = report["llm_method"]
        print(
            "LLM method: "
            f"hit@1_any={llm['hit@1_any']:.3f}, "
            f"hit@3_any={llm['hit@3_any']:.3f}, "
            f"mrr_any={llm['mrr_any']:.3f}, "
            f"hit@1_root_proxy={llm['hit@1_root_proxy']:.3f}, "
            f"mean_runtime_ms={llm['mean_runtime_ms']:.2f}"
        )
    print(f"Artifacts written under {args.output_dir}")
    return 0


def _print_report(report: dict[str, object]) -> None:
    print(f"Dataset size: {report['dataset_size']}")
    print(f"Weights: {report['weights']}")
    print("Methods:")
    methods = report["methods"]
    assert isinstance(methods, dict)
    for name, metrics in methods.items():
        assert isinstance(metrics, dict)
        print(
            "  "
            f"{name}: hit@1={metrics['hit@1']:.3f}, "
            f"hit@3={metrics['hit@3']:.3f}, "
            f"mrr={metrics['mrr']:.3f}, "
            f"mean_runtime_ms={metrics['mean_runtime_ms']:.2f}, "
            f"n={metrics['evaluated_scenarios']}"
        )


def _build_llm_runner(args: argparse.Namespace) -> LLMBaselineRunner | None:
    if not getattr(args, "include_llm", False):
        return None
    config = load_config_from_env(timeout_s=args.llm_timeout, cache_path=args.llm_cache)
    return LLMBaselineRunner(config)
