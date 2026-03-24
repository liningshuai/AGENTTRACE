# AgentTrace-Repro

Unofficial reproduction of **"AGENTTRACE: CAUSAL GRAPH TRACING FOR ROOT CAUSE ANALYSIS IN DEPLOYED MULTI-AGENT SYSTEMS"** (published as a workshop paper at the **ICLR 2026 AIWILD Workshop**).

> This repository is **not** the official code release and is **not** maintained by the paper authors.
> The original paper referenced a public repository, but that repository was unavailable when this reproduction was built.
> This codebase therefore reconstructs the benchmark and method from the paper text and appendix details, then extends the study to the real-world `PatronusAI/TRAIL` GAIA subset.

## Overview

This repository implements:

- a paper-style synthetic benchmark generator for multi-agent failure traces
- causal graph construction over agent execution steps
- backward tracing from a known final error node
- lightweight root-cause ranking with grouped features
- classic baselines and a real LLM baseline
- an additional adaptation to the `TRAIL/GAIA` dataset from Hugging Face

The project is intentionally split into two experimental settings:

1. **Synthetic AgentTrace-style reproduction**
   - built to match the paper's intended task as closely as possible
   - one scenario has one injected upstream bug and one designated root cause

2. **Real-world GAIA extension**
   - built on human-labeled telemetry traces from Hugging Face
   - one trace can contain multiple labeled bad locations, so the task becomes closer to error localization than strict single-root-cause recovery

## Reproduction Status

Current reproduced findings:

- On the reconstructed synthetic benchmark, the graph-based method clearly outperforms simple baselines and remains substantially faster than an API-backed LLM baseline.
- On `TRAIL/GAIA`, the lightweight graph-style ranking outperforms the current real LLM prompt on `top-k` error localization metrics.
- Exact paper numbers are **not** guaranteed because the official benchmark generator and original data release were not available.

## Repository Layout

Main files:

- `pyproject.toml`: project metadata managed by `uv`
- `src/agenttrace_repro/cli.py`: command-line entrypoints
- `src/agenttrace_repro/generator.py`: synthetic benchmark generation
- `src/agenttrace_repro/graph.py`: causal graph construction and backward traversal
- `src/agenttrace_repro/ranker.py`: feature extraction, scoring, and weight search
- `src/agenttrace_repro/evaluation.py`: synthetic benchmark evaluation
- `src/agenttrace_repro/llm_baseline.py`: OpenAI-compatible LLM baseline with caching
- `src/agenttrace_repro/trail_gaia.py`: GAIA ingestion, candidate construction, ranking, and evaluation
- `.env.example`: example environment variables for LLM evaluation

## Installation

This project uses `uv`.

```bash
uv sync
```

This creates `.venv` and installs the package in editable mode.

Python requirement:

- `>= 3.13`

## Datasets

### 1. Synthetic AgentTrace-Style Benchmark

This dataset is **constructed in this repository** to reproduce the experimental intent of the paper.

Why it exists:

- the workshop paper describes a synthetic benchmark
- the paper mentions a public repository
- that repository was not available when this reproduction was built
- to reproduce the method, we recreated the benchmark generator from the paper description instead of downloading official data

How the synthetic data is generated:

- `10` application domains inspired by the paper
- `3-5` agent roles per scenario
- `8-15` steps per trace
- `5` bug families:
  - logic error
  - communication failure
  - data corruption
  - missing validation
  - role confusion
- bug-position sampling follows the paper-style early / middle / late split:
  - early: `60%`
  - middle: `30%`
  - late: `10%`

What one synthetic scenario contains:

- a clean step sequence before the injected bug
- one injected upstream bug step
- downstream propagation steps
- one final error node where the failure becomes visible

Crucially, this benchmark gives us **two distinct labels**:

- `root_cause_node_id`: the injected bug step
- `error_node_id`: the final node where the failure manifests

That distinction is what makes the synthetic setup suitable for **root cause analysis**, not just error spotting.

Generated files:

- `benchmark.jsonl`: main evaluation split
- `validation.jsonl`: validation split used for weight search

### 2. TRAIL / GAIA from Hugging Face

This dataset is **not generated locally**. It is loaded directly from Hugging Face:

- dataset: `PatronusAI/TRAIL`
- subset used here: `gaia`
- source page: `https://huggingface.co/datasets/PatronusAI/TRAIL/viewer/default/gaia`

How it is loaded:

- via the `datasets` library
- each example contains:
  - `trace`: a nested telemetry span tree encoded as JSON text
  - `labels`: human-written annotations encoded as JSON text

What is different from the synthetic benchmark:

- traces are nested span trees, not pre-cleaned step lists
- labels point to **multiple error locations**
- labels do **not** specify one canonical root cause
- some labeled traces do not have an explicit runtime `Error` status
- in the current subset, `3` traces contain no positive labels

So GAIA is better understood as:

- **error localization**: "which spans look bad?"

rather than:

- **strict single-root-cause analysis**: "which one upstream step is the unique origin of the failure?"

Because of that mismatch, the GAIA pipeline reports two kinds of metrics:

- `*_any`: hit against **any** human-labeled bad span
- `*_root_proxy`: hit against the **earliest** labeled bad span, used only as a rough root-cause proxy

Important note:

- `root_proxy` is **not** an official GAIA label
- it is a convenience approximation used in this repository so we can ask a root-cause-like question on a dataset that was not originally labeled for single-root-cause evaluation

## Synthetic Method Summary

The synthetic reproduction follows the paper-style pipeline:

1. Build a directed graph where nodes are trace steps.
2. Add three edge types:
   - sequential edges
   - communication edges
   - data-dependency edges
3. Start from the known `error_node_id`.
4. Trace backward with breadth-first traversal to collect candidate ancestors.
5. Score candidate nodes using grouped features:
   - position
   - structure
   - content
   - flow
   - confidence
6. Rank nodes by weighted score.
7. Evaluate against `root_cause_node_id`.

Important interpretation:

- the backward BFS is the **candidate collection** step
- the final `Hit@1` / `Hit@3` / `MRR` numbers are measured against the **root cause**, not the final visible error node

## GAIA Adaptation Summary

The GAIA pipeline is deliberately different because the dataset structure is different.

For each trace:

1. Parse the nested telemetry tree.
2. Flatten all spans.
3. Keep only candidate `action spans`:
   - leaf spans only
   - remove wrappers such as `main`, `CodeAgent.run`, and `ToolCallingAgent.run`
   - remove `Step N` container spans
4. Score each candidate span using lightweight suspiciousness signals:
   - later position in the trace
   - tree depth
   - whether the span is an `LiteLLMModel.__call__`
   - whether the span is a tool call
   - error-like keywords in the span text
   - text length
5. Rank candidate spans by score.
6. Evaluate both:
   - whether the top-ranked span hits **any** labeled bad span
   - whether it hits the earliest labeled span as a root-cause proxy

This means the GAIA experiment is best read as:

- a real-world extension of the repository
- not a one-to-one reproduction of the paper's synthetic root-cause setup

## Running Synthetic Experiments

### End-to-End Synthetic Pipeline

```bash
uv run agenttrace-repro pipeline \
  --output-dir runs/paper_like \
  --weight-profile robust
```

This command will:

- generate `benchmark.jsonl`
- generate `validation.jsonl`
- learn ranking weights on the validation split
- evaluate graph and non-LLM baselines
- write `report.json`

### Generate Data Only

```bash
uv run agenttrace-repro generate \
  --output runs/custom/benchmark.jsonl \
  --validation-output runs/custom/validation.jsonl \
  --validation-count 50 \
  --seed 42
```

### Evaluate an Existing Synthetic Benchmark

```bash
uv run agenttrace-repro evaluate \
  --input runs/custom/benchmark.jsonl \
  --validation runs/custom/validation.jsonl \
  --learn-weights \
  --weight-profile robust \
  --report runs/custom/report.json
```

## Weight Profiles

Three validation-time search profiles are implemented:

### `paper`

- closest to the narrow grid described in the paper appendix
- use this when methodological faithfulness matters most

### `wide`

- explores a broader search grid
- useful while modifying the synthetic generator

### `robust`

- tuned for the blind synthetic variant in this repository
- currently the best default for graph-vs-LLM comparison on our reconstructed benchmark

## LLM Baseline Configuration

The real LLM baseline uses an OpenAI-compatible `chat/completions` API.

Required environment variables:

PowerShell:

```powershell
$env:LLM_API_KEY="your_key_here"
$env:LLM_BASE_URL="https://api.deepseek.com"
$env:LLM_MODEL="deepseek-chat"
```

Bash:

```bash
export LLM_API_KEY="your_key_here"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"
```

Optional helper file:

- `.env.example`

Recommended workflow:

PowerShell:

```powershell
Copy-Item .env.example .env
```

Bash:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your own values. The CLI will read `.env` automatically if shell variables are not already set.

### Synthetic Benchmark with Real LLM Baseline

```bash
uv run agenttrace-repro evaluate \
  --input runs/paper_like/benchmark.jsonl \
  --validation runs/paper_like/validation.jsonl \
  --learn-weights \
  --weight-profile robust \
  --include-llm \
  --llm-max-scenarios 50 \
  --llm-cache runs/paper_like/llm_cache_50.json \
  --report runs/paper_like/report_llm_50.json
```

Notes:

- if `benchmark.jsonl` and `validation.jsonl` do not exist yet, run the synthetic pipeline first
- `--llm-max-scenarios` is recommended first because API calls cost time and money
- `--llm-cache` avoids re-paying for the same scenario and model combination
- the current synthetic LLM prompt returns a single step id, so `Hit@1` is the most faithful comparison metric for that baseline

## Running GAIA Experiments

### GAIA Graph-Only

```bash
uv run agenttrace-repro gaia --output-dir runs/gaia_graph_only
```

### GAIA with Real LLM Baseline

```bash
uv run agenttrace-repro gaia \
  --output-dir runs/gaia_full \
  --include-llm \
  --llm-cache runs/gaia_full/llm_cache_gaia.json
```

This command evaluates the full current GAIA subset in Hugging Face, not a small sample.

## Current Results

### Synthetic Benchmark

From `runs/paper_like/report.json` and `runs/paper_like/report_llm_50.json`:

- benchmark size: `550` scenarios
- graph method (`agenttrace`):
  - `Hit@1 = 0.865`
  - `Hit@3 = 0.964`
  - `MRR = 0.918`
- real LLM baseline (`50`-scenario API subset):
  - `Hit@1 = 0.540`
  - `Hit@3 = 0.780`
  - `MRR = 0.681`

Interpretation:

- the overall paper-style trend is reproduced
- the graph method clearly beats simple baselines
- the reproduction does **not** exactly match the paper's claimed `94.9% Hit@1`
- this gap is expected because the official generator and official benchmark were unavailable

### TRAIL / GAIA

From `runs/gaia_full/gaia_report.json`:

- total traces: `117`
- traces with at least one positive label: `114`
- average candidate spans per trace: `18.15`
- average positive locations per trace: `3.31`

Graph method:

- `Hit@1_any = 0.596`
- `Hit@3_any = 0.825`
- `Hit@5_any = 0.904`
- `MRR_any = 0.735`
- `Hit@1_root_proxy = 0.167`
- `MRR_root_proxy = 0.387`

Real LLM baseline:

- `Hit@1_any = 0.175`
- `Hit@3_any = 0.605`
- `Hit@5_any = 0.772`
- `MRR_any = 0.417`
- `Hit@1_root_proxy = 0.053`
- `MRR_root_proxy = 0.263`

Interpretation:

- on GAIA, the graph-style method is much better at finding a human-labeled bad span than the current real LLM prompt
- recovering the earliest likely bad span is much harder than hitting any bad span
- this is expected because GAIA provides multiple annotated bad locations rather than one official root cause

## Output Files

Typical outputs:

- `runs/paper_like/report.json`
- `runs/paper_like/report_llm_50.json`
- `runs/smoke/report.json`
- `runs/gaia_full/gaia_report.json`

The repository intentionally does **not** track large generated datasets, validation splits, or LLM cache files. Only the selected summary JSON outputs are kept under `runs/`.

## Limitations

Please read the results with the right scope:

- this is an independent reproduction, not the official repository
- the synthetic benchmark is reconstructed from the paper description, not downloaded from the authors
- the GAIA experiment is an extension beyond the original paper
- the GAIA task is only partially comparable to the original paper because it lacks a single official root-cause label
- the current LLM baselines are prompt-sensitive and can likely be improved further

## Citation

If this repository is useful in your work:

- cite the original **AgentTrace** workshop paper for the method idea
- cite this repository separately as an unofficial reproduction if needed

## License

No separate license file has been added in this workspace yet. Add one before external distribution if required by your team or institution.
