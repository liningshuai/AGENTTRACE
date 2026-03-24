from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request

from .models import Scenario, save_json


PROMPT_TEMPLATE = """You are an expert debugger analyzing a multi-agent system execution trace.
The system encountered an error.

Your task is to identify the ROOT CAUSE - the earliest step where something went wrong that led to the final error.

## Execution Trace:
{trace_content}

## Error Description:
The system failed at step {error_step}: {error_description}

## Instructions:
1. Analyze the execution trace carefully
2. Identify causal relationships between steps
3. Find the EARLIEST step that caused the error
4. Consider: logic errors, miscommunication, data issues, missing validation

## Output Format:
Respond with ONLY the step number, e.g., "3"

Root cause step:
"""


@dataclass(slots=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_s: float = 60.0
    temperature: float = 0.0
    max_retries: int = 3
    cache_path: Path | None = None


def build_prompt(scenario: Scenario) -> str:
    lines: list[str] = []
    for step in scenario.steps:
        lines.append(f"Step {step.step_id} [{step.agent}] {step.action_type}")
        lines.append(f"Input: {step.input}")
        lines.append(f"Output: {step.output}")
    error_step = scenario.ground_truth.error_node_id
    error_description = scenario.steps[error_step - 1].output
    return PROMPT_TEMPLATE.format(
        trace_content="\n".join(lines),
        error_step=error_step,
        error_description=error_description,
    )


def load_config_from_env(
    timeout_s: float = 60.0,
    temperature: float = 0.0,
    max_retries: int = 3,
    cache_path: Path | None = None,
) -> LLMConfig:
    dotenv_values = _load_dotenv_values(Path(".env"))
    api_key = os.getenv("LLM_API_KEY") or dotenv_values.get("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or dotenv_values.get("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL") or dotenv_values.get("LLM_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL must be set to use the real LLM baseline.")
    return LLMConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_s=timeout_s,
        temperature=temperature,
        max_retries=max_retries,
        cache_path=cache_path,
    )


class LLMBaselineRunner:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._cache = self._load_cache(config.cache_path)

    def rank(self, scenario: Scenario) -> list[int]:
        prediction = self.predict_step(scenario)
        node_ids = [step.step_id for step in scenario.steps]
        if prediction is None:
            return node_ids
        remainder = [node_id for node_id in node_ids if node_id != prediction]
        remainder.sort(key=lambda node_id: (abs(node_id - prediction), node_id))
        return [prediction] + remainder

    def predict_step(self, scenario: Scenario) -> int | None:
        cache_key = f"{scenario.scenario_id}:{self.config.model}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached.get("predicted_step")

        prompt = build_prompt(scenario)
        response_text = self._chat_completion(prompt)
        predicted_step = _extract_step_number(response_text, valid_steps={step.step_id for step in scenario.steps})
        self._cache[cache_key] = {
            "scenario_id": scenario.scenario_id,
            "model": self.config.model,
            "predicted_step": predicted_step,
            "raw_response": response_text,
        }
        self._flush_cache()
        return predicted_step

    def _chat_completion(self, prompt: str) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            req = request.Request(url=url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.config.timeout_s) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                return str(data["choices"][0]["message"]["content"]).strip()
            except (error.HTTPError, error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM request failed after {self.config.max_retries} attempts: {last_error}") from last_error

    def _load_cache(self, cache_path: Path | None) -> dict[str, dict[str, Any]]:
        if cache_path is None or not cache_path.exists():
            return {}
        return json.loads(cache_path.read_text(encoding="utf-8"))

    def _flush_cache(self) -> None:
        if self.config.cache_path is None:
            return
        self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.config.cache_path, self._cache)


def llm_baseline_placeholder(_: Scenario, __: LLMConfig | None = None) -> list[int]:
    raise RuntimeError(
        "The LLM baseline is intentionally left as a placeholder. "
        "Set the LLM_* environment variables and use LLMBaselineRunner instead."
    )


def _extract_step_number(response_text: str, valid_steps: set[int]) -> int | None:
    match = re.search(r"\b(\d+)\b", response_text)
    if not match:
        return None
    predicted = int(match.group(1))
    if predicted in valid_steps:
        return predicted
    return None


def _load_dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
