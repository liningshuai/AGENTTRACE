from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Step:
    step_id: int
    agent: str
    action_type: str
    input: str
    output: str
    timestamp: str
    produces: list[str] = field(default_factory=list)
    consumes: list[str] = field(default_factory=list)
    message_to: str | None = None
    confidence: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent": self.agent,
            "action_type": self.action_type,
            "input": self.input,
            "output": self.output,
            "timestamp": self.timestamp,
            "produces": self.produces,
            "consumes": self.consumes,
            "message_to": self.message_to,
            "confidence": self.confidence,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Step:
        return cls(
            step_id=int(payload["step_id"]),
            agent=str(payload["agent"]),
            action_type=str(payload["action_type"]),
            input=str(payload["input"]),
            output=str(payload["output"]),
            timestamp=str(payload["timestamp"]),
            produces=list(payload.get("produces", [])),
            consumes=list(payload.get("consumes", [])),
            message_to=payload.get("message_to"),
            confidence=payload.get("confidence"),
            tags=list(payload.get("tags", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class GroundTruth:
    error_node_id: int
    root_cause_node_id: int
    bug_type: str
    bug_description: str
    bug_position_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_node_id": self.error_node_id,
            "root_cause_node_id": self.root_cause_node_id,
            "bug_type": self.bug_type,
            "bug_description": self.bug_description,
            "bug_position_bucket": self.bug_position_bucket,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GroundTruth:
        return cls(
            error_node_id=int(payload["error_node_id"]),
            root_cause_node_id=int(payload["root_cause_node_id"]),
            bug_type=str(payload["bug_type"]),
            bug_description=str(payload["bug_description"]),
            bug_position_bucket=str(payload["bug_position_bucket"]),
        )


@dataclass(slots=True)
class Scenario:
    scenario_id: str
    domain: str
    domain_group: str
    interaction_pattern: str
    agents: list[str]
    steps: list[Step]
    ground_truth: GroundTruth
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "domain": self.domain,
            "domain_group": self.domain_group,
            "interaction_pattern": self.interaction_pattern,
            "agents": self.agents,
            "steps": [step.to_dict() for step in self.steps],
            "ground_truth": self.ground_truth.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Scenario:
        return cls(
            scenario_id=str(payload["scenario_id"]),
            domain=str(payload["domain"]),
            domain_group=str(payload["domain_group"]),
            interaction_pattern=str(payload["interaction_pattern"]),
            agents=list(payload["agents"]),
            steps=[Step.from_dict(item) for item in payload["steps"]],
            ground_truth=GroundTruth.from_dict(payload["ground_truth"]),
            metadata=dict(payload.get("metadata", {})),
        )


def save_scenarios(path: Path, scenarios: list[Scenario]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario.to_dict(), ensure_ascii=False))
            handle.write("\n")


def load_scenarios(path: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            scenarios.append(Scenario.from_dict(json.loads(line)))
    return scenarios


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
