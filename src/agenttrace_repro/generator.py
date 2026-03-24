from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections.abc import Iterable
import random

from .config import BUG_POSITION_WEIGHTS, BUG_TYPE_WEIGHTS, DOMAINS, PAPER_DOMAIN_COUNTS, DomainConfig
from .models import GroundTruth, Scenario, Step


BUG_DESCRIPTIONS = {
    "logic_error": "Incorrect decision rule or branch condition introduced upstream.",
    "communication_failure": "A critical detail was dropped or distorted during an inter-agent handoff.",
    "data_corruption": "Structured payload fields were swapped or overwritten with inconsistent values.",
    "missing_validation": "A required check was skipped, allowing a bad artifact to move downstream.",
    "role_confusion": "An agent acted outside its expertise and injected an unsafe assumption.",
}


@dataclass(slots=True)
class GenerationBundle:
    benchmark: list[Scenario]
    validation: list[Scenario]


class ScenarioGenerator:
    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._base_time = datetime(2026, 3, 1, 9, 0, 0)

    def generate_paper_bundle(self, validation_count: int = 50) -> GenerationBundle:
        benchmark: list[Scenario] = []
        for domain in DOMAINS:
            benchmark.extend(self.generate_domain_scenarios(domain, PAPER_DOMAIN_COUNTS[domain.name]))
        validation = self.generate_sampled_scenarios(validation_count)
        self._rng.shuffle(benchmark)
        return GenerationBundle(benchmark=benchmark, validation=validation)

    def generate_sampled_scenarios(self, count: int) -> list[Scenario]:
        weights = [PAPER_DOMAIN_COUNTS[domain.name] for domain in DOMAINS]
        scenarios: list[Scenario] = []
        for _ in range(count):
            domain = self._rng.choices(DOMAINS, weights=weights, k=1)[0]
            scenarios.append(self.generate_scenario(domain))
        return scenarios

    def generate_domain_scenarios(self, domain: DomainConfig, count: int) -> list[Scenario]:
        return [self.generate_scenario(domain) for _ in range(count)]

    def generate_scenario(self, domain: DomainConfig) -> Scenario:
        num_steps = self._rng.randint(8, 15)
        bug_type = self._weighted_choice(BUG_TYPE_WEIGHTS)
        bug_step, position_bucket = self._choose_bug_step(num_steps)
        agent_sequence = self._build_agent_sequence(domain, num_steps)
        scenario_id = f"{domain.slug}_{self._rng.randrange(16**8):08x}"
        shared_plan_var = f"{scenario_id}_shared_plan"

        steps: list[Step] = []
        previous_artifact = "task_brief"
        anchor_artifacts: list[str] = []
        bug_signature = self._bug_signature(bug_type, domain)

        for step_id in range(1, num_steps + 1):
            agent = agent_sequence[step_id - 1]
            produces = [f"{scenario_id}_artifact_{step_id}"]
            consumes: list[str]
            if step_id == 1:
                consumes = ["task_brief"]
                produces.append(shared_plan_var)
            else:
                consumes = [previous_artifact]
                if step_id in {3, 4, num_steps}:
                    consumes.append(shared_plan_var)
                if anchor_artifacts and step_id == num_steps - 1:
                    consumes.append(anchor_artifacts[0])

            action_type = self._action_type(agent, step_id, num_steps)
            next_agent = agent_sequence[step_id] if step_id < num_steps else None
            message_to = next_agent if next_agent and next_agent != agent else None
            input_text = self._compose_input(domain, step_id, consumes)

            if step_id == bug_step:
                output_text = self._compose_bug_output(domain, bug_type, agent, step_id, message_to, bug_signature)
                confidence = round(self._rng.uniform(0.28, 0.58), 2)
                tags = [bug_type, "bug"]
                log_signals = self._log_signals(bug_type, phase="bug")
            elif step_id == num_steps:
                output_text = self._compose_error_output(domain, bug_type, bug_signature)
                confidence = round(self._rng.uniform(0.18, 0.42), 2)
                tags = ["error"]
                log_signals = self._log_signals(bug_type, phase="error")
            elif step_id > bug_step:
                output_text = self._compose_propagation_output(domain, agent, step_id, bug_type, bug_signature)
                confidence = round(self._rng.uniform(0.62, 0.82), 2)
                tags = ["propagation"]
                log_signals = self._log_signals(bug_type, phase="propagation")
            else:
                output_text = self._compose_clean_output(domain, agent, step_id)
                confidence = round(self._rng.uniform(0.76, 0.96), 2)
                tags = ["clean"]
                log_signals = self._log_signals(bug_type, phase="clean")

            timestamp = (self._base_time + timedelta(minutes=len(steps) * 3 + self._rng.randint(0, 2))).isoformat()
            steps.append(
                Step(
                    step_id=step_id,
                    agent=agent,
                    action_type=action_type,
                    input=input_text,
                    output=output_text,
                    timestamp=timestamp,
                    produces=produces,
                    consumes=consumes,
                    message_to=message_to,
                    confidence=confidence,
                    tags=tags,
                    metadata={
                        "position_bucket": position_bucket,
                        "bug_step": bug_step,
                        "failure_surface": domain.failure_surface,
                        **log_signals,
                    },
                )
            )

            if step_id <= 2:
                anchor_artifacts.append(produces[0])
            previous_artifact = produces[0]

        return Scenario(
            scenario_id=scenario_id,
            domain=domain.name,
            domain_group=domain.group,
            interaction_pattern=domain.interaction_pattern,
            agents=list(domain.agents),
            steps=steps,
            ground_truth=GroundTruth(
                error_node_id=num_steps,
                root_cause_node_id=bug_step,
                bug_type=bug_type,
                bug_description=BUG_DESCRIPTIONS[bug_type],
                bug_position_bucket=position_bucket,
            ),
            metadata={
                "trace_length": num_steps,
                "shared_plan_var": shared_plan_var,
            },
        )

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        names = list(weights)
        return self._rng.choices(names, weights=[weights[name] for name in names], k=1)[0]

    def _choose_bug_step(self, num_steps: int) -> tuple[int, str]:
        bucket = self._weighted_choice(BUG_POSITION_WEIGHTS)
        candidates = self._candidate_positions(bucket, num_steps)
        if not candidates:
            for fallback in ("middle", "early", "late"):
                candidates = self._candidate_positions(fallback, num_steps)
                if candidates:
                    bucket = fallback
                    break
        return self._rng.choice(candidates), bucket

    def _candidate_positions(self, bucket: str, num_steps: int) -> list[int]:
        if bucket == "early":
            return [step for step in (2, 3) if step < num_steps]
        if bucket == "middle":
            return [step for step in range(4, min(7, num_steps))]
        return [step for step in range(7, num_steps)]

    def _build_agent_sequence(self, domain: DomainConfig, num_steps: int) -> list[str]:
        agents = list(domain.agents)
        templates = {
            "Sequential + Review Loop": [0, 1, 2, 1, 2, 3],
            "Hierarchical Dispatch": [0, 1, 1, 2, 1, 3],
            "Pipeline + Feedback": [0, 1, 2, 1, 2, 3],
            "Iterative Refinement": [0, 1, 2, 1, 2, 3],
            "Parallel Analysis": [0, 1, 2, 1, 2, 3],
            "Consultation Chain": [0, 1, 2, 1, 2, 3],
            "Document Pipeline": [0, 1, 2, 1, 2, 3],
            "Adaptive Loop": [0, 1, 2, 1, 2, 3],
            "Aggregation Pattern": [0, 1, 2, 1, 2, 3],
            "Incident Response": [0, 1, 2, 1, 2, 3],
        }
        indices: list[int] = []
        while len(indices) < num_steps:
            indices.extend(templates[domain.interaction_pattern])
        indices = indices[: num_steps - 1] + [len(agents) - 1]
        return [agents[index] for index in indices]

    def _action_type(self, agent: str, step_id: int, num_steps: int) -> str:
        lowered = agent.lower()
        if step_id == 1:
            if any(token in lowered for token in ("plan", "router", "scheduler", "monitor", "triager", "collector", "researcher", "searcher", "assessor")):
                return "plan"
            return "intake"
        if step_id == num_steps:
            return "finalize"
        if any(token in lowered for token in ("review", "validator", "evaluator", "verifier", "riskmanager")):
            return "review"
        if any(token in lowered for token in ("executor", "resolver", "coordinator", "reporter")):
            return "execute"
        if any(token in lowered for token in ("coder", "drafter", "writer", "contentgenerator", "advisor", "remediator")):
            return "draft"
        if any(token in lowered for token in ("analyst", "diagnoser", "specialist", "strategist", "tutor", "pharmacist")):
            return "analyze"
        return "handoff"

    def _compose_input(self, domain: DomainConfig, step_id: int, consumes: Iterable[str]) -> str:
        artifact = domain.artifacts[min(step_id - 1, len(domain.artifacts) - 1)]
        return f"Update the {artifact} for {domain.name.lower()} using inputs from {', '.join(consumes)}."

    def _compose_clean_output(self, domain: DomainConfig, agent: str, step_id: int) -> str:
        artifact = domain.artifacts[min(step_id - 1, len(domain.artifacts) - 1)]
        return (
            f"{agent} produces a consistent {artifact} that preserves the original constraints, "
            f"includes the expected context, and is ready for the next handoff."
        )

    def _compose_bug_output(
        self,
        domain: DomainConfig,
        bug_type: str,
        agent: str,
        step_id: int,
        next_agent: str | None,
        bug_signature: str,
    ) -> str:
        artifact = domain.artifacts[min(step_id - 1, len(domain.artifacts) - 1)]
        if bug_type == "logic_error":
            return (
                f"{agent} updates the {artifact} using an alternate decision path and forwards the result. "
                f"The note is concise and only lightly qualified."
            )
        if bug_type == "communication_failure":
            handoff = next_agent or "the next agent"
            return (
                f"{agent} prepares a condensed handoff for {handoff} and forwards the {artifact}. "
                f"The summary looks complete but one constraint is no longer spelled out."
            )
        if bug_type == "data_corruption":
            return (
                f"{agent} rewrites part of the {artifact} before forwarding it. "
                f"The payload remains well-formed even though two adjacent values no longer line up cleanly."
            )
        if bug_type == "missing_validation":
            return (
                f"{agent} marks the {artifact} as ready and forwards it. "
                f"The handoff is smooth, but no extra verification detail is recorded here."
            )
        return (
            f"{agent} directly updates the {artifact} and sends it onward instead of waiting for another specialist. "
            f"The action keeps the workflow moving, although the role boundary becomes a little blurry."
        )

    def _compose_propagation_output(
        self,
        domain: DomainConfig,
        agent: str,
        step_id: int,
        bug_type: str,
        bug_signature: str,
    ) -> str:
        artifact = domain.artifacts[min(step_id - 1, len(domain.artifacts) - 1)]
        return (
            f"{agent} extends the {artifact} using the current upstream assumption. "
            f"The earlier issue remains latent, so this step preserves the same flawed state without re-checking it."
        )

    def _compose_error_output(self, domain: DomainConfig, bug_type: str, bug_signature: str) -> str:
        if bug_type == "logic_error":
            symptom = "the final result follows a branch that conflicts with the original requirement"
        elif bug_type == "communication_failure":
            symptom = "the final result is missing a required constraint that earlier coordination should have preserved"
        elif bug_type == "data_corruption":
            symptom = "the final result contains inconsistent fields that no longer match the source state"
        elif bug_type == "missing_validation":
            symptom = "an unchecked artifact reaches the final stage and fails under downstream scrutiny"
        else:
            symptom = "the final result mixes responsibilities and no longer matches the intended workflow ownership"
        return f"ERROR: the final {domain.failure_surface} fails because {symptom}."

    def _bug_signature(self, bug_type: str, domain: DomainConfig) -> str:
        artifact = domain.artifacts[min(2, len(domain.artifacts) - 1)]
        signatures = {
            "logic_error": f"the decision rule on the {artifact} uses the wrong branch condition",
            "communication_failure": f"the critical constraint in the {artifact} was omitted in a message",
            "data_corruption": f"the {artifact} now contains mismatched field values",
            "missing_validation": f"the {artifact} skipped a mandatory verification gate",
            "role_confusion": f"ownership of the {artifact} shifted to the wrong specialist",
        }
        return signatures[bug_type]

    def _log_signals(self, bug_type: str, phase: str) -> dict[str, float | bool]:
        signals: dict[str, float | bool] = {
            "decision_consistency": 0.95,
            "handoff_completeness": 0.95,
            "data_integrity": 0.95,
            "validation_recorded": True,
            "role_boundary_score": 0.95,
        }
        if phase == "propagation":
            signals.update(
                {
                    "decision_consistency": 0.72,
                    "handoff_completeness": 0.72,
                    "data_integrity": 0.72,
                    "role_boundary_score": 0.72,
                }
            )
        if phase == "error":
            signals.update(
                {
                    "decision_consistency": 0.55,
                    "handoff_completeness": 0.55,
                    "data_integrity": 0.55,
                    "role_boundary_score": 0.55,
                }
            )
        if phase == "bug":
            if bug_type == "logic_error":
                signals["decision_consistency"] = 0.15
            elif bug_type == "communication_failure":
                signals["handoff_completeness"] = 0.15
            elif bug_type == "data_corruption":
                signals["data_integrity"] = 0.15
            elif bug_type == "missing_validation":
                signals["validation_recorded"] = False
            elif bug_type == "role_confusion":
                signals["role_boundary_score"] = 0.15
        return signals
