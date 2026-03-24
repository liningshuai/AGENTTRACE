from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class DomainConfig:
    name: str
    slug: str
    group: str
    agents: tuple[str, str, str, str]
    interaction_pattern: str
    artifacts: tuple[str, ...]
    failure_surface: str


PAPER_DOMAIN_COUNTS: Final[dict[str, int]] = {
    "Software Development": 52,
    "Customer Service": 51,
    "Research Analysis": 51,
    "Planning & Scheduling": 46,
    "Financial Trading": 50,
    "Healthcare Coordination": 60,
    "Legal Document Analysis": 60,
    "Educational Tutoring": 60,
    "Financial Advisory": 60,
    "DevOps Automation": 60,
}

BUG_TYPE_WEIGHTS: Final[dict[str, float]] = {
    "logic_error": 0.30,
    "communication_failure": 0.20,
    "data_corruption": 0.20,
    "missing_validation": 0.16,
    "role_confusion": 0.14,
}

BUG_POSITION_WEIGHTS: Final[dict[str, float]] = {
    "early": 0.60,
    "middle": 0.30,
    "late": 0.10,
}

DEFAULT_GROUP_WEIGHTS: Final[dict[str, float]] = {
    "position": 0.70,
    "structure": 0.20,
    "content": 0.05,
    "flow": 0.03,
    "confidence": 0.02,
}

PAPER_WEIGHT_GRID: Final[dict[str, tuple[float, ...]]] = {
    "position": (0.5, 0.6, 0.7, 0.8),
    "structure": (0.1, 0.15, 0.2, 0.25),
    "content": (0.03, 0.05, 0.07, 0.1),
    "flow": (0.02, 0.03, 0.05),
    "confidence": (0.01, 0.02, 0.03),
}

WIDE_WEIGHT_GRID: Final[dict[str, tuple[float, ...]]] = {
    "position": (0.35, 0.45, 0.55, 0.65),
    "structure": (0.1, 0.15, 0.2, 0.25),
    "content": (0.05, 0.1, 0.15, 0.2, 0.25),
    "flow": (0.0, 0.02, 0.05),
    "confidence": (0.0, 0.03, 0.05, 0.1),
}

ROBUST_WEIGHT_GRID: Final[dict[str, tuple[float, ...]]] = {
    "position": (0.25, 0.35, 0.45, 0.55),
    "structure": (0.05, 0.1, 0.15, 0.2),
    "content": (0.15, 0.2, 0.25, 0.3),
    "flow": (0.0, 0.05, 0.1),
    "confidence": (0.1, 0.15, 0.2, 0.25),
}

DOMAINS: Final[tuple[DomainConfig, ...]] = (
    DomainConfig(
        name="Software Development",
        slug="software_dev",
        group="Technical",
        agents=("Planner", "Coder", "Reviewer", "Executor"),
        interaction_pattern="Sequential + Review Loop",
        artifacts=(
            "requirements brief",
            "implementation plan",
            "code patch",
            "review packet",
            "test report",
            "release summary",
        ),
        failure_surface="test execution",
    ),
    DomainConfig(
        name="Customer Service",
        slug="customer_service",
        group="Service",
        agents=("Router", "Specialist", "Resolver", "Logger"),
        interaction_pattern="Hierarchical Dispatch",
        artifacts=(
            "customer issue summary",
            "order lookup result",
            "resolution draft",
            "response message",
            "ticket log",
            "closure note",
        ),
        failure_surface="customer response",
    ),
    DomainConfig(
        name="Research Analysis",
        slug="research_analysis",
        group="Knowledge",
        agents=("Searcher", "Analyzer", "Synthesizer", "Writer"),
        interaction_pattern="Pipeline + Feedback",
        artifacts=(
            "search brief",
            "paper shortlist",
            "finding matrix",
            "synthesis memo",
            "draft report",
            "final brief",
        ),
        failure_surface="research report",
    ),
    DomainConfig(
        name="Planning & Scheduling",
        slug="planning",
        group="Planning",
        agents=("Scheduler", "Optimizer", "Validator", "Notifier"),
        interaction_pattern="Iterative Refinement",
        artifacts=(
            "request intake",
            "draft schedule",
            "constraint map",
            "optimized plan",
            "validation summary",
            "notification package",
        ),
        failure_surface="schedule delivery",
    ),
    DomainConfig(
        name="Financial Trading",
        slug="trading",
        group="Business",
        agents=("Analyst", "Strategist", "RiskManager", "Executor"),
        interaction_pattern="Parallel Analysis",
        artifacts=(
            "market snapshot",
            "signal pack",
            "trade thesis",
            "risk assessment",
            "execution plan",
            "trade log",
        ),
        failure_surface="trade execution",
    ),
    DomainConfig(
        name="Healthcare Coordination",
        slug="healthcare",
        group="Service",
        agents=("Triager", "Specialist", "Pharmacist", "Coordinator"),
        interaction_pattern="Consultation Chain",
        artifacts=(
            "patient intake",
            "clinical summary",
            "treatment recommendation",
            "medication plan",
            "coordination note",
            "care handoff",
        ),
        failure_surface="care plan",
    ),
    DomainConfig(
        name="Legal Document Analysis",
        slug="legal",
        group="Knowledge",
        agents=("Researcher", "Analyst", "Drafter", "Reviewer"),
        interaction_pattern="Document Pipeline",
        artifacts=(
            "matter summary",
            "authority bundle",
            "issue outline",
            "draft memorandum",
            "review note",
            "client memo",
        ),
        failure_surface="legal memo",
    ),
    DomainConfig(
        name="Educational Tutoring",
        slug="education",
        group="Knowledge",
        agents=("Assessor", "Tutor", "ContentGenerator", "Evaluator"),
        interaction_pattern="Adaptive Loop",
        artifacts=(
            "student profile",
            "skill diagnosis",
            "lesson outline",
            "practice set",
            "feedback note",
            "progress summary",
        ),
        failure_surface="lesson feedback",
    ),
    DomainConfig(
        name="Financial Advisory",
        slug="advisory",
        group="Business",
        agents=("DataCollector", "Analyst", "Advisor", "Reporter"),
        interaction_pattern="Aggregation Pattern",
        artifacts=(
            "client brief",
            "portfolio snapshot",
            "analysis memo",
            "advice draft",
            "review summary",
            "client report",
        ),
        failure_surface="advice report",
    ),
    DomainConfig(
        name="DevOps Automation",
        slug="devops",
        group="Technical",
        agents=("Monitor", "Diagnoser", "Remediator", "Verifier"),
        interaction_pattern="Incident Response",
        artifacts=(
            "incident alert",
            "diagnostic bundle",
            "remediation plan",
            "change set",
            "verification result",
            "incident summary",
        ),
        failure_surface="remediation workflow",
    ),
)

DOMAINS_BY_NAME: Final[dict[str, DomainConfig]] = {domain.name: domain for domain in DOMAINS}
