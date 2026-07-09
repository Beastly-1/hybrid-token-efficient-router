from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoutingState:
    # ==========================
    # Input
    # ==========================
    task_id: str = ""
    query: str = ""

    # ==========================
    # Task Analyzer
    # ==========================
    task_type: str = "general"
    difficulty: str = "medium"
    tool_candidate: Optional[str] = None

    # ==========================
    # Local Model
    # ==========================
    local_answer: str = ""
    logprobs: list = field(default_factory=list)

    # ==========================
    # Confidence Evaluator
    # ==========================
    route_score: float = 0.0
    confidence_details: dict = field(default_factory=dict)

    # ==========================
    # Self Consistency
    # ==========================
    consistency_score: float = 0.0

    # ==========================
    # Remote Model
    # ==========================
    remote_answer: str = ""

    # ==========================
    # Final Decision
    # ==========================
    use_remote: bool = False
    final_answer: str = ""

    # ==========================
    # Debug Information
    # ==========================
    routing_reason: str = ""