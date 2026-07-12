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
    task_signals: dict = field(default_factory=dict)

    # ==========================
    # Pre-router controls
    # ==========================
    system_prompt: str = ""
    max_new_tokens: int = 0
    timeout_seconds: int = 0
    input_valid: bool = True
    input_issues: list[str] = field(default_factory=list)
    current_date: str = ""
    current_time: str = ""
    current_datetime: str = ""

    # ==========================
    # Local Model
    # ==========================
    local_answer: str = ""
    logprobs: list = field(default_factory=list)
    local_device: str = ""
    local_fallback_used: bool = False
    local_runtime_error: str = ""

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
    selected_model: str = ""
    estimated_remote_cost: float = 0.0
    remote_prompt_tokens: Optional[int] = None
    remote_completion_tokens: Optional[int] = None
    remote_total_tokens: Optional[int] = None
    remote_latency_ms: Optional[float] = None
    remote_usage_source: Optional[str] = None
    actual_remote_cost: Optional[float] = None
    remote_cost_source: Optional[str] = None

    # ==========================
    # Final Decision
    # ==========================
    use_remote: bool = False
    force_remote: bool = False
    final_answer: str = ""

    # ==========================
    # Debug Information
    # ==========================
    routing_reason: str = ""
    tool_used: Optional[str] = None
