"""Benchmark runner: load cases, execute inference strategies, score, log results."""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from benchmark.dataset import BenchmarkCase, load_dataset
from benchmark.scorers import ScoreResult, score_answer
from config import ALLOWED_MODELS
from state import RoutingState


@dataclass
class BenchmarkResult:
    """Per-case benchmark result record."""

    benchmark_run_id: str
    task_id: str
    category: str
    difficulty: str
    strategy: str
    scoring_method: str
    answer: Optional[str]
    score: Optional[float]
    correct: Optional[bool]
    scoring_status: str
    scoring_details: dict
    route_source: Optional[str]
    selected_model: Optional[str]
    tool_used: Optional[str]
    route_score: Optional[float]
    total_latency_ms: Optional[float]
    remote_latency_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    estimated_remote_cost: Optional[float]
    actual_remote_cost: Optional[float]
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]

    def to_dict(self) -> dict:
        return {
            "benchmark_run_id": self.benchmark_run_id,
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "strategy": self.strategy,
            "scoring_method": self.scoring_method,
            "answer": self.answer,
            "score": self.score,
            "correct": self.correct,
            "scoring_status": self.scoring_status,
            "scoring_details": self.scoring_details,
            "route_source": self.route_source,
            "selected_model": self.selected_model,
            "tool_used": self.tool_used,
            "route_score": self.route_score,
            "total_latency_ms": self.total_latency_ms,
            "remote_latency_ms": self.remote_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_remote_cost": self.estimated_remote_cost,
            "actual_remote_cost": self.actual_remote_cost,
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def parse_strategy(strategy_str: str) -> tuple[str, Optional[str]]:
    """Parse strategy string. Returns (mode, model_id_or_none)."""
    if strategy_str == "router":
        return "router", None
    if strategy_str == "local":
        return "local", None
    if strategy_str.startswith("remote:"):
        model_id = strategy_str[len("remote:"):]
        return "remote", model_id
    raise ValueError(f"Unknown strategy: '{strategy_str}'. Use 'router', 'local', or 'remote:<model-id>'")


def validate_remote_model(model_id: str, allowed_models: Optional[list] = None) -> None:
    """Raise if model_id is not in ALLOWED_MODELS."""
    models = allowed_models if allowed_models is not None else ALLOWED_MODELS
    if not models:
        raise ValueError("No ALLOWED_MODELS configured. Cannot use remote strategy.")
    if model_id not in models:
        raise ValueError(
            f"Model '{model_id}' is not in ALLOWED_MODELS. "
            f"Allowed: {models}"
        )


class BenchmarkRunner:
    """Runs benchmark cases against a selected inference strategy."""

    def __init__(
        self,
        strategy: str,
        allow_remote: bool = False,
        run_id: Optional[str] = None,
        timeout_seconds: float = 120.0,
        router_factory: Optional[Callable] = None,
        local_model_factory: Optional[Callable] = None,
        remote_model_factory: Optional[Callable] = None,
    ):
        self._mode, self._model_id = parse_strategy(strategy)
        self._allow_remote = allow_remote
        self._run_id = run_id or str(uuid.uuid4())[:8]
        self._timeout = timeout_seconds
        self._strategy_str = strategy

        # Validate remote access
        if self._mode == "remote":
            if not allow_remote:
                raise ValueError(
                    "Remote strategy requires --allow-remote flag. "
                    "Refusing to make Fireworks calls without explicit permission."
                )
            validate_remote_model(self._model_id)

        # Factories for dependency injection (testing)
        self._router_factory = router_factory
        self._local_model_factory = local_model_factory
        self._remote_model_factory = remote_model_factory

    @property
    def run_id(self) -> str:
        return self._run_id

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """Execute a single benchmark case. Never raises."""
        start = time.perf_counter()
        try:
            answer, state = self._execute(case)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Score
            score_result = score_answer(case, answer)

            # Truncate answer for logging
            answer_str = str(answer)[:500] if answer is not None else None

            return BenchmarkResult(
                benchmark_run_id=self._run_id,
                task_id=case.task_id,
                category=case.category,
                difficulty=case.difficulty,
                strategy=self._strategy_str,
                scoring_method=case.scoring_method,
                answer=answer_str,
                score=score_result.score,
                correct=score_result.correct,
                scoring_status=score_result.status,
                scoring_details=score_result.details,
                route_source=getattr(state, '_route_source', None) if state else None,
                selected_model=state.selected_model if state else None,
                tool_used=state.tool_used if state else None,
                route_score=state.route_score if state else None,
                total_latency_ms=round(elapsed_ms, 2),
                remote_latency_ms=state.remote_latency_ms if state else None,
                prompt_tokens=state.remote_prompt_tokens if state else None,
                completion_tokens=state.remote_completion_tokens if state else None,
                total_tokens=state.remote_total_tokens if state else None,
                estimated_remote_cost=state.estimated_remote_cost if state and state.estimated_remote_cost else None,
                actual_remote_cost=state.actual_remote_cost if state else None,
                success=True,
                error_type=None,
                error_message=None,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return BenchmarkResult(
                benchmark_run_id=self._run_id,
                task_id=case.task_id,
                category=case.category,
                difficulty=case.difficulty,
                strategy=self._strategy_str,
                scoring_method=case.scoring_method,
                answer=None,
                score=None,
                correct=None,
                scoring_status="error",
                scoring_details={},
                route_source=None,
                selected_model=None,
                tool_used=None,
                route_score=None,
                total_latency_ms=round(elapsed_ms, 2),
                remote_latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                estimated_remote_cost=None,
                actual_remote_cost=None,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
            )

    def run_dataset(
        self,
        cases: list[BenchmarkCase],
        output_path: Path,
        max_cases: Optional[int] = None,
    ) -> list[BenchmarkResult]:
        """Run all cases and write JSONL results."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = []
        subset = cases[:max_cases] if max_cases else cases

        with output_path.open("w", encoding="utf-8") as f:
            for case in subset:
                result = self.run_case(case)
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                results.append(result)

        return results

    def _execute(self, case: BenchmarkCase) -> tuple[Any, Optional[RoutingState]]:
        """Execute the case using the configured strategy."""
        if self._mode == "router":
            return self._execute_router(case)
        elif self._mode == "local":
            return self._execute_local(case)
        else:
            return self._execute_remote(case)

    def _execute_router(self, case: BenchmarkCase) -> tuple[Any, RoutingState]:
        """Run through the full HybridRouter."""
        if self._router_factory:
            router = self._router_factory()
        else:
            from router import HybridRouter
            router = HybridRouter()

        state = RoutingState(task_id=case.task_id, query=case.prompt)
        state = router.route(state)

        # Track route source
        if state.tool_used:
            state._route_source = "tool"
        elif state.use_remote:
            state._route_source = "fireworks"
        else:
            state._route_source = "local"

        return state.final_answer, state

    def _execute_local(self, case: BenchmarkCase) -> tuple[Any, RoutingState]:
        """Run local model directly."""
        if self._local_model_factory:
            local_model = self._local_model_factory()
        else:
            from models.local_model import LocalModel
            local_model = LocalModel()

        state = RoutingState(task_id=case.task_id, query=case.prompt)
        # Prepare minimal state for local model
        from analyzers.pre_router import PreRouterPipeline
        state = PreRouterPipeline().prepare(state)

        state = local_model.generate(state)
        state.final_answer = state.local_answer
        state._route_source = "local"
        return state.final_answer, state

    def _execute_remote(self, case: BenchmarkCase) -> tuple[Any, RoutingState]:
        """Run remote model directly."""
        if self._remote_model_factory:
            remote_model = self._remote_model_factory()
        else:
            from models.remote_model import RemoteModel
            remote_model = RemoteModel()

        state = RoutingState(task_id=case.task_id, query=case.prompt)
        # Prepare minimal state
        from analyzers.pre_router import PreRouterPipeline
        state = PreRouterPipeline().prepare(state)

        state.selected_model = self._model_id
        state = remote_model.generate(state)
        state.final_answer = state.remote_answer
        state._route_source = "fireworks"
        return state.final_answer, state
