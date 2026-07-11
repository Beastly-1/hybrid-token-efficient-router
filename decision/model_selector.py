import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import (
    ALLOWED_MODELS,
    FIREWORKS_MODEL,
    FIREWORKS_MODEL_CATALOG,
    MODEL_SELECTION_MODE,
    PROJECT_ROOT,
)


_DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}
_TASK_PREFERENCE = {
    "code_generation": [
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/minimax-m3",
    ],
    "code_debug": [
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/minimax-m3",
    ],
    "math": [
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/kimi-k2p7-code",
    ],
    "logic": [
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
    ],
    "factual": [
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/minimax-m3",
    ],
    "summarization": [
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/minimax-m3",
    ],
    "ner": [
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/minimax-m3",
    ],
    "sentiment": [
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/minimax-m3",
    ],
}


@dataclass(frozen=True)
class ModelProfile:
    identifier: str
    input_cost_per_million: float
    output_cost_per_million: float
    max_difficulty: str
    task_types: frozenset[str]


# Default calibration artifact path
_DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "routing_calibration.json"

# Minimum empirical accuracy to prefer a model in calibrated mode
_DEFAULT_MIN_MODEL_ACCURACY = 0.70

# Epsilon to avoid division by zero in cost-per-success
_EPSILON = 1e-6


class ModelSelector:
    """Selects a Fireworks model using the configured cost/quality policy."""

    def __init__(self, catalog=None, allowed_models=None, selection_mode=None,
                 calibration_path=None, min_model_accuracy=None):
        catalog = FIREWORKS_MODEL_CATALOG if catalog is None else catalog
        allowed_models = ALLOWED_MODELS if allowed_models is None else allowed_models
        self._selection_mode = (selection_mode or MODEL_SELECTION_MODE).strip().lower()
        self._profiles = self._parse_catalog(catalog, allowed_models)
        self._min_model_accuracy = min_model_accuracy if min_model_accuracy is not None else _DEFAULT_MIN_MODEL_ACCURACY
        self._calibration_data = self._load_calibration(
            calibration_path if calibration_path is not None else _DEFAULT_CALIBRATION_PATH
        )
        self._selection_reason: Optional[str] = None

    @property
    def selection_reason(self) -> Optional[str]:
        """Reason for the last model selection decision."""
        return self._selection_reason

    def select(self, state):
        eligible = [profile for profile in self._profiles if self._supports(profile, state)]
        if not eligible:
            eligible = self._profiles
        if not eligible:
            raise RuntimeError(
                "No Fireworks model is configured. Set FIREWORKS_MODEL_CATALOG "
                "or FIREWORKS_MODEL."
            )

        # Try calibrated selection if data is available
        if self._calibration_data:
            calibrated = self._calibrated_select(eligible, state)
            if calibrated is not None:
                state.selected_model = calibrated.identifier
                state.estimated_remote_cost = self._estimated_cost(calibrated, state)
                return state

        # Fallback to static policy
        selected = self._select_profile(eligible, state)
        self._selection_reason = f"static_policy:{self._selection_mode}"
        state.selected_model = selected.identifier
        state.estimated_remote_cost = self._estimated_cost(selected, state)
        return state

    def _parse_catalog(self, catalog, allowed_models):
        profiles = []
        for item in catalog:
            identifier = item["id"]
            if allowed_models and identifier not in allowed_models:
                continue
            profiles.append(
                ModelProfile(
                    identifier=identifier,
                    input_cost_per_million=float(item["input_cost_per_million"]),
                    output_cost_per_million=float(item["output_cost_per_million"]),
                    max_difficulty=item.get("max_difficulty", "hard"),
                    task_types=frozenset(item.get("task_types", [])),
                )
            )
        if profiles:
            return profiles
        if allowed_models:
            return [
                ModelProfile(model, 0.0, 0.0, "hard", frozenset())
                for model in allowed_models
            ]
        return [ModelProfile(FIREWORKS_MODEL, 0.0, 0.0, "hard", frozenset())]

    def _supports(self, profile, state):
        return (
            _DIFFICULTY_RANK[profile.max_difficulty] >= _DIFFICULTY_RANK[state.difficulty]
            and (not profile.task_types or state.task_type in profile.task_types)
        )

    def _select_profile(self, eligible, state):
        preference = _TASK_PREFERENCE.get(state.task_type, [])
        preference_rank = {
            model_id: rank for rank, model_id in enumerate(preference)
        }

        def cost_rank(profile):
            return (
                self._estimated_cost(profile, state),
                preference_rank.get(profile.identifier, len(preference)),
            )

        def quality_rank(profile):
            return (
                preference_rank.get(profile.identifier, len(preference)),
                self._estimated_cost(profile, state),
            )

        def balanced_rank(profile):
            return (
                self._estimated_cost(profile, state),
                preference_rank.get(profile.identifier, len(preference)),
                _DIFFICULTY_RANK.get(profile.max_difficulty, 2),
            )

        policy = self._selection_mode
        if policy == "quality_first":
            sort_key = quality_rank
        elif policy == "balanced":
            sort_key = balanced_rank
        else:
            sort_key = cost_rank

        return min(eligible, key=sort_key)

    def calculate_actual_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float | None:
        """Return actual cost using catalog pricing, or None if unavailable."""
        profile = self._find_profile(model_id)
        if profile is None:
            return None
        if not self._valid_price(profile.input_cost_per_million):
            return None
        if not self._valid_price(profile.output_cost_per_million):
            return None
        return (
            prompt_tokens * profile.input_cost_per_million
            + completion_tokens * profile.output_cost_per_million
        ) / 1_000_000

    def _find_profile(self, model_id: str) -> ModelProfile | None:
        for profile in self._profiles:
            if profile.identifier == model_id:
                return profile
        return None

    @staticmethod
    def _valid_price(value) -> bool:
        """Return True only for finite non-negative numeric values."""
        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float)):
            return False
        if math.isnan(value) or math.isinf(value):
            return False
        return value >= 0

    # ==================================================================
    # Calibration support
    # ==================================================================

    @staticmethod
    def _load_calibration(path) -> Optional[dict]:
        """Load calibration artifact. Returns None on any failure.

        Rejects synthetic artifacts unless ALLOW_SYNTHETIC_CALIBRATION=true.
        """
        import os
        try:
            p = Path(path)
            if not p.exists():
                return None
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            if "model_profiles" not in data:
                return None
            # Reject synthetic data in production by default
            data_source = data.get("data_source", "unknown")
            if data_source == "synthetic":
                allow = os.getenv("ALLOW_SYNTHETIC_CALIBRATION", "false").lower() == "true"
                if not allow:
                    return None
            return data
        except Exception:
            return None

    def _get_model_profile_data(self, model_id: str, category: str, difficulty: str) -> Optional[dict]:
        """Lookup empirical profile with fallback hierarchy."""
        if not self._calibration_data:
            return None
        profiles = self._calibration_data.get("model_profiles", [])

        # Level 1: model + category + difficulty
        for p in profiles:
            if p["model_id"] == model_id and p["category"] == category and p["difficulty"] == difficulty:
                return p
        # Level 2: model + category (aggregate across difficulties)
        cat_profiles = [p for p in profiles if p["model_id"] == model_id and p["category"] == category]
        if cat_profiles:
            return self._aggregate_profiles(cat_profiles)
        # Level 3: overall model
        model_profiles = [p for p in profiles if p["model_id"] == model_id]
        if model_profiles:
            return self._aggregate_profiles(model_profiles)
        # Level 4: no data
        return None

    @staticmethod
    def _aggregate_profiles(profiles: list[dict]) -> dict:
        """Aggregate multiple profile entries into a single summary."""
        total_samples = sum(p.get("sample_count", 0) for p in profiles)
        if total_samples == 0:
            return None
        # Weighted averages
        acc = sum(p.get("accuracy", 0) * p.get("sample_count", 0) for p in profiles) / total_samples
        cost = sum(p.get("average_cost", 0) * p.get("sample_count", 0) for p in profiles) / total_samples
        lat = sum(p.get("average_latency_ms", 0) * p.get("sample_count", 0) for p in profiles) / total_samples
        p95 = max((p.get("p95_latency_ms", 0) for p in profiles), default=0)
        return {
            "model_id": profiles[0]["model_id"],
            "sample_count": total_samples,
            "accuracy": acc,
            "average_cost": cost,
            "average_latency_ms": lat,
            "p95_latency_ms": p95,
            "low_confidence": total_samples < 10,
        }

    def _calibrated_select(self, eligible: list, state) -> Optional["ModelProfile"]:
        """Select using empirical accuracy and cost data.

        Algorithm:
        1. For each eligible model, lookup empirical profile
        2. Prefer models meeting min accuracy
        3. Among qualifying, choose lowest expected_cost_per_success
        4. Tie-break: higher accuracy, lower p95 latency, lower raw cost
        """
        scored = []
        for profile in eligible:
            emp = self._get_model_profile_data(
                profile.identifier, state.task_type, state.difficulty
            )
            if emp is None:
                continue
            acc = emp.get("accuracy", 0.0)
            avg_cost = emp.get("average_cost", 0.0)
            est_cost = self._estimated_cost(profile, state)
            # Use empirical cost if available, else estimated
            request_cost = avg_cost if avg_cost > 0 else est_cost
            cost_per_success = request_cost / max(acc, _EPSILON)
            p95 = emp.get("p95_latency_ms", 0.0)
            scored.append((profile, acc, cost_per_success, p95, est_cost, emp.get("sample_count", 0)))

        if not scored:
            self._selection_reason = "calibrated:no_empirical_data"
            return None

        # Split into meets-accuracy and doesn't
        meets = [(p, a, cps, p95, ec, n) for p, a, cps, p95, ec, n in scored if a >= self._min_model_accuracy]

        if meets:
            # Sort by cost_per_success, then -accuracy, p95, raw cost
            meets.sort(key=lambda x: (x[2], -x[1], x[3], x[4]))
            winner = meets[0]
            self._selection_reason = (
                f"calibrated:cost_per_success={winner[2]:.6f},"
                f"accuracy={winner[1]:.3f},samples={winner[5]}"
            )
            return winner[0]
        else:
            # No model meets accuracy: pick highest accuracy, cost tie-break
            scored.sort(key=lambda x: (-x[1], x[4]))
            winner = scored[0]
            self._selection_reason = (
                f"calibrated:below_min_accuracy,best_accuracy={winner[1]:.3f},"
                f"samples={winner[5]}"
            )
            return winner[0]

    def _estimated_cost(self, profile, state):
        input_tokens = max(1, len(state.query) // 4)
        return (
            input_tokens * profile.input_cost_per_million
            + state.max_new_tokens * profile.output_cost_per_million
        ) / 1_000_000
