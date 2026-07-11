from pathlib import Path

from decision.model_selector import ModelSelector
from state import RoutingState

# Disable calibration in static-policy tests
_NO_CAL = Path("__nonexistent_calibration__")


def test_selects_lowest_cost_eligible_model():
    selector = ModelSelector(
        catalog=[
            {
                "id": "cheap-general",
                "input_cost_per_million": 0.1,
                "output_cost_per_million": 0.1,
                "max_difficulty": "medium",
                "task_types": [],
            },
            {
                "id": "expensive-specialist",
                "input_cost_per_million": 1.0,
                "output_cost_per_million": 1.0,
                "max_difficulty": "hard",
                "task_types": ["summarization"],
            },
        ],
        allowed_models=["cheap-general", "expensive-specialist"],
        calibration_path=_NO_CAL,
    )
    state = RoutingState(
        query="Summarize this short text.",
        task_type="summarization",
        difficulty="medium",
        max_new_tokens=96,
    )

    state = selector.select(state)

    assert state.selected_model == "cheap-general"
    assert state.estimated_remote_cost > 0


def test_quality_first_prefers_task_specialist():
    selector = ModelSelector(
        catalog=[
            {
                "id": "accounts/fireworks/models/gemma-4-31b-it",
                "input_cost_per_million": 0.1,
                "output_cost_per_million": 0.1,
                "max_difficulty": "hard",
                "task_types": [],
            },
            {
                "id": "accounts/fireworks/models/minimax-m3",
                "input_cost_per_million": 0.5,
                "output_cost_per_million": 0.5,
                "max_difficulty": "hard",
                "task_types": ["logic"],
            },
        ],
        selection_mode="quality_first",
        calibration_path=_NO_CAL,
    )
    state = RoutingState(query="Solve this.", task_type="logic", difficulty="hard")

    state = selector.select(state)

    assert state.selected_model == "accounts/fireworks/models/minimax-m3"


def test_balanced_still_prefers_cheaper_model_when_gap_is_clear():
    selector = ModelSelector(
        catalog=[
            {
                "id": "accounts/fireworks/models/gemma-4-31b-it",
                "input_cost_per_million": 0.1,
                "output_cost_per_million": 0.1,
                "max_difficulty": "hard",
                "task_types": [],
            },
            {
                "id": "accounts/fireworks/models/minimax-m3",
                "input_cost_per_million": 1.0,
                "output_cost_per_million": 1.0,
                "max_difficulty": "hard",
                "task_types": ["logic"],
            },
        ],
        selection_mode="balanced",
        calibration_path=_NO_CAL,
    )
    state = RoutingState(query="Solve this.", task_type="logic", difficulty="hard")

    state = selector.select(state)

    assert state.selected_model == "accounts/fireworks/models/gemma-4-31b-it"
