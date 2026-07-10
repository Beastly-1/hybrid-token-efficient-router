from dataclasses import dataclass

from config import (
    ALLOWED_MODELS,
    FIREWORKS_MODEL,
    FIREWORKS_MODEL_CATALOG,
    MODEL_SELECTION_MODE,
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


class ModelSelector:
    """Selects a Fireworks model using the configured cost/quality policy."""

    def __init__(self, catalog=None, allowed_models=None, selection_mode=None):
        catalog = FIREWORKS_MODEL_CATALOG if catalog is None else catalog
        allowed_models = ALLOWED_MODELS if allowed_models is None else allowed_models
        self._selection_mode = (selection_mode or MODEL_SELECTION_MODE).strip().lower()
        self._profiles = self._parse_catalog(catalog, allowed_models)

    def select(self, state):
        eligible = [profile for profile in self._profiles if self._supports(profile, state)]
        if not eligible:
            eligible = self._profiles
        if not eligible:
            raise RuntimeError(
                "No Fireworks model is configured. Set FIREWORKS_MODEL_CATALOG "
                "or FIREWORKS_MODEL."
            )

        selected = self._select_profile(eligible, state)
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

    def _estimated_cost(self, profile, state):
        input_tokens = max(1, len(state.query) // 4)
        return (
            input_tokens * profile.input_cost_per_million
            + state.max_new_tokens * profile.output_cost_per_million
        ) / 1_000_000
