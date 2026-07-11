import time

import requests

from config import (
    FIREWORKS_API_KEY,
    FIREWORKS_BASE_URL,
    FIREWORKS_MODEL,
    FIREWORKS_TIMEOUT_SECONDS,
)


class RemoteModel:
    """Fireworks chat-completions client for the final answer."""

    def __init__(self, post=requests.post):
        self._post = post

    def generate(self, state):
        if not FIREWORKS_API_KEY:
            raise RuntimeError("FIREWORKS_API_KEY is required for final generation.")

        payload = {
            "model": state.selected_model or FIREWORKS_MODEL,
            "messages": [
                {"role": "system", "content": self._system_message(state)},
                {"role": "user", "content": state.query},
            ],
            "max_tokens": state.max_new_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        }
        endpoint = f"{FIREWORKS_BASE_URL.rstrip('/')}/chat/completions"

        start = time.perf_counter()
        try:
            response = self._post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=FIREWORKS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, TypeError) as error:
            state.remote_latency_ms = round((time.perf_counter() - start) * 1000, 2)
            raise RuntimeError(f"Fireworks generation failed: {error}") from error

        state.remote_latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Fireworks returned an empty answer.")

        self._parse_usage(state, body)

        state.remote_answer = content.strip()
        state.use_remote = True
        state.routing_reason = (
            f"Fireworks fallback: {state.selected_model or FIREWORKS_MODEL}."
        )
        return state

    def _parse_usage(self, state, body: dict) -> None:
        """Extract token usage from the API response."""
        usage = body.get("usage")
        if not isinstance(usage, dict):
            state.remote_usage_source = "missing"
            return

        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")

        prompt = self._valid_token_count(prompt)
        completion = self._valid_token_count(completion)
        total = self._valid_token_count(total)

        state.remote_prompt_tokens = prompt
        state.remote_completion_tokens = completion

        if total is not None:
            state.remote_total_tokens = total
            state.remote_usage_source = "api"
        elif prompt is not None and completion is not None:
            state.remote_total_tokens = prompt + completion
            state.remote_usage_source = "derived"
        else:
            state.remote_total_tokens = None
            state.remote_usage_source = "api" if (prompt is not None or completion is not None) else "missing"

    @staticmethod
    def _valid_token_count(value) -> int | None:
        """Return value only if it is a non-negative integer."""
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        return value if value >= 0 else None

    def _system_message(self, state):
        message = (
            f"{state.system_prompt}\n\n"
            "Local pre-analysis metadata:\n"
            f"- task type: {state.task_type}\n"
            f"- difficulty: {state.difficulty}\n"
            f"- candidate tool: {state.tool_candidate or 'none'}\n"
        )
        if state.tool_used is not None:
            message += (
                f"- deterministic tool: {state.tool_used}\n"
                f"- deterministic tool result: {state.final_answer}\n"
            )
        return message + "Use this metadata as context, but answer the user's request directly."
