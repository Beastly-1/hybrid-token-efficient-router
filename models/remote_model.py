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

        try:
            response = self._post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=FIREWORKS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Fireworks generation failed: {error}") from error

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Fireworks returned an empty answer.")

        state.remote_answer = content.strip()
        state.use_remote = True
        state.routing_reason = (
            f"Fireworks fallback: {state.selected_model or FIREWORKS_MODEL}."
        )
        return state

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
