import requests

from config import LOCAL_ENDPOINT, LOCAL_MODEL


SYSTEM_PROMPT = """
You are a routing assistant.

Rules:
- Return plain text only.
- Never use Markdown,Latex
- Be concise,don't explain if not asked explicitly.
- If the question has a single answer, output only that answer.
"""


class LocalModel:
    """
    Wrapper around the local LLM (Gemma).
    """

    def generate(self, state):
        """
        Runs the local model and updates the RoutingState.
        """

        payload = {
            "model": LOCAL_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\nUser: {state.query}\nAssistant:",
            "stream": False,
            "logprobs": True,
            "top_logprobs": 5,
        }

        try:
            response = requests.post(
                LOCAL_ENDPOINT,
                json=payload,
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            state.local_answer = data.get("response", "").strip()
            state.logprobs = data.get("logprobs", [])

            return state

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to contact local model: {e}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error in LocalModel: {e}")