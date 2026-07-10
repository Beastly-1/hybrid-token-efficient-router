import requests

from config import LOCAL_ENDPOINT, LOCAL_MODEL


SYSTEM_PROMPT = """
You are a routing assistant.

Rules:
- Return plain text only.
- Never use Markdown or LaTeX.
- Be concise.
- If the question has a single answer, output only that answer.
"""


class LocalModel:
    """
    Wrapper around the local LLM (Ollama/Gemma/Qwen).
    """

    def generate(self, state):

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

            # ---------------------------------------------------
            # DEBUG (Temporary)
            # ---------------------------------------------------

            print("\n========== OLLAMA RESPONSE ==========")
            print("Model:", data.get("model"))
            print("Answer:", data.get("response"))
            print("Number of logprobs:", len(data.get("logprobs", [])))
            print("=====================================\n")

            # ---------------------------------------------------
            # Answer
            # ---------------------------------------------------

            state.local_answer = data.get("response", "").strip()

            # ---------------------------------------------------
            # Logprobs
            # ---------------------------------------------------

            state.logprobs = data.get("logprobs", [])

            print("LOGPROBS TYPE :", type(state.logprobs))
            print("LOGPROBS VALUE:", state.logprobs)

            return state

        except requests.RequestException as e:

            raise RuntimeError(
                f"Failed to contact local model: {e}"
            )

        except Exception as e:

            raise RuntimeError(
                f"Unexpected error in LocalModel: {e}"
            )