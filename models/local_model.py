from pathlib import Path

from analyzers.sanity_checks import SanityChecker
from config import (
    LOCAL_FALLBACK_DEVICE,
    LOCAL_MODEL_CACHE_DIR,
    LOCAL_MODEL_PATH,
    LOCAL_NPU_GENERATE_HINT,
    LOCAL_PREFERRED_DEVICE,
)


class LocalModel:
    """OpenVINO GenAI local model with an NPU-first CPU fallback."""

    def __init__(self, pipeline_factory=None):
        self._pipeline_factory = pipeline_factory or self._create_pipeline
        self._pipeline = None
        self._device = ""
        self._sanity_checker = SanityChecker()

    def generate(self, state):
        prompt = self._sanity_checker.validate_prompt(
            f"{state.system_prompt}\n\nUser: {state.query}\nAssistant:"
        )

        try:
            answer = self._generate_with_active_pipeline(
                prompt,
                state.max_new_tokens,
            )
        except Exception as error:
            if self._device == LOCAL_FALLBACK_DEVICE:
                raise RuntimeError(
                    f"Local model failed on {LOCAL_FALLBACK_DEVICE}: {error}"
                ) from error

            answer = self._switch_to_cpu_and_generate(
                prompt,
                state.max_new_tokens,
                error,
            )

            state.local_fallback_used = True

        valid, reason = self._sanity_checker.validate_output(answer)

        state.local_answer = answer.strip() if valid else ""
        state.local_runtime_error = reason
        state.local_device = self._device
        state.local_fallback_used = (
            LOCAL_PREFERRED_DEVICE != LOCAL_FALLBACK_DEVICE
            and self._device == LOCAL_FALLBACK_DEVICE
        )
        state.logprobs = []

        return state

    def _generate_with_active_pipeline(self, prompt, max_new_tokens):
        if self._pipeline is None:
            self._load_preferred_pipeline()

        return str(
            self._pipeline.generate(
                prompt,
                max_new_tokens=max_new_tokens,
            )
        )

    def _load_preferred_pipeline(self):
        try:
            self._pipeline = self._pipeline_factory(LOCAL_PREFERRED_DEVICE)
            self._device = LOCAL_PREFERRED_DEVICE
        except Exception:
            self._pipeline = self._pipeline_factory(LOCAL_FALLBACK_DEVICE)
            self._device = LOCAL_FALLBACK_DEVICE

    def _switch_to_cpu_and_generate(self, prompt, max_new_tokens, npu_error):
        try:
            self._pipeline = self._pipeline_factory(LOCAL_FALLBACK_DEVICE)
            self._device = LOCAL_FALLBACK_DEVICE

            return str(
                self._pipeline.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                )
            )

        except Exception as cpu_error:
            raise RuntimeError(
                f"NPU inference failed ({npu_error}); CPU fallback also failed ({cpu_error})."
            ) from cpu_error

    def _create_pipeline(self, device):
        model_path = Path(LOCAL_MODEL_PATH)

        if not model_path.is_dir():
            raise FileNotFoundError(
                f"OpenVINO model directory does not exist: {model_path}. "
                "Export or download a model, then set LOCAL_MODEL_PATH."
            )

        try:
            import openvino_genai as ov_genai
        except ImportError as error:
            raise RuntimeError(
                "OpenVINO GenAI is not installed. Run `pip install -r requirements.txt`."
            ) from error

        cache_dir = Path(LOCAL_MODEL_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        config = {"CACHE_DIR": str(cache_dir)}

        if device == "NPU":
            config["GENERATE_HINT"] = LOCAL_NPU_GENERATE_HINT

        return ov_genai.LLMPipeline(
            str(model_path),
            device,
            **config,
        )