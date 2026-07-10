from models.local_model import LocalModel
from state import RoutingState


class FakePipeline:
    def __init__(self, answer="ok", raises=False):
        self.answer = answer
        self.raises = raises

    def generate(self, prompt, max_new_tokens):
        if self.raises:
            raise RuntimeError("NPU execution error")
        return self.answer


def test_falls_back_to_cpu_when_npu_generation_fails(monkeypatch):
    monkeypatch.setattr("models.local_model.LOCAL_PREFERRED_DEVICE", "NPU")
    monkeypatch.setattr("models.local_model.LOCAL_FALLBACK_DEVICE", "CPU")
    calls = []

    def factory(device):
        calls.append(device)
        return FakePipeline(raises=device == "NPU")

    state = LocalModel(pipeline_factory=factory).generate(
        RoutingState(query="hello", system_prompt="system", max_new_tokens=32)
    )

    assert calls == ["NPU", "CPU"]
    assert state.local_answer == "ok"
    assert state.local_device == "CPU"
    assert state.local_fallback_used


def test_falls_back_to_cpu_when_npu_cannot_initialize(monkeypatch):
    monkeypatch.setattr("models.local_model.LOCAL_PREFERRED_DEVICE", "NPU")
    monkeypatch.setattr("models.local_model.LOCAL_FALLBACK_DEVICE", "CPU")

    def factory(device):
        if device == "NPU":
            raise RuntimeError("NPU unavailable")
        return FakePipeline(answer="CPU answer")

    state = LocalModel(pipeline_factory=factory).generate(
        RoutingState(query="hello", system_prompt="system", max_new_tokens=32)
    )

    assert state.local_answer == "CPU answer"
    assert state.local_device == "CPU"
