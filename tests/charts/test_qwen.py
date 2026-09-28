from types import SimpleNamespace

import pytest
from openai import OpenAIError
from PIL import Image
from pydantic import SecretStr

from chart_annotator import qwen
from chart_annotator.config import ModelConfig


def test_fake_transport_configuration_and_redacted_error(monkeypatch):
    opened, sent = {}, {}

    class FakeClient:
        def __init__(self, **kwargs):
            opened.update(kwargs)
            assert kwargs["api_key"] == "synthetic-test-key"
            assert kwargs["timeout"] == 900
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(
                model="Qwen3.8-27B",
                usage=SimpleNamespace(total_tokens=100),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"axes":[]}'),
                        finish_reason="stop",
                    )
                ],
            )

    monkeypatch.setattr(qwen, "OpenAI", FakeClient)
    settings = ModelConfig(api_key=SecretStr("synthetic-test-key"))
    adapter = qwen.QwenModel(settings)
    assert (
        adapter.complete([{"role": "user", "content": "offline"}]).text == '{"axes":[]}'
    )
    assert sent["reasoning_effort"] == "xhigh"
    assert sent["extra_body"] is None
    assert sent["max_completion_tokens"] == 65536
    assert "synthetic-test-key" not in str(sent)
    settings.extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
    adapter.complete([])
    assert sent["extra_body"] == settings.extra_body

    sent.clear()
    settings.base_url = "https://openrouter.ai/api/v1/chat/completions"
    settings.extra_body = {}
    adapter.complete([])
    assert opened["base_url"] == "https://openrouter.ai/api/v1"
    assert sent["max_tokens"] == 65536
    assert sent["extra_body"] == {"reasoning": {"effort": "xhigh"}}
    assert "reasoning_effort" not in sent
    assert "max_completion_tokens" not in sent

    def fail(self, **kwargs):
        raise OpenAIError("synthetic-test-key: server diagnostics")

    monkeypatch.setattr(FakeClient, "create", fail)
    with pytest.raises(RuntimeError) as error:
        adapter.complete([])
    assert str(error.value) == "Qwen transport failed: OpenAIError, HTTP unavailable"
    assert "synthetic-test-key" not in str(error.value)


@pytest.mark.parametrize(
    "returned_model,answer,expected_status",
    [
        ("Qwen3.8-27B", "9.8", "passed"),
        ("wrong-model", "9.8", "unexpected_response"),
        ("Qwen3.8-27B", "1.2", "unexpected_response"),
    ],
)
def test_model_check_uses_fake_and_keeps_expected_out_of_request(
    tmp_path, monkeypatch, returned_model, answer, expected_status
):
    settings = ModelConfig(api_key=SecretStr("synthetic-check-key"))
    monkeypatch.setattr(qwen, "load_model_config", lambda: settings)

    def fake_complete(self, messages):
        assert "9.8" not in messages[0]["content"][0]["text"]
        assert messages[0]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        return qwen.ModelReply(answer, returned_model, 123, "stop")

    monkeypatch.setattr(qwen.QwenModel, "complete", fake_complete)
    image = tmp_path / "input.png"
    Image.new("RGB", (20, 20), "white").save(image)
    result = qwen.check_model(image, "9.8", tmp_path / "output")
    assert result["status"] == expected_status
    assert "synthetic-check-key" not in str(result)


@pytest.mark.parametrize("finish_reason,text", [("length", "partial"), ("stop", "")])
def test_incomplete_response_cannot_pass(monkeypatch, finish_reason, text):
    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=finish_reason,
                        message=SimpleNamespace(content=text),
                    )
                ]
            )

    monkeypatch.setattr(qwen, "OpenAI", FakeClient)
    with pytest.raises(RuntimeError):
        qwen.QwenModel(ModelConfig(api_key=SecretStr("synthetic-key"))).complete([])
