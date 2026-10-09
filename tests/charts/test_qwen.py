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
            assert kwargs["max_retries"] == 3
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
    assert "OpenAIError, HTTP unavailable: [REDACTED]: server diagnostics" in str(
        error.value
    )
    assert "synthetic-test-key" not in str(error.value)


def test_adapter_preserves_native_tool_calls_and_reasoning_continuation(monkeypatch):
    sent = {}

    class Call:
        def model_dump(self, **kwargs):
            return {
                "id": "native_call",
                "type": "function",
                "function": {"name": "submit_axes", "arguments": "{bad-json"},
            }

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 3
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(
                model="fake",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[Call()],
                            model_extra={
                                "reasoning_details": [
                                    {"id": "r1", "data": "continuation"}
                                ]
                            },
                        ),
                    )
                ],
            )

    monkeypatch.setattr(qwen, "OpenAI", Client)
    adapter = qwen.QwenModel(ModelConfig(api_key=SecretStr("test-key")))
    reply = adapter.complete([], tools=[{"test": "schema"}], tool_choice="auto")
    assert reply.text is None
    assert reply.tool_calls[0]["function"]["arguments"] == "{bad-json"
    assert reply.assistant_fields == {
        "reasoning_details": [{"id": "r1", "data": "continuation"}]
    }
    assert sent["parallel_tool_calls"] is False
    assert sent["tool_choice"] == "auto"
    assert sent["tools"] == [{"test": "schema"}]


@pytest.mark.parametrize("status", [503, 504])
def test_provider_failure_body_survives_artifacts_and_graph(
    tmp_path, monkeypatch, status
):
    import json

    import httpx
    from openai import InternalServerError

    from chart_annotator.domain.workflow import Evidence, Geometry
    from chart_annotator.graph import guarded
    from chart_annotator.tool_workflow import PROTOCOL, model_node

    secret = "synthetic-provider-key"
    body = "<html>Upstream unavailable; diagnostic: " + secret + "</html>"

    class FailedClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            response = httpx.Response(
                status,
                text=body,
                headers={"x-request-id": "request-test-123"},
                request=httpx.Request(
                    "POST", "https://example.test/v1/chat/completions"
                ),
            )
            raise InternalServerError("Upstream failed", response=response, body=body)

    monkeypatch.setattr(qwen, "OpenAI", FailedClient)
    adapter = qwen.QwenModel(ModelConfig(api_key=SecretStr(secret)))
    image = tmp_path / "input.png"
    Image.new("RGB", (100, 100)).save(image)
    evidence = Evidence(
        source_id="test",
        geometry=Geometry(
            image_size=(100, 100), preprocessing="test", spines=[], ticks=[]
        ),
        texts=[],
    )
    (tmp_path / "evidence.json").write_text(
        evidence.model_dump_json(), encoding="utf-8"
    )
    result = guarded(
        "plan_axes",
        model_node("axes", adapter),
    )(
        {
            "workflow_protocol": PROTOCOL,
            "run_dir": str(tmp_path),
            "rendered_figure": str(image),
            "evidence_graph": str(tmp_path / "evidence.json"),
            "source_asset": __import__(
                "chart_annotator.domain.models", fromlist=["SourceAsset"]
            ).SourceAsset(source_id="test", path=str(image), kind="image"),
        }
    )
    failure = json.loads((tmp_path / "tools/axes.json").read_text(encoding="utf-8"))[
        "events"
    ][-1]
    assert failure["status_code"] == status
    assert failure["request_id"] == "request-test-123"
    assert failure["response_body"] == body.replace(secret, "[REDACTED]")
    assert failure["response_body"] in result["validation_issues"][0].message
    assert secret not in str(failure) + str(result)


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


@pytest.mark.parametrize("statuses", [[503, 504, 429, 200], [503] * 4, [401]])
def test_sdk_retries_transient_failures_with_exponential_backoff(monkeypatch, statuses):
    import httpx2
    from openai import _base_client

    calls, delays = [], []

    def respond(request):
        status = statuses[len(calls)]
        calls.append(status)
        if status != 200:
            return httpx2.Response(
                status, json={"error": {"message": "upstream detail"}}
            )
        return httpx2.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    def client(**kwargs):
        return httpx2.Client(transport=httpx2.MockTransport(respond), **kwargs)

    monkeypatch.setattr(qwen, "DefaultHttpxClient", client)
    monkeypatch.setattr(_base_client.time, "sleep", delays.append)
    monkeypatch.setattr(_base_client, "random", lambda: 0)
    model = qwen.QwenModel(
        ModelConfig(
            api_key=SecretStr("synthetic-key"), base_url="https://example.test/v1"
        )
    )
    if statuses[-1] == 200:
        reply = model.complete([])
        assert reply.text == "ok"
        assert reply.http_attempts == len(statuses)
    else:
        with pytest.raises(qwen.ModelCallError, match="upstream detail") as error:
            model.complete([])
        assert error.value.metadata["http_attempts"] == len(statuses)
        assert error.value.metadata["status_code"] == statuses[-1]
    assert calls == statuses
    assert delays == ([0.5, 1.0, 2.0] if len(statuses) == 4 else [])
