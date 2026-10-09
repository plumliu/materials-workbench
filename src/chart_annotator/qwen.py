"""Credential-isolated transport used by the semantic protocol and live checks."""

import base64
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from openai import DefaultHttpxClient, OpenAI, OpenAIError
from PIL import Image

from chart_annotator.config import ModelConfig, load_model_config
from chart_annotator.intake import write_json


@dataclass(frozen=True)
class ModelReply:
    text: str | None
    model: str
    total_tokens: int | None
    finish_reason: str
    tool_calls: list[dict] = field(default_factory=list)
    assistant_fields: dict = field(default_factory=dict)
    http_attempts: int | None = None


class ModelCallError(RuntimeError):
    """Provider diagnostics with credentials redacted by the transport."""

    def __init__(self, message: str, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


def image_message(image: Path, prompt: str) -> list[dict]:
    """Rendered PNG only: no filenames, reference TAR or credentials in prompts."""
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        }
    ]


class VisionLanguageModel(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=False,
    ) -> ModelReply: ...


class QwenModel:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def complete(
        self,
        messages: list[dict],
        *,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=False,
    ) -> ModelReply:
        base_url = self._config.base_url.rstrip("/")
        base_url = base_url.removesuffix("/chat/completions")
        openrouter = "openrouter.ai" in base_url
        request = {
            "model": self._config.model,
            "messages": messages,
            "extra_body": self._config.extra_body or None,
        }
        if tools is not None:
            request.update(
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
        if openrouter:
            request["max_tokens"] = self._config.max_completion_tokens
            request["extra_body"] = {
                **self._config.extra_body,
                "reasoning": {"effort": self._config.reasoning_effort},
            }
        else:
            request["reasoning_effort"] = self._config.reasoning_effort
            request["max_completion_tokens"] = self._config.max_completion_tokens
        http_attempts = 0

        def count_attempt(_request):
            nonlocal http_attempts
            http_attempts += 1

        try:
            with (
                DefaultHttpxClient(
                    event_hooks={"request": [count_attempt]}
                ) as http_client,
                OpenAI(
                    api_key=self._config.api_key.get_secret_value(),
                    base_url=base_url,
                    timeout=900,
                    max_retries=3,  # SDK exponential backoff for transient transport failures.
                    http_client=http_client,
                ) as client,
            ):
                response = client.chat.completions.create(**request)
        except OpenAIError as error:
            status = getattr(error, "status_code", None)
            secret = self._config.api_key.get_secret_value()
            response = getattr(error, "response", None)
            details = str(error).replace(secret, "[REDACTED]")
            body = (
                response.text.replace(secret, "[REDACTED]")
                if response is not None
                else None
            )
            if body and body not in details:
                details += "\n" + body
            raise ModelCallError(
                f"Model transport failed: {type(error).__name__}, "
                f"HTTP {status if isinstance(status, int) else 'unavailable'}: {details}",
                {
                    "exception_type": type(error).__name__,
                    "http_attempts": http_attempts,
                    "status_code": status,
                    "request_id": (getattr(error, "request_id", None) or "").replace(
                        secret, "[REDACTED]"
                    )
                    or None,
                    "response_body": body,
                },
            ) from None
        text = response.choices[0].message.content if response.choices else None
        if not response.choices:
            raise ModelCallError("Qwen returned no choices")
        message = response.choices[0].message
        if response.choices[0].finish_reason not in {"stop", "tool_calls"} or getattr(
            message, "refusal", None
        ):
            usage = getattr(response, "usage", None)
            raise ModelCallError(
                "Qwen response exhausted output budget or did not finish normally",
                {
                    "finish_reason": response.choices[0].finish_reason,
                    "http_attempts": http_attempts,
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "output_budget": self._config.max_completion_tokens,
                },
            )
        calls = [
            call.model_dump(exclude_none=True)
            for call in getattr(message, "tool_calls", None) or []
        ]
        if not text and not calls and tools is None:
            raise ModelCallError("Qwen returned no text")
        secret = self._config.api_key.get_secret_value()
        # Preserve provider reasoning continuation fields without requesting reasoning.
        extra = getattr(message, "model_extra", None) or {}
        fields = {
            key: value
            for key, value in extra.items()
            if key in {"reasoning_content", "reasoning_details"}
        }
        safe = json.loads(
            json.dumps({"calls": calls, "fields": fields}).replace(secret, "[REDACTED]")
        )
        return ModelReply(
            text.replace(secret, "[REDACTED]") if text is not None else None,
            response.model.replace(secret, "[REDACTED]"),
            response.usage.total_tokens if response.usage else None,
            response.choices[0].finish_reason,
            safe["calls"],
            safe["fields"],
            http_attempts,
        )


def check_model(image: Path, expected: str, output: Path) -> dict:
    """Explicit opt-in live vision check, separate from the offline test suite."""
    with Image.open(image) as opened:
        if opened.format != "PNG":
            raise ValueError("Model check requires a rendered PNG")
        opened.verify()
    config = load_model_config()
    directory = output.resolve() / uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {
        "schema_version": "model-check/v1",
        "configured_model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "extra_body_configured": bool(config.extra_body),
        "max_completion_tokens": config.max_completion_tokens,
    }
    try:
        reply = QwenModel(config).complete(
            image_message(
                image,
                "Read the Figure identifier printed in the image caption. Reply with the hierarchical number only, without the word Figure. Do not return coordinates or data points.",
            )
        )
        result.update(asdict(reply))
        result["status"] = (
            "passed"
            if reply.text.strip() == expected and reply.model == config.model
            else "unexpected_response"
        )
    except ModelCallError as error:
        result.update(status="failed", error=str(error), **error.metadata)
    except (RuntimeError, ValueError):
        result.update(
            status="failed", error="Model check failed; provider details suppressed"
        )
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json(directory / "result.json", result)
    return {**result, "artifact": str(directory / "result.json")}
