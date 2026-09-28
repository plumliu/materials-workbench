"""Credential-isolated transport used by the semantic protocol and live checks."""

import base64
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from openai import OpenAI, OpenAIError
from PIL import Image

from chart_annotator.config import ModelConfig, load_model_config
from chart_annotator.intake import write_json


@dataclass(frozen=True)
class ModelReply:
    text: str
    model: str
    total_tokens: int | None
    finish_reason: str


class ModelCallError(RuntimeError):
    """Safe diagnostic metadata, constructed without provider bodies or headers."""

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
    def complete(self, messages: list[dict]) -> ModelReply: ...


class QwenModel:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def complete(self, messages: list[dict]) -> ModelReply:
        base_url = self._config.base_url.rstrip("/")
        base_url = base_url.removesuffix("/chat/completions")
        openrouter = "openrouter.ai" in base_url
        request = {
            "model": self._config.model,
            "messages": messages,
            "extra_body": self._config.extra_body or None,
        }
        if openrouter:
            request["max_tokens"] = self._config.max_completion_tokens
            request["extra_body"] = {
                **self._config.extra_body,
                "reasoning": {"effort": self._config.reasoning_effort},
            }
        else:
            request["reasoning_effort"] = self._config.reasoning_effort
            request["max_completion_tokens"] = self._config.max_completion_tokens
        try:
            with OpenAI(
                api_key=self._config.api_key.get_secret_value(),
                base_url=base_url,
                timeout=900,
                max_retries=0,
            ) as client:
                response = client.chat.completions.create(**request)
        except OpenAIError as error:
            # Only SDK type and numeric HTTP status; never stringify provider bodies.
            status = getattr(error, "status_code", None)
            raise ModelCallError(
                f"Qwen transport failed: {type(error).__name__}, HTTP {status if isinstance(status, int) else 'unavailable'}"
            ) from None
        text = response.choices[0].message.content if response.choices else None
        if not response.choices:
            raise ModelCallError("Qwen returned no choices")
        if response.choices[0].finish_reason != "stop":
            usage = getattr(response, "usage", None)
            raise ModelCallError(
                "Qwen response exhausted output budget or did not finish normally",
                {
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "output_budget": self._config.max_completion_tokens,
                },
            )
        if not text:
            raise ModelCallError("Qwen returned no text")
        secret = self._config.api_key.get_secret_value()
        return ModelReply(
            text.replace(secret, "[REDACTED]"),
            response.model.replace(secret, "[REDACTED]"),
            response.usage.total_tokens if response.usage else None,
            response.choices[0].finish_reason,
        )


def check_model(image: Path, expected: str, output: Path) -> dict:
    """Explicit opt-in live vision check, separate from the offline test suite."""
    with Image.open(image) as opened:
        if opened.format != "PNG":
            raise ValueError("Model check requires a rendered PNG")
        opened.verify()
    config = load_model_config()
    directory = output.resolve() / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
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
    except (RuntimeError, ValueError):
        result.update(
            status="failed", error="Model check failed; provider details suppressed"
        )
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json(directory / "result.json", result)
    return {**result, "artifact": str(directory / "result.json")}
