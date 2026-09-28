"""Credentials stay outside graph state, artifacts and serializable settings."""

import json
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field, SecretStr

from chart_annotator.domain.models import Model

REPO_ROOT = Path(__file__).resolve().parents[2]


class ModelConfig(Model):
    model: str = "Qwen3.8-27B"
    base_url: str = "http://8.137.148.158:28002/v1"
    api_key: SecretStr = Field(exclude=True, repr=False)
    reasoning_effort: Literal["xhigh"] = "xhigh"
    max_completion_tokens: int = Field(default=65536, gt=0)
    # Provider extension only; health-check thinking flags are not defaults.
    extra_body: dict = Field(default_factory=dict)


def load_model_config() -> ModelConfig:
    values = dotenv_values(REPO_ROOT / ".env", interpolate=False)
    key = (values.get("CHART_ANNOTATOR_API_KEY") or "").strip()
    if not key:
        raise ValueError("CHART_ANNOTATOR_API_KEY is required")
    try:
        extra_body = json.loads(values.get("CHART_ANNOTATOR_EXTRA_BODY") or "{}")
        if not isinstance(extra_body, dict) or set(extra_body) - {
            "chat_template_kwargs"
        }:
            raise ValueError
        if "chat_template_kwargs" in extra_body and not isinstance(
            extra_body["chat_template_kwargs"], dict
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError(
            "Invalid CHART_ANNOTATOR_EXTRA_BODY provider configuration"
        ) from None
    return ModelConfig(
        model=values.get("CHART_ANNOTATOR_MODEL") or "Qwen3.8-27B",
        base_url=values.get("CHART_ANNOTATOR_BASE_URL") or "http://8.137.148.158:28002/v1",
        api_key=SecretStr(key),
        max_completion_tokens=int(
            values.get("CHART_ANNOTATOR_MAX_COMPLETION_TOKENS") or "65536"
        ),
        extra_body=extra_body,
    )
