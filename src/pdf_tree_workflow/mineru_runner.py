from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from importlib.metadata import PackageNotFoundError, version

from dotenv import dotenv_values


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)


def _sdk_version() -> str | None:
    try:
        return version("mineru-open-sdk")
    except PackageNotFoundError:
        return None


def _mineru_token(env_file: Path | None = None) -> str:
    env_file = env_file or Path.cwd() / ".env"
    if not env_file.is_file():
        raise RuntimeError(f"Missing {env_file}. Copy .env.example to .env and edit it in a text editor.")
    token = (dotenv_values(env_file, interpolate=False).get("MINERU_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(f"MINERU_TOKEN is empty in {env_file}. Edit .env in a text editor.")
    return token


def run_mineru(
    *,
    run_dir: Path,
    model: str = "vlm",
    language: str = "en",
    timeout: int = 1200,
    segment_ids: set[str] | None = None,
    force: bool = False,
    env_file: Path | None = None,
) -> dict[str, Any]:
    token = _mineru_token(env_file)
    from mineru import MinerU

    run_dir = run_dir.expanduser().resolve()
    segments = json.loads((run_dir / "segments.json").read_text(encoding="utf-8"))
    selected = [
        item
        for item in segments
        if segment_ids is None or item["segment_id"] in segment_ids
    ]
    missing = (segment_ids or set()) - {item["segment_id"] for item in selected}
    if missing:
        raise ValueError(f"Unknown segment id(s): {', '.join(sorted(missing))}")

    client = MinerU(token=token)
    completed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for segment in selected:
        segment_id = segment["segment_id"]
        output = run_dir / "ocr" / segment_id
        marker = output / "result_summary.json"
        if marker.is_file() and not force:
            skipped += 1
            continue
        if output.exists() and force:
            import shutil

            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        try:
            result = client.extract(
                str(run_dir / segment["pdf"]),
                model=model,
                ocr=True,
                formula=True,
                table=True,
                language=language,
                timeout=timeout,
            )
            content_list = _json_value(getattr(result, "content_list", None))
            _write_json(output / "content_list.json", content_list or [])
            markdown = getattr(result, "markdown", None)
            if markdown is not None:
                result.save_markdown(str(output / "result.md"), with_images=True)
            summary = {
                "segment_id": segment_id,
                "model": model,
                "language": language,
                "sdk_version": _sdk_version(),
                "parameters": {
                    "ocr": True,
                    "formula": True,
                    "table": True,
                    "timeout_seconds": timeout,
                },
                "task_id": _json_value(getattr(result, "task_id", None)),
                "state": _json_value(getattr(result, "state", None)),
                "filename": _json_value(getattr(result, "filename", None)),
                "err_code": _json_value(getattr(result, "err_code", None)),
                "error": _json_value(getattr(result, "error", None)),
                "zip_url": _json_value(getattr(result, "zip_url", None)),
                "physical_pages": segment["physical_pages"],
            }
            _write_json(output / "result_summary.json", summary)
            (output / "failed.json").unlink(missing_ok=True)
            completed += 1
        except Exception as exc:
            failure = {"segment_id": segment_id, "error": str(exc).replace(token, "[credential]")}
            _write_json(output / "failed.json", failure)
            failures.append(failure)

    summary = {
        "selected": len(selected),
        "completed": completed,
        "skipped": skipped,
        "failed": failures,
    }
    _write_json(run_dir / "mineru_summary.json", summary)
    return summary
