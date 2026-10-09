"""Bounded figure dispatch and independent SQLite checkpoints."""

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from chart_annotator.graph import build_workflow
from chart_annotator.intake import write_json
from chart_annotator.tool_workflow import PROTOCOL


def run_figure(
    source: Path | None,
    output: Path,
    checkpoint: Path | None = None,
    thread_id: str | None = None,
    *,
    model=None,
    datasets: bool = False,
    source_context: dict | None = None,
):
    thread_id = thread_id or uuid4().hex
    checkpoint = checkpoint or output.resolve() / "checkpoints" / f"{thread_id}.sqlite"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    with closing(
        sqlite3.connect(str(checkpoint), check_same_thread=False)
    ) as connection:
        saver = SqliteSaver(
            connection,
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("chart_annotator.domain.models", "SourceAsset"),
                    ("chart_annotator.domain.models", "ValidationIssue"),
                ]
            ),
        )
        graph = build_workflow(model=model, checkpointer=saver)
        previous = graph.get_state(config)
        if source is not None and previous.values:
            raise ValueError("Existing thread: use resume or a new thread ID")
        if source is None and not previous.values:
            raise ValueError("No checkpoint exists for that thread")
        if previous.values and previous.values.get("workflow_protocol") != PROTOCOL:
            raise ValueError(
                "Incompatible Figure checkpoint protocol; restart with the workbench retry flow"
            )
        if source is None and not previous.next:
            result = previous.values
        else:
            result = graph.invoke(
                {
                    "input_path": str(source),
                    "output_dir": str(output),
                    "mode": "figure",
                    "datasets_enabled": datasets,
                    "source_context": source_context,
                    "workflow_protocol": PROTOCOL,
                }
                if source is not None
                else None,
                config,
            )
    completed = result["status"] == "exported"
    if completed:
        checkpoint.unlink(missing_ok=True)
    summary = {
        "status": result["status"],
        "run_dir": result.get("run_dir"),
        "checkpoint": None if completed else str(checkpoint.resolve()),
        "thread_id": thread_id,
        "datasets_enabled": result.get("datasets_enabled", False),
        "active_stage": result.get("active_stage"),
        "model_turns": result.get("model_turns", {}),
        "grounding_submissions": result.get("grounding_submissions", 0),
        "nodes": result.get("node_status", {}),
        "issues": [
            i.model_dump(mode="json") for i in result.get("validation_issues", [])
        ],
        "artifacts": result.get("output_artifacts", []),
        "skipped_axes": result.get("skipped_axes"),
        "skipped_datasets": result.get("skipped_datasets"),
        "unresolved": (
            result.get("stage_data", {}).get("structure", {}).get("unresolved", [])
        ),
    }
    if result.get("run_dir"):
        report = Path(result["run_dir"]) / "audit/summary.json"
        write_json(report, summary)
    return summary
