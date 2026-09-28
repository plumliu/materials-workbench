"""Bounded intake and figure workflows with evidence-gated export."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from chart_annotator import figure, intake, stages, text_evidence
from chart_annotator.domain.models import (
    SourceAsset,
    TextEvidence,
    ValidationIssue,
)
from chart_annotator.qwen import VisionLanguageModel

Status = Literal[
    "running",
    "render_complete",
    "text_evidence_complete",
    "needs_resolution",
    "failed",
    "exported",
]
NodeStatus = Literal["completed", "failed"]


class GraphState(TypedDict, total=False):
    input_path: str
    output_dir: str
    source_context: dict
    run_dir: str
    mode: Literal["figure", "render", "text-evidence"]
    status: Status
    node_status: dict[str, NodeStatus]
    source_asset: SourceAsset
    rendered_figure: str
    pdf_text_observations: str
    ocr_text_observations: str
    reconciled_text_candidates: str
    evidence_graph: str
    datasets_enabled: bool
    axis_plan: str
    dataset_plan: str
    dataset_context_sheet: str
    structure_plan: str
    grounding_plan: str
    snapped_evidence: str
    validated_chart_plan: str
    calibration_plan: str
    calibration_review_sheet: str
    failure_review_sheet: str
    failure_axis_ids: list[str]
    failure_review_pending: bool
    failure_review: str
    visual_review: str
    visual_review_status: Literal["pending", "consistent", "corrected", "blocked"]
    reviewed_calibration: str
    validation_issues: list[ValidationIssue]
    repair_attempts: dict[str, int]
    evidence_preprocessing: Literal["original", "alternative"]
    output_artifacts: list[str]
    skipped_axes: str
    skipped_datasets: str


FIGURE_NODES = (
    "prepare_figure",
    "plan_axes",
    "ground_axes",
    "snap_and_fit",
    "review_calibration",
    "plan_datasets",
    "export_and_validate",
)
Node = Callable[[GraphState], GraphState]


def ingest_figure(state: GraphState) -> GraphState:
    directory, asset = figure.ingest_figure(
        Path(state["input_path"]), Path(state["output_dir"]), context=state.get("source_context")
    )
    return {
        "run_dir": str(directory),
        "source_asset": asset,
        "status": "running",
        "output_artifacts": [str(directory / "source.json")],
    }


def render_figure(state: GraphState) -> GraphState:
    directory = Path(state["run_dir"])
    asset, image = figure.render_figure(directory, state["source_asset"])
    return {
        "source_asset": asset,
        "rendered_figure": str(image),
        "status": "render_complete" if state.get("mode") == "render" else "running",
        "output_artifacts": [
            *state.get("output_artifacts", []),
            str(image),
            str(directory / "render/transform.json"),
        ],
    }


def guarded(name: str, node: Node) -> Node:
    def run(state: GraphState) -> GraphState:
        try:
            result = node(state)
        except Exception:  # noqa: BLE001 - sanitized graph error boundary
            result = {
                "status": "failed",
                "validation_issues": [
                    *state.get("validation_issues", []),
                    ValidationIssue(
                        code="node_failed",
                        message=f"{name} failed; inspect local input and stage artifacts",
                        node=name,
                    ),
                ],
            }
        status: NodeStatus = "completed"
        if result.get("status") == "failed":
            status = "failed"
        result["node_status"] = {**state.get("node_status", {}), name: status}
        return result

    return run


def extract_pdf_text(state: GraphState) -> GraphState:
    directory = Path(state["run_dir"])
    text_evidence.extract_pdf_text(directory, state["source_asset"])
    return {
        "pdf_text_observations": str(directory / "evidence/pdf_text.json"),
        "status": "running",
    }


def run_general_ocr(state: GraphState) -> GraphState:
    directory = Path(state["run_dir"])
    path = directory / "evidence/ocr_general.json"
    try:
        text_evidence.run_general_ocr(
            directory, state["source_asset"], Path(state["rendered_figure"])
        )
    except Exception:  # noqa: BLE001 - sanitized local OCR error boundary
        if not path.exists():
            intake.write_json(
                path,
                TextEvidence(
                    source_id=state["source_asset"].source_id,
                    status="failed",
                    reason="ocr_failed",
                ).model_dump(mode="json"),
            )
        return {
            "ocr_text_observations": str(path),
            "status": "failed",
            "validation_issues": [
                ValidationIssue(
                    code="ocr_failed",
                    message="General OCR failed; PDF observations retained",
                    node="run_general_ocr",
                )
            ],
        }
    return {"ocr_text_observations": str(path), "status": "running"}


def reconcile_text_evidence(state: GraphState) -> GraphState:
    pdf = TextEvidence.model_validate_json(
        Path(state["pdf_text_observations"]).read_text(encoding="utf-8")
    )
    ocr = TextEvidence.model_validate_json(
        Path(state["ocr_text_observations"]).read_text(encoding="utf-8")
    )
    if (
        pdf.source_id != ocr.source_id
        or pdf.source_id != state["source_asset"].source_id
    ):
        raise ValueError("Evidence source mismatch")
    matches = text_evidence.reconcile_text(pdf.observations, ocr.observations)
    issues = [
        ValidationIssue(
            code=match.status,
            message="Text observations require resolution; no candidate selected",
            node="reconcile_text_evidence",
            evidence_ids=match.pdf_observation_ids + match.ocr_observation_ids,
            repair="evidence",
        )
        for match in matches
        if match.status in ("text_conflict", "position_conflict", "multiple_candidates")
    ]
    if not pdf.observations and not ocr.observations:
        issues.append(
            ValidationIssue(
                code="no_readable_text",
                message="Neither source produced readable text",
                node="reconcile_text_evidence",
            )
        )
    directory = Path(state["run_dir"])
    path = directory / "evidence/text_reconciliation.json"
    by_id = {item.id: item for item in [*pdf.observations, *ocr.observations]}
    canonical = []
    for match in matches:
        identifiers = [*match.pdf_observation_ids, *match.ocr_observation_ids]
        if match.status == "agreed" and match.pdf_observation_ids:
            identifiers = match.pdf_observation_ids[:1]
        canonical.extend(by_id[identifier] for identifier in identifiers)
    canonical_path = directory / "evidence/canonical_text.json"
    canonical_evidence = TextEvidence(
        source_id=pdf.source_id,
        status="available" if canonical else "unavailable",
        reason=None if canonical else "no_readable_text",
        observations=canonical,
    )
    intake.write_json(canonical_path, canonical_evidence.model_dump(mode="json"))
    intake.write_json(
        path,
        {
            "schema_version": "text-reconciliation/v1",
            "source_id": pdf.source_id,
            "pdf_status": pdf.status,
            "ocr_status": ocr.status,
            "matches": [match.model_dump(mode="json") for match in matches],
            "canonical_observation_ids": [item.id for item in canonical],
            "issues": [issue.model_dump(mode="json") for issue in issues],
        },
    )
    return {
        "reconciled_text_candidates": str(canonical_path),
        "validation_issues": issues,
        "status": ("needs_resolution" if issues else "text_evidence_complete")
        if state.get("mode") == "text-evidence"
        else "running",
        "output_artifacts": [
            *state.get("output_artifacts", []),
            state["pdf_text_observations"],
            state["ocr_text_observations"],
            str(path),
            str(canonical_path),
        ],
    }


def prepare_figure(state: GraphState) -> GraphState:
    """Ingest, render and build canonical evidence as one resumable local stage."""
    current = {**state, **ingest_figure(state)}
    current.update(render_figure(current))
    if state.get("mode") == "render":
        return current
    current.update(extract_pdf_text(current))
    current.update(run_general_ocr(current))
    if current.get("status") == "failed":
        return current
    current.update(reconcile_text_evidence(current))
    if state.get("mode") == "text-evidence":
        return current
    current.update(stages.build_evidence_graph(current))
    return current


def build_workflow(
    *,
    offline_nodes: dict[str, Node] | None = None,
    model: VisionLanguageModel | None = None,
    checkpointer=None,
):
    """Credentials remain in the model closure, outside serializable graph state."""
    graph = StateGraph(GraphState)
    nodes = stages.nodes(model)
    nodes["prepare_figure"] = prepare_figure
    if offline_nodes:
        unknown = offline_nodes.keys() - nodes.keys()
        if unknown:
            raise ValueError("Unknown offline node names")
        nodes.update(offline_nodes)
    for name, node in nodes.items():
        graph.add_node(name, guarded(name, node))
    graph.add_edge(START, "prepare_figure")
    for name, following in zip(FIGURE_NODES, (*FIGURE_NODES[1:], END), strict=True):
        if name == "review_calibration":
            graph.add_conditional_edges(
                name,
                lambda state: (
                    "plan_datasets"
                    if state.get("status") == "running"
                    and state.get("datasets_enabled")
                    else "export_and_validate"
                    if state.get("status") == "running"
                    else END
                ),
                ["plan_datasets", "export_and_validate", END],
            )
        elif name == "plan_datasets":
            graph.add_conditional_edges(
                name,
                lambda state: (
                    "export_and_validate" if state.get("status") == "running" else END
                ),
                ["export_and_validate", END],
            )
        elif name == "export_and_validate":
            graph.add_edge(name, END)
        else:
            graph.add_conditional_edges(
                name,
                lambda state, destination=following: (
                    destination if state.get("status") == "running" else END
                ),
                [following, END],
            )
    return graph.compile(checkpointer=checkpointer)
