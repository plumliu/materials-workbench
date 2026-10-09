"""Concrete FigureGraph stages. Large evidence stays in regenerable files."""

from pathlib import Path

from chart_annotator import (
    dataset_context,
    wpd,
)
from chart_annotator.domain.models import (
    CalibrationBinding,
    ChartPlan,
    TextEvidence,
)
from chart_annotator.domain.workflow import (
    AxisStructure,
    Calibration,
    DatasetStructure,
    Evidence,
    Geometry,
    Structure,
)
from chart_annotator.intake import write_json


def read(path, schema):
    return schema.model_validate_json(Path(path).read_text(encoding="utf-8"))


def combine_structure(
    axes: AxisStructure, datasets: DatasetStructure | None = None
) -> Structure:
    expanded = axes.expand()
    return Structure(
        axes=expanded,
        datasets=datasets.expand() if datasets else [],
        unresolved=[*axes.unresolved, *(datasets.unresolved if datasets else [])],
    )


def retain_complete_axes(structure, grounding, bindings, calibrated, issues):
    """Keep independently calibrated Axes when siblings cannot be resolved."""
    fit_keys = {(fit.axis_id, fit.direction) for fit in calibrated.fits}
    complete = {
        axis.id
        for axis in structure.axes
        if all(
            (axis.id, direction) in fit_keys
            for direction in ("x", "y")
            if getattr(axis, f"{direction}_scale") != "categorical"
        )
    }
    all_axes = {axis.id for axis in structure.axes}
    failed = all_axes - complete
    if not complete or not failed:
        return structure, grounding, bindings, calibrated, issues, []

    skipped = [
        issue
        for issue in issues
        if failed.intersection(issue.evidence_ids)
        or issue.code in {"grounding_unresolved", "binding_unresolved"}
    ]
    blocking = [issue for issue in issues if issue not in skipped]
    kept_datasets = [
        dataset for dataset in structure.datasets if dataset.axis_id in complete
    ]
    kept_axes = [
        axis.model_copy(
            update={
                "shared_x_axis_id": axis.shared_x_axis_id
                if axis.shared_x_axis_id in complete
                else None
            },
            deep=True,
        )
        for axis in structure.axes
        if axis.id in complete
    ]
    structure = structure.model_copy(
        update={
            "axes": kept_axes,
            "datasets": kept_datasets,
        },
        deep=True,
    )
    grounding = grounding.model_copy(
        update={
            "axes": [axis for axis in grounding.axes if axis.axis_id in complete],
            "unresolved": [],
        },
        deep=True,
    )
    bindings = bindings.model_copy(
        update={
            "axes": [axis for axis in bindings.axes if axis.axis_id in complete],
            "unresolved": [],
        },
        deep=True,
    )
    calibrated = calibrated.model_copy(
        update={"fits": [fit for fit in calibrated.fits if fit.axis_id in complete]},
        deep=True,
    )
    return structure, grounding, bindings, calibrated, blocking, skipped


def incomplete_axis_ids(structure: Structure, calibrated: Calibration) -> set[str]:
    """Identify Axes missing at least one required numeric direction fit."""
    fit_keys = {(fit.axis_id, fit.direction) for fit in calibrated.fits}
    return {
        axis.id
        for axis in structure.axes
        if any(
            (axis.id, direction) not in fit_keys
            for direction in ("x", "y")
            if getattr(axis, f"{direction}_scale") != "categorical"
        )
    }


def build_evidence_graph(state):
    asset = state["source_asset"]
    observed = read(state["reconciled_text_candidates"], TextEvidence).observations
    found = Geometry(
        image_size=(asset.render_width, asset.render_height),
        preprocessing="canonical_text_only",
        spines=[],
        ticks=[],
    )
    evidence = Evidence(
        source_id=state["source_asset"].source_id, geometry=found, texts=observed
    )
    path = Path(state["run_dir"]) / "evidence/evidence_graph.json"
    write_json(path, evidence.model_dump(mode="json"))
    return {"evidence_graph": str(path), "status": "running"}


def build_dataset_context(state):
    """Render only approved calibrated Axes before asking for Dataset ownership."""
    path, _ = dataset_context.render(
        Path(state["rendered_figure"]),
        read(state["structure_plan"], Structure),
        read(state["calibration_plan"], Calibration),
        Path(state["run_dir"]) / "dataset-context",
    )
    return {
        "dataset_context_sheet": str(path),
        "status": "running",
        "output_artifacts": [
            *state.get("output_artifacts", []),
            str(path),
            str(path.with_suffix(".json")),
        ],
    }


def compile_chart_plan(state):
    if state.get("validation_issues"):
        raise ValueError("Unresolved validation issues")
    if not state.get("calibration_revision") or state.get(
        "confirmed_calibration_revision"
    ) != state.get("calibration_revision"):
        raise ValueError("Current calibration needs visual approval")
    structure = Structure.model_validate(state["stage_data"]["structure"])
    calibrated = Calibration.model_validate(state["stage_data"]["calibration"])
    if not structure.axes or incomplete_axis_ids(structure, calibrated):
        raise ValueError("Current calibration is numerically incomplete")
    axes = [a.model_copy(deep=True) for a in structure.axes]
    for axis in axes:
        axis.calibration = {
            f.direction: [
                CalibrationBinding(
                    tick_id=p.tick_id, value=p.value, text_observation_ids=[p.text_id]
                )
                for p in f.points
            ]
            for f in calibrated.fits
            if f.axis_id == axis.id
        }
    plan = ChartPlan(
        source_asset=state["source_asset"], axes=axes, datasets=structure.datasets
    )
    path = Path(state["run_dir"]) / "plan/chart_plan.json"
    write_json(path, plan.model_dump(mode="json"))
    return {"validated_chart_plan": str(path), "status": "running"}


def export_and_validate(state):
    from chart_annotator.tool_workflow import materialize

    state = {**state, **materialize(state)}
    state = {**state, **compile_chart_plan(state)}
    path = Path(state["run_dir"]) / "output/chart.pending.tar"
    wpd.export(
        path,
        read(state["validated_chart_plan"], ChartPlan),
        read(state["calibration_plan"], Calibration),
        Path(state["rendered_figure"]),
    )
    result = wpd.validate(
        path,
        read(state["validated_chart_plan"], ChartPlan),
        read(state["calibration_plan"], Calibration),
        Path(state["rendered_figure"]),
    )
    final = path.with_name("chart.tar")
    path.replace(final)
    report = Path(state["run_dir"]) / "audit/roundtrip.json"
    write_json(report, result)
    return {
        "status": "exported",
        "output_artifacts": [
            *state.get("output_artifacts", []),
            str(final),
            state["validated_chart_plan"],
            str(report),
        ],
    }
