"""Concrete FigureGraph stages. Large evidence stays in regenerable files."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from chart_annotator import (
    calibration,
    calibration_review,
    dataset_context,
    semantic,
    snapping,
    wpd,
)
from chart_annotator.config import load_model_config
from chart_annotator.domain.models import (
    CalibrationBinding,
    ChartPlan,
    TextEvidence,
    ValidationIssue,
)
from chart_annotator.domain.workflow import (
    AxisStructure,
    Calibration,
    DatasetStructure,
    Evidence,
    Geometry,
    Grounding,
    Structure,
)
from chart_annotator.intake import write_json
from chart_annotator.qwen import QwenModel, VisionLanguageModel

if TYPE_CHECKING:
    from chart_annotator.graph import GraphState


def read(path, schema):
    return schema.model_validate_json(Path(path).read_text(encoding="utf-8"))


def attempt_dir(state: "GraphState") -> Path:
    attempts = state.get("repair_attempts", {})
    return (
        Path(state["run_dir"])
        / "attempts"
        / (
            f"e{attempts.get('evidence', 0)}_s{attempts.get('semantic', 0)}"
            + (
                f"_f{attempts['failure_visual']}"
                if attempts.get("failure_visual")
                else ""
            )
            + (f"_v{attempts['visual']}" if attempts.get("visual") else "")
            + (f"_d{attempts['dataset']}" if attempts.get("dataset") else "")
        )
    )


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


def filter_structure_axes(structure: Structure, axis_ids: set[str]) -> Structure:
    """Return a self-contained Structure view for a selected Axis set."""
    axes = [axis for axis in structure.axes if axis.id in axis_ids]
    datasets = [
        dataset for dataset in structure.datasets if dataset.axis_id in axis_ids
    ]
    return structure.model_copy(
        update={
            "axes": axes,
            "datasets": datasets,
            "unresolved": [],
        },
        deep=True,
    )


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


LOCALIZATION_FAILURE_CODES = {
    "invalid_anchor_geometry",
    "local_spine_missing",
    "local_ticks_missing",
    "local_range_missing",
    "insufficient_snapped_ticks",
    "grounded_anchor_conflict",
}


def is_localization_failure(issue: ValidationIssue, failed_axis_ids: set[str]) -> bool:
    """True when one failed Axis can plausibly be repaired by moving its anchors."""
    return (
        issue.node == "fit_and_validate_calibration"
        and issue.code in LOCALIZATION_FAILURE_CODES
        and bool(failed_axis_ids.intersection(issue.evidence_ids))
    )


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
    directory = attempt_dir(state)
    path = directory / "evidence_graph.json"
    write_json(path, evidence.model_dump(mode="json"))
    return {"evidence_graph": str(path), "status": "running"}


def model_node(task: str, model: VisionLanguageModel | None):
    def run(state):
        issue_nodes = {i.node for i in state.get("validation_issues", [])}
        current_task = task
        if task == "repair":
            current_task = (
                "axes-repair"
                if "validate_axes_plan" in issue_nodes
                else "datasets-repair"
                if "validate_dataset_plan" in issue_nodes
                else "repair"
            )
        evidence = read(
            state.get("snapped_evidence", state["evidence_graph"])
            if current_task == "repair"
            else state["evidence_graph"],
            Evidence,
        )
        structure = None
        previous_datasets = None
        if current_task == "axes-repair":
            structure = read(state["axis_plan"], AxisStructure)
        elif current_task in ("datasets", "datasets-repair"):
            structure = read(state["structure_plan"], Structure).model_copy(
                update={"datasets": [], "unresolved": []},
                deep=True,
            )
            if current_task == "datasets-repair":
                previous_datasets = read(state["dataset_plan"], DatasetStructure)
        elif current_task in ("binding", "repair"):
            structure = (
                read(state["axis_plan"], AxisStructure)
                if current_task == "binding"
                else read(state["structure_plan"], Structure)
            )
        directory = attempt_dir(state) / "model" / current_task
        image = Path(state["rendered_figure"])
        context = None
        if current_task in ("datasets", "datasets-repair"):
            image = Path(state["dataset_context_sheet"])
            context = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
        elif current_task == "repair" and state.get("calibration_review_sheet"):
            image = Path(state["calibration_review_sheet"])
            context = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
        result = semantic.request(
            model or QwenModel(load_model_config()),
            current_task,
            evidence,
            image,
            directory,
            structure,
            state.get("validation_issues")
            if current_task == "repair" or current_task.endswith("-repair")
            else None,
            read(state["grounding_plan"], Grounding)
            if current_task == "repair"
            else None,
            review_context=context,
            previous_datasets=previous_datasets,
            validation_report=state.get("structure_validation_report")
            if current_task in ("axes-repair", "datasets-repair")
            else None,
        )
        path = directory / (
            "expanded_grounding.json"
            if current_task in ("binding", "repair")
            else "validated_response.json"
        )
        write_json(path, result.model_dump(mode="json"))
        target = (
            "axis_plan"
            if current_task in ("axes", "axes-repair")
            else "dataset_plan"
            if current_task in ("datasets", "datasets-repair")
            else "grounding_plan"
        )
        return {
            target: str(path),
            "status": "running",
        }

    return run


def validate_axes_plan(state):
    axes_plan = read(state["axis_plan"], AxisStructure)
    path = attempt_dir(state) / "axis_plan.json"
    write_json(path, axes_plan.model_dump(mode="json"))
    checks = []
    issues = semantic.validate_axes(
        axes_plan,
        read(state["evidence_graph"], Evidence),
        state["source_asset"],
        checks=checks,
    )
    if issues:
        write_json(
            attempt_dir(state) / "axis_validation.json",
            [i.model_dump(mode="json") for i in issues],
        )
    result = {
        "axis_plan": str(path),
        "structure_validation_report": {
            "scope": "structural_checks_only",
            "target": "structure",
            "checks": checks,
        },
        "status": "needs_resolution" if issues else "running",
        "validation_issues": issues,
    }
    structure_path = attempt_dir(state) / "structure_plan.json"
    write_json(structure_path, combine_structure(axes_plan).model_dump(mode="json"))
    result["structure_plan"] = str(structure_path)
    return result


def validate_dataset_plan(state):
    approved = read(state["structure_plan"], Structure)
    datasets = read(state["dataset_plan"], DatasetStructure)
    structure = approved.model_copy(
        update={
            "datasets": datasets.expand(),
            "unresolved": datasets.unresolved,
        },
        deep=True,
    )
    checks = []
    issues = semantic.validate_datasets(
        datasets,
        approved.axes,
        state["source_asset"],
        node="validate_dataset_plan",
        checks=checks,
    )
    validation_report = {
        "scope": "structural_checks_only",
        "target": "previous",
        "checks": checks,
    }
    if issues:
        write_json(
            attempt_dir(state) / "dataset_validation.json",
            [i.model_dump(mode="json") for i in issues],
        )
    if issues and state.get("repair_attempts", {}).get("dataset", 0) >= 1:
        skipped = attempt_dir(state) / "skipped_datasets.json"
        write_json(
            skipped,
            {
                "reason": "Dataset validation still failed after one focused repair",
                "issues": [issue.model_dump(mode="json") for issue in issues],
                "proposal": datasets.model_dump(mode="json"),
            },
        )
        fallback = approved.model_copy(
            update={"datasets": [], "unresolved": []},
            deep=True,
        )
        rejected = attempt_dir(state) / "rejected_dataset_structure.json"
        write_json(rejected, structure.model_dump(mode="json"))
        path = attempt_dir(state) / "structure_plan_without_datasets.json"
        write_json(path, fallback.model_dump(mode="json"))
        return {
            "structure_plan": str(path),
            "structure_validation_report": validation_report,
            "status": "running",
            "validation_issues": [],
            "skipped_datasets": str(skipped),
            "output_artifacts": [
                *state.get("output_artifacts", []),
                str(rejected),
                str(skipped),
            ],
        }
    path = attempt_dir(state) / "dataset_structure_plan.json"
    write_json(path, structure.model_dump(mode="json"))
    return {
        "structure_plan": str(path),
        "structure_validation_report": validation_report,
        "status": "needs_resolution" if issues else "running",
        "validation_issues": issues,
    }


def build_dataset_context(state):
    """Render only approved calibrated Axes before asking for Dataset ownership."""
    path, _ = dataset_context.render(
        Path(state["rendered_figure"]),
        read(state["structure_plan"], Structure),
        read(state["calibration_plan"], Calibration),
        attempt_dir(state) / "dataset-context",
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


def fit_and_validate_calibration(state):
    alternative = state.get("evidence_preprocessing") == "alternative"
    directory = attempt_dir(state)
    structure = read(state["structure_plan"], Structure)
    grounding = read(state["grounding_plan"], Grounding)
    bindings, local, issues = snapping.snap_grounding(
        structure,
        grounding,
        read(state["evidence_graph"], Evidence),
        Path(state["rendered_figure"]),
        directory / "snap",
        alternative=alternative,
    )
    result, fit_issues = calibration.fit_bindings(structure, bindings, local)
    failed_during_snap = {
        evidence_id
        for issue in issues
        for evidence_id in issue.evidence_ids
        if any(axis.id == evidence_id for axis in structure.axes)
    }
    fit_issues = [
        issue
        for issue in fit_issues
        if not failed_during_snap.intersection(issue.evidence_ids)
    ]
    issues.extend(fit_issues)
    path = directory / "calibration.json"
    write_json(path, result.model_dump(mode="json"))
    if issues:
        write_json(
            directory / "validation.json", [i.model_dump(mode="json") for i in issues]
        )

    artifacts = []
    snapped_evidence = (
        str(directory / "snap/local_evidence.json")
        if (directory / "snap/local_evidence.json").exists()
        else state["evidence_graph"]
    )
    base = {
        "calibration_plan": str(path),
        "calibration_review_sheet": "",
        "failure_review_sheet": "",
        "failure_axis_ids": [],
        "failure_review_pending": False,
        "reviewed_calibration": "",
        "visual_review_status": "pending",
        "snapped_evidence": snapped_evidence,
        "validation_issues": issues,
        "status": "needs_resolution" if issues else "running",
    }
    failed_axis_ids = incomplete_axis_ids(structure, result)
    attempts = state.get("repair_attempts", {})

    # The wider deterministic search is always exhausted before asking the
    # model to move a failed direction. No review image is needed for this
    # Python-only retry.
    if (
        issues
        and all(issue.repair == "evidence" for issue in issues)
        and attempts.get("evidence", 0) < 1
    ):
        return {
            **base,
            "output_artifacts": [*state.get("output_artifacts", [])],
        }

    localization_failure = bool(failed_axis_ids) and all(
        is_localization_failure(issue, failed_axis_ids) for issue in issues
    )
    if (
        localization_failure
        and attempts.get("failure_visual", 0) < 1
        and not attempts.get("visual", 0)
    ):
        failed_grounding = grounding.model_copy(
            update={
                "axes": [
                    axis for axis in grounding.axes if axis.axis_id in failed_axis_ids
                ],
                "unresolved": [],
            },
            deep=True,
        )
        failed_bindings = bindings.model_copy(
            update={
                "axes": [
                    axis for axis in bindings.axes if axis.axis_id in failed_axis_ids
                ],
                "unresolved": [],
            },
            deep=True,
        )
        sheet, _ = calibration_review.contact_sheet(
            Path(state["rendered_figure"]),
            failed_grounding,
            failed_bindings,
            local,
            directory,
            filename="failed_axis_review.png",
            context_updates={
                "mode": "failed_axes",
                "failed_axis_ids": sorted(failed_axis_ids),
                "successful_fit_directions": [
                    {"axis_id": fit.axis_id, "direction": fit.direction}
                    for fit in result.fits
                    if fit.axis_id in failed_axis_ids
                ],
                "failure_issues": [issue.model_dump(mode="json") for issue in issues],
            },
        )
        return {
            **base,
            "failure_review_sheet": str(sheet),
            "failure_axis_ids": sorted(failed_axis_ids),
            "failure_review_pending": True,
            "output_artifacts": [
                *state.get("output_artifacts", []),
                str(sheet),
                str(sheet.with_suffix(".json")),
            ],
        }

    # Semantic protocol errors still use the existing general repair before
    # any incomplete Axis is removed.
    semantic_retry_pending = (
        issues
        and not localization_failure
        and all(issue.repair == "semantic" for issue in issues)
        and attempts.get("semantic", 0) < 1
    )
    if not semantic_retry_pending:
        structure, grounding, bindings, result, issues, skipped = retain_complete_axes(
            structure, grounding, bindings, result, issues
        )
        if skipped:
            partial = directory / "partial"
            structure_path = partial / "structure_plan.json"
            grounding_path = partial / "grounding_plan.json"
            skipped_path = partial / "skipped_axes.json"
            write_json(structure_path, structure.model_dump(mode="json"))
            write_json(grounding_path, grounding.model_dump(mode="json"))
            write_json(
                skipped_path,
                {
                    "issues": [issue.model_dump(mode="json") for issue in skipped],
                    "exported_axis_ids": [axis.id for axis in structure.axes],
                },
            )
            base.update(
                structure_plan=str(structure_path),
                grounding_plan=str(grounding_path),
                skipped_axes=str(skipped_path),
            )
            artifacts.append(str(skipped_path))
            path = partial / "calibration.json"
            write_json(path, result.model_dump(mode="json"))
            write_json(
                partial / "validation.json",
                [issue.model_dump(mode="json") for issue in issues],
            )
            base.update(
                calibration_plan=str(path),
                validation_issues=issues,
                status="needs_resolution" if issues else "running",
            )

    # With no complete Axis there is nothing meaningful to send to the final
    # visual reviewer. Preserve the diagnostic artifacts and stop closed.
    complete_axis_ids = {axis.id for axis in structure.axes} - incomplete_axis_ids(
        structure, result
    )
    if not complete_axis_ids:
        if localization_failure and attempts.get("failure_visual", 0):
            issues = [
                issue.model_copy(update={"repair": "none"}, deep=True)
                for issue in issues
            ]
            write_json(
                directory / "exhausted_validation.json",
                [issue.model_dump(mode="json") for issue in issues],
            )
            base.update(validation_issues=issues, status="needs_resolution")
        return {
            **base,
            "output_artifacts": [*state.get("output_artifacts", []), *artifacts],
        }

    sheet, _ = calibration_review.contact_sheet(
        Path(state["rendered_figure"]),
        grounding,
        bindings,
        local,
        directory,
    )
    return {
        **base,
        "calibration_plan": str(path),
        "calibration_review_sheet": str(sheet),
        "output_artifacts": [
            *state.get("output_artifacts", []),
            str(sheet),
            str(sheet.with_suffix(".json")),
            *artifacts,
        ],
        "validation_issues": issues,
        "status": "needs_resolution" if issues else "running",
    }


def failed_axis_review_node(model: VisionLanguageModel | None):
    """Ask Qwen once to relocate only directions Python could not fit."""

    def run(state):
        failed_axis_ids = set(state.get("failure_axis_ids", []))
        if not failed_axis_ids or not state.get("failure_review_sheet"):
            raise ValueError("Failed-Axis review requires an explicit failed Axis set")
        full_structure = read(state["structure_plan"], Structure)
        full_grounding = read(state["grounding_plan"], Grounding)
        failed_structure = filter_structure_axes(full_structure, failed_axis_ids)
        if {axis.id for axis in failed_structure.axes} != failed_axis_ids:
            raise ValueError("Failed-Axis review references an unknown Axis")
        failed_grounding = full_grounding.model_copy(
            update={
                "axes": [
                    axis
                    for axis in full_grounding.axes
                    if axis.axis_id in failed_axis_ids
                ],
                "unresolved": [],
            },
            deep=True,
        )
        sheet = Path(state["failure_review_sheet"])
        context = json.loads(sheet.with_suffix(".json").read_text(encoding="utf-8"))
        issues = [
            issue
            for issue in state.get("validation_issues", [])
            if failed_axis_ids.intersection(issue.evidence_ids)
        ]
        directory = attempt_dir(state) / "model/failed-axis-review"
        correction = semantic.request(
            model or QwenModel(load_model_config()),
            "failed-axis-review",
            read(state["snapped_evidence"], Evidence),
            sheet,
            directory,
            failed_structure,
            issues,
            failed_grounding,
            context,
        )
        if correction.unresolved:
            raise ValueError(
                "Failed-Axis correction must return its best concrete anchors"
            )
        corrected_by_id = {axis.axis_id: axis for axis in correction.axes}
        if (
            len(corrected_by_id) != len(correction.axes)
            or corrected_by_id.keys() != failed_axis_ids
        ):
            raise ValueError(
                "Failed-Axis correction must cover exactly the failed Axes"
            )

        fitted = read(state["calibration_plan"], Calibration)
        successful_directions = {(fit.axis_id, fit.direction) for fit in fitted.fits}
        expected = {axis.id: axis for axis in failed_structure.axes}
        merged_axes = []
        for original in full_grounding.axes:
            if original.axis_id not in failed_axis_ids:
                merged_axes.append(original)
                continue
            revised = corrected_by_id[original.axis_id]
            axis = expected[original.axis_id]
            update = {}
            for direction in ("x", "y"):
                categorical = getattr(axis, f"{direction}_scale") == "categorical"
                candidate = getattr(revised, direction)
                if categorical != (candidate is None):
                    raise ValueError(
                        "Failed-Axis correction changed the numeric directions"
                    )
                # The sheet shows successful/shared directions as context. They
                # are immutable here; only a missing fit may receive new anchors.
                update[direction] = (
                    getattr(original, direction)
                    if (original.axis_id, direction) in successful_directions
                    else candidate
                )
            merged_axes.append(original.model_copy(update=update, deep=True))

        revised_grounding = full_grounding.model_copy(
            update={"axes": merged_axes, "unresolved": []}, deep=True
        )
        path = directory / "corrected_grounding.json"
        write_json(path, revised_grounding.model_dump(mode="json"))
        response_path = directory / "validated_response.json"
        attempts = dict(state.get("repair_attempts", {}))
        attempts["failure_visual"] = attempts.get("failure_visual", 0) + 1
        return {
            "grounding_plan": str(path),
            "failure_review": str(response_path),
            "failure_review_pending": False,
            "validation_issues": [],
            "status": "running",
            "repair_attempts": attempts,
            "output_artifacts": [
                *state.get("output_artifacts", []),
                str(response_path),
                str(path),
            ],
        }

    return run


def visual_review_node(model: VisionLanguageModel | None):
    def run(state):
        sheet = Path(state["calibration_review_sheet"])
        context = json.loads(sheet.with_suffix(".json").read_text(encoding="utf-8"))
        attempts = state.get("repair_attempts", {})
        context["correction_available"] = attempts.get("visual", 0) < 1
        structure = read(state["structure_plan"], Structure)
        grounding = read(state["grounding_plan"], Grounding)
        directory = attempt_dir(state) / "model/visual-review"
        review = semantic.request(
            model or QwenModel(load_model_config()),
            "visual-review",
            read(state["snapped_evidence"], Evidence),
            sheet,
            directory,
            structure,
            state.get("validation_issues"),
            grounding,
            context,
        )
        path = directory / "validated_response.json"
        write_json(path, review.model_dump(mode="json"))
        expected = {a.id: a for a in structure.axes}
        if (
            len(review.axes) != len(expected)
            or {a.axis_id for a in review.axes} != expected.keys()
        ):
            raise ValueError("Visual review must cover every Axis exactly once")
        for item in review.axes:
            correction = item.corrected_grounding
            axis = expected[item.axis_id]
            if correction is not None and (
                any(
                    (getattr(correction, d) is None)
                    != (getattr(axis, f"{d}_scale") == "categorical")
                    for d in ("x", "y")
                )
            ):
                raise ValueError("Visual correction must preserve numeric directions")
        issues = list(state.get("validation_issues", []))
        result = {
            "visual_review": str(path),
            "output_artifacts": [*state.get("output_artifacts", []), str(path)],
            "reviewed_calibration": "",
            "visual_review_status": "blocked",
            "status": "needs_resolution",
        }
        if not review.unresolved and all(a.status == "consistent" for a in review.axes):
            if not issues:
                result.update(
                    visual_review_status="consistent",
                    status="running",
                    reviewed_calibration=state["calibration_plan"],
                )
        elif (
            not review.unresolved
            and all(a.status != "unclear" for a in review.axes)
            and attempts.get("visual", 0) < 1
        ):
            corrections = {
                a.axis_id: a.corrected_grounding
                for a in review.axes
                if a.status == "wrong"
            }
            revised = grounding.model_copy(
                update={
                    "axes": [corrections.get(a.axis_id, a) for a in grounding.axes],
                }
            )
            corrected = directory / "corrected_grounding.json"
            write_json(corrected, revised.model_dump(mode="json"))
            result.update(
                grounding_plan=str(corrected),
                status="running",
                visual_review_status="corrected",
                repair_attempts={**attempts, "visual": 1},
            )
        if result["visual_review_status"] == "blocked":
            issues.append(
                ValidationIssue(
                    code="visual_review_blocked",
                    node="qwen_visual_review",
                    message="Visual review unresolved, correction budget exhausted, or numerical gate still failing",
                )
            )
        result["validation_issues"] = issues
        return result

    return run


def compile_chart_plan(state):
    if state.get("validation_issues"):
        raise ValueError("Unresolved validation issues")
    if (
        state.get("visual_review_status") != "consistent"
        or not state.get("reviewed_calibration")
        or state["reviewed_calibration"] != state["calibration_plan"]
    ):
        raise ValueError("Current calibration needs visual approval")
    structure = read(state["structure_plan"], Structure)
    calibrated = read(state["calibration_plan"], Calibration)
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


def plan_axes_node(model: VisionLanguageModel | None):
    propose = model_node("axes", model)
    repair = model_node("axes-repair", model)

    def run(state):
        current = {**state, **propose(state)}
        current.update(validate_axes_plan(current))
        if current.get("validation_issues"):
            current["status"] = "running"
            current.update(repair(current))
            current["repair_attempts"] = {
                **current.get("repair_attempts", {}),
                "semantic": 1,
            }
            current.update(validate_axes_plan(current))
        return current

    return run


def snap_and_fit_node(model: VisionLanguageModel | None):
    relocate = failed_axis_review_node(model)
    repair = model_node("repair", model)

    def run(state):
        current = dict(state)
        for _ in range(4):
            current["status"] = "running"
            current.update(fit_and_validate_calibration(current))
            issues = current.get("validation_issues", [])
            if not issues:
                return current
            if current.get("failure_review_pending"):
                current.update(relocate(current))
                continue
            attempts = current.get("repair_attempts", {})
            if all(issue.repair == "evidence" for issue in issues) and not attempts.get(
                "evidence"
            ):
                current["evidence_preprocessing"] = "alternative"
                current["repair_attempts"] = {**attempts, "evidence": 1}
                continue
            if all(issue.repair == "semantic" for issue in issues) and not attempts.get(
                "semantic"
            ):
                current["status"] = "running"
                current.update(repair(current))
                current["repair_attempts"] = {**attempts, "semantic": 1}
                continue
            return current
        return current

    return run


def review_calibration_node(model: VisionLanguageModel | None):
    review = visual_review_node(model)

    def run(state):
        current = {**state, **review(state)}
        if current.get("visual_review_status") == "corrected":
            current["status"] = "running"
            current.update(fit_and_validate_calibration(current))
            if current.get("validation_issues"):
                return current
            current.update(review(current))
        return current

    return run


def plan_datasets_node(model: VisionLanguageModel | None):
    propose = model_node("datasets", model)
    repair = model_node("datasets-repair", model)

    def run(state):
        current = {**state, **build_dataset_context(state)}
        current.update(propose(current))
        current.update(validate_dataset_plan(current))
        if current.get("validation_issues"):
            current["status"] = "running"
            current.update(repair(current))
            current["repair_attempts"] = {
                **current.get("repair_attempts", {}),
                "dataset": 1,
            }
            current.update(validate_dataset_plan(current))
        return current

    return run


def nodes(model: VisionLanguageModel | None):
    return {
        "plan_axes": plan_axes_node(model),
        "ground_axes": model_node("binding", model),
        "snap_and_fit": snap_and_fit_node(model),
        "review_calibration": review_calibration_node(model),
        "plan_datasets": plan_datasets_node(model),
        "export_and_validate": export_and_validate,
    }
