"""Versioned JSON protocols over the existing credential-isolated transport."""

import json
import re
import time
import unicodedata
from pathlib import Path

from pydantic import ValidationError

from chart_annotator.domain.models import (
    AxisPlan,
    ChartPlan,
    SourceAsset,
    ValidationIssue,
)
from chart_annotator.domain.workflow import (
    AxisDirection,
    AxisGrounding,
    AxisStructure,
    DatasetStructure,
    DirectionGrounding,
    Evidence,
    Grounding,
    GroundingResponse,
    Structure,
    VisualReview,
)
from chart_annotator.intake import write_json
from chart_annotator.qwen import ModelCallError, VisionLanguageModel, image_message

PROMPTS = Path(__file__).parent / "prompts/v2"

# This is the complete production Axis curriculum.  Each image teaches a
# distinct coordinate-system problem; Dataset-only examples are deliberately
# excluded from the default pipeline.
AXIS_EXAMPLE_FILES = (
    "3_2_1_10.json",  # ordinary shared X with several Y mappings
    "3_2_1_1.json",  # broken numeric X and side-specific Y mappings
    "3_2_1_7.json",  # categorical X and a shared 110/30 boundary
    "3_2_1_6.json",  # four spatially independent coordinate frames
    "3_5_1_1.json",  # logarithmic X with vertically separate Y mappings
    "3_2_7_2_3.json",  # two complete unit systems in one physical frame
)
CORE_AXIS_EXAMPLE_FILES = (
    "3_2_1_10.json",
    "3_2_1_7.json",
    "3_2_7_2_3.json",
)

DATASET_EXAMPLE_FILES = {
    "3_2_1_7.json": "dataset_3_2_1_7.png",  # categorical scatter and marker names
    "3_2_1_11.json": "dataset_3_2_1_11.png",  # vertical point groups and curves
    "3_2_1_6.json": "dataset_3_2_1_6.png",  # filled distributions: boundary only
}


def compact_axes(axes: list[AxisPlan]) -> list[dict]:
    """Expose only immutable Dataset ownership choices to the Dataset model."""
    return [
        {
            "id": axis.id,
            "name": axis.display_name,
            "x": {
                "name": axis.x_label,
                "scale": axis.x_scale,
                **(
                    {"categories": axis.categories}
                    if axis.x_scale == "categorical"
                    else {}
                ),
            },
            "y": {
                "name": axis.y_label,
                "scale": axis.y_scale,
                **(
                    {"categories": axis.categories}
                    if axis.y_scale == "categorical"
                    else {}
                ),
            },
        }
        for axis in axes
    ]


def grounding_groups(structure: AxisStructure | Structure):
    """Return ordered semantic groups and their deterministic WPD Axes."""
    if isinstance(structure, AxisStructure):
        expanded = structure.expand()
        groups = []
        offset = 0
        for group in structure.groups:
            axes = expanded[offset : offset + len(group.ys)]
            groups.append((group.x, list(group.ys), axes))
            offset += len(group.ys)
        return groups

    by_id = {axis.id: axis for axis in structure.axes}
    grouped: dict[str, list[AxisPlan]] = {}
    for axis in structure.axes:
        root = axis.shared_x_axis_id if axis.shared_x_axis_id in by_id else axis.id
        grouped.setdefault(root, []).append(axis)
    return [
        (
            AxisDirection(
                name=axes[0].x_label,
                scale=axes[0].x_scale,
                categories=axes[0].categories
                if axes[0].x_scale == "categorical"
                else [],
            ),
            [
                AxisDirection(
                    name=axis.y_label,
                    scale=axis.y_scale,
                    categories=axis.categories if axis.y_scale == "categorical" else [],
                )
                for axis in axes
            ],
            axes,
        )
        for axes in grouped.values()
    ]


def compact_grounding_structure(structure: AxisStructure | Structure) -> dict:
    """Give the model exactly one X and 1-N Ys per coordinate group."""
    return {
        "groups": [
            {
                "x": x.model_dump(mode="json"),
                "ys": [y.model_dump(mode="json") for y in ys],
            }
            for x, ys, _ in grounding_groups(structure)
        ]
    }


def expand_grounding(
    response: GroundingResponse, structure: AxisStructure | Structure
) -> Grounding:
    """Attach Python-owned Axis IDs after the grouped model response is validated."""
    groups = grounding_groups(structure)
    if len(response.groups) != len(groups):
        raise ValueError(
            "Grounding must cover each approved coordinate group exactly once"
        )

    axes = []
    for grounded, (x, ys, plans) in zip(response.groups, groups, strict=True):
        if len(grounded.ys) != len(ys):
            raise ValueError("Grounding must preserve the approved Y order and count")
        if (x.scale == "categorical") != (not grounded.x):
            raise ValueError(
                "Numeric X needs anchors; categorical X must use an empty list"
            )
        if grounded.x and len(grounded.x) < 2:
            raise ValueError("Numeric X needs two to four anchors")
        x_grounding = DirectionGrounding(anchors=grounded.x) if grounded.x else None
        for y, plan, anchors in zip(ys, plans, grounded.ys, strict=True):
            if (y.scale == "categorical") != (not anchors):
                raise ValueError(
                    "Numeric Y needs anchors; categorical Y must use an empty list"
                )
            if anchors and len(anchors) < 2:
                raise ValueError("Numeric Y needs two to four anchors")
            axes.append(
                AxisGrounding(
                    axis_id=plan.id,
                    x=x_grounding,
                    y=DirectionGrounding(anchors=anchors) if anchors else None,
                )
            )
    return Grounding(axes=axes, unresolved=response.unresolved)


def compact_grounding(
    grounding: Grounding, structure: AxisStructure | Structure
) -> GroundingResponse:
    """Remove repeated internal X copies before any model review or repair."""
    by_axis = {axis.axis_id: axis for axis in grounding.axes}
    groups = []
    for _, _, plans in grounding_groups(structure):
        grounded = [by_axis[axis.id] for axis in plans]
        x = grounded[0].x.anchors if grounded[0].x else []
        groups.append(
            {
                "x": x,
                "ys": [axis.y.anchors if axis.y else [] for axis in grounded],
            }
        )
    return GroundingResponse(groups=groups, unresolved=grounding.unresolved)


def task_evidence(task: str, evidence: Evidence) -> dict:
    """Send each model node only the text it can act on."""
    if task in ("visual-review", "failed-axis-review"):
        return {"image_size": evidence.geometry.image_size}
    if task in ("axes", "axes-repair"):
        return {
            "visible_text": list(
                dict.fromkeys(
                    text.normalized_text
                    for text in evidence.texts
                    if text.normalized_text
                )
            )
        }
    if task in ("datasets", "datasets-repair"):
        return {
            "legend_and_caption_text": list(
                dict.fromkeys(
                    text.normalized_text
                    for text in evidence.texts
                    if text.normalized_text and text.numeric_value is None
                )
            )
        }
    unit = re.compile(
        r"(?:%|percent|ksi|(?:k|m|g)?pa|°[cf]|rpm|hr|h|min|sec|s|in\.?|mm|cm)$",
        re.IGNORECASE,
    )
    return {
        "image_size": evidence.geometry.image_size,
        "numeric_text": [
            {
                "id": text.id,
                "text": text.normalized_text,
                "value": text.numeric_value,
                "bbox": text.bbox_px,
            }
            for text in evidence.texts
            if text.numeric_value is not None or unit.search(text.normalized_text)
        ],
    }


def dataset_example_output(example: dict) -> dict:
    """Project reviewed historical plans into the minimal production protocol."""
    return {
        "datasets": [
            {
                "axis": dataset["axis_id"],
                "name": dataset["display_name"],
                "kind": dataset["kind"],
            }
            for dataset in example["output"]["datasets"]
        ],
        "unresolved": [],
    }


def system_prompt() -> str:
    """Return the shared contract and handbook context used by every Qwen node."""
    return "\n\n".join(
        (PROMPTS / name).read_text(encoding="utf-8").strip()
        for name in ("system.md", "materials-context.md")
    )


def select_examples(
    evidence: Evidence, image: Path, *, task: str | None = None
) -> list[dict]:
    """Select reviewed examples for the requested protocol node."""
    text = " ".join(t.normalized_text for t in evidence.texts)
    normalized = "".join(text.casefold().split())
    if task in ("datasets", "datasets-repair"):
        selected = []
        for filename, clean_image in DATASET_EXAMPLE_FILES.items():
            example = json.loads(
                (PROMPTS / "examples" / filename).read_text(encoding="utf-8")
            )
            same_figure = evidence.source_id in example["source_ids"] or (
                example.get("figure_id")
                and re.search(
                    r"figure" + re.escape(example["figure_id"]) + r"(?!\d)",
                    normalized,
                )
            )
            if not same_figure:
                selected.append({**example, "image": clean_image})
        return selected
    examples = []
    for path in sorted((PROMPTS / "examples").glob("*.json")):
        example = json.loads(path.read_text(encoding="utf-8"))
        supported_tasks = example.get("tasks")
        if supported_tasks is not None and task not in supported_tasks:
            continue
        same_figure = evidence.source_id in example["source_ids"] or (
            example.get("figure_id")
            and re.search(
                r"figure" + re.escape(example["figure_id"]) + r"(?!\d)", normalized
            )
        )
        if same_figure and task not in ("axes", "axes-repair", "binding"):
            continue
        score = sum(
            "".join(cue.casefold().split()) in normalized
            for cue in example["selection_cues"]
        )
        if task in ("axes", "axes-repair") and "axis_output" not in example:
            continue
        if task == "binding" and "binding_output" not in example:
            continue
        examples.append((score, path.name, example))
    if task in ("axes", "axes-repair"):
        by_filename = {filename: example for _, filename, example in examples}
        curriculum = (
            AXIS_EXAMPLE_FILES if task == "axes-repair" else CORE_AXIS_EXAMPLE_FILES
        )
        return [
            by_filename[filename] for filename in curriculum if filename in by_filename
        ]

    examples.sort(key=lambda item: (-item[0], item[1]))
    # Binding keeps both reviewed coordinate examples.
    limit = None if task == "binding" else 2
    selected = examples[:limit] if limit is not None else examples
    return [example for _, _, example in selected]


def request(
    model: VisionLanguageModel,
    task: str,
    evidence: Evidence,
    image: Path,
    directory: Path,
    structure: AxisStructure | Structure | None = None,
    issues: list[ValidationIssue] | None = None,
    grounding: Grounding | None = None,
    review_context: dict | None = None,
    previous_datasets: DatasetStructure | None = None,
):
    structural = task in ("axes", "axes-repair", "datasets", "datasets-repair")
    grounding_task = task in ("binding", "repair", "failed-axis-review")
    schema = (
        AxisStructure
        if task in ("axes", "axes-repair")
        else DatasetStructure
        if task in ("datasets", "datasets-repair")
        else VisualReview
        if task == "visual-review"
        else GroundingResponse
        if grounding_task
        else Grounding
    )
    payload = {
        "evidence": task_evidence(task, evidence),
        "schema": schema.model_json_schema(),
    }
    if structure is not None:
        if task in ("datasets", "datasets-repair"):
            payload["axes"] = compact_axes(structure.axes)
        elif grounding_task:
            payload["structure"] = compact_grounding_structure(structure)
        else:
            payload["structure"] = structure.model_dump(mode="json")
    if previous_datasets is not None:
        payload["previous"] = previous_datasets.model_dump(mode="json")
    if grounding is not None:
        payload["grounding"] = (
            compact_grounding(grounding, structure).model_dump(mode="json")
            if grounding_task and structure is not None
            else grounding.model_dump(mode="json")
        )
    if review_context is not None:
        payload["review_context"] = review_context
    if issues:
        payload["issues"] = [i.model_dump(mode="json") for i in issues]
    instructions = (PROMPTS / f"{task}-task.md").read_text(encoding="utf-8")
    if task in ("datasets", "datasets-repair"):
        instructions = (
            (PROMPTS / "dataset-contract.md").read_text(encoding="utf-8")
            + "\n"
            + instructions
        )
    prompt = (
        instructions
        + "\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    system = system_prompt()
    messages = [{"role": "system", "content": system}]
    examples = (
        select_examples(evidence, image, task=task)
        if structural or task == "binding"
        else []
    )
    for example in examples:
        if task in ("axes", "axes-repair"):
            lesson = example["axis_lesson"]
            example_input = example["input"]
            example_output = example["axis_output"]
        elif task == "binding":
            lesson = example["binding_lesson"]
            example_input = example["binding_input"]
            example_output = example["binding_output"]
        elif task in ("datasets", "datasets-repair"):
            lesson = example["lesson"]
            example_input = {
                "axes": compact_axes(
                    [
                        AxisPlan.model_validate(axis)
                        for axis in example["output"]["axes"]
                    ]
                )
            }
            example_output = dataset_example_output(example)
        else:
            lesson = example["lesson"]
            example_input = example["input"]
            example_output = example["output"]
        messages.extend(
            image_message(
                PROMPTS / "examples" / example["image"],
                "REFERENCE EXAMPLE ONLY. "
                + lesson
                + "\n"
                + json.dumps(example_input, ensure_ascii=False),
            )
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    example_output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    messages.extend(image_message(image, prompt))
    write_json(
        directory / "request.json",
        {
            "protocol": "qwen/v2",
            "task": task,
            "system": system,
            "payload": payload,
            "instructions": instructions,
            "example_files": [example["image"] for example in examples],
            "image_size": evidence.geometry.image_size,
        },
    )
    started = time.perf_counter()
    try:
        reply = model.complete(messages)
    except RuntimeError as error:
        safe = (
            str(error) if isinstance(error, ModelCallError) else "Model adapter failed"
        )
        metadata = error.metadata if isinstance(error, ModelCallError) else {}
        write_json(
            directory / "failure.json", {"status": "failed", "error": safe, **metadata}
        )
        raise RuntimeError("Model request failed; see sanitized failure.json") from None
    write_json(
        directory / "response.json",
        {
            "text": reply.text,
            "model": reply.model,
            "total_tokens": reply.total_tokens,
            "finish_reason": reply.finish_reason,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )
    # Strict JSON; no eval, code execution, coordinate extraction or fuzzy repair.
    text = reply.text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    result = schema.model_validate_json(text)
    if grounding_task:
        if structure is None:
            raise ValueError("Grouped grounding requires an approved Axis structure")
        write_json(
            directory / "validated_response.json", result.model_dump(mode="json")
        )
        return expand_grounding(result, structure)
    return result


def validate_axes(
    structure: AxisStructure,
    evidence: Evidence,
    asset: SourceAsset,
    *,
    node: str = "validate_axes_plan",
) -> list[ValidationIssue]:
    problems = []
    if not structure.groups or structure.unresolved:
        problems.append("Axis structure empty or explicitly unresolved")
    if any(
        len({y.name.strip() for y in group.ys}) != len(group.ys)
        for group in structure.groups
    ):
        problems.append("Y mappings in one group must have distinct image-truth names")
    try:
        expanded = structure.expand()
        ChartPlan(source_asset=asset, axes=expanded, datasets=[])
    except (ValidationError, ValueError):
        expanded = []
        problems.append("Expanded Axis names or mappings violate ChartPlan rules")
    placeholder = re.compile(
        r"\b(?:axis|dataset|curve)[_ ]\d+\b|<[^>]+>|\.\.\.|…|图片文字不清",
        re.IGNORECASE,
    )
    unit_only = {"%", "percent", "pa", "kpa", "mpa", "gpa", "ksi"}
    direction_names = [
        direction.name
        for group in structure.groups
        for direction in (group.x, *group.ys)
    ]
    if any(placeholder.search(name) for name in direction_names):
        problems.append("Axis names must use visible image text, not placeholders")
    if any(
        "".join(unicodedata.normalize("NFKC", name).split()).casefold() in unit_only
        for name in direction_names
    ):
        problems.append("A unit alone is not a numerical quantity")
    return [
        ValidationIssue(
            code="axis_structure_invalid",
            message=p,
            node=node,
            repair="semantic",
        )
        for p in problems
    ]


def validate_datasets(
    proposal: DatasetStructure,
    axes: list[AxisPlan],
    asset: SourceAsset,
    *,
    node: str = "validate_dataset_plan",
) -> list[ValidationIssue]:
    """Validate the minimal Dataset response without requiring redundant metadata."""
    issues = []

    def issue(code: str, message: str, *ids: str) -> None:
        issues.append(
            ValidationIssue(
                code=code,
                message=message,
                node=node,
                evidence_ids=list(ids),
                repair="semantic",
            )
        )

    if not proposal.datasets:
        issue("dataset_inventory_empty", "At least one visible Dataset is required")
    if proposal.unresolved:
        issue(
            "dataset_unresolved",
            "Dataset inventory contains an unresolved visual ambiguity",
        )
    axis_ids = {axis.id for axis in axes}
    used_axes = {dataset.axis for dataset in proposal.datasets}
    unknown = used_axes - axis_ids
    if unknown:
        issue(
            "dataset_axis_unknown",
            "Dataset must reference one of the approved Axis IDs",
            *sorted(unknown),
        )
    missing = axis_ids - used_axes
    if missing:
        issue(
            "axis_without_dataset",
            "Every approved Axis needs at least one visible Dataset",
            *sorted(missing),
        )

    names = set()
    bands: dict[tuple[str, str], set[str]] = {}
    marker = re.compile(
        r"(?:\b(?:open|filled|half-filled)\s+(?:inverted\s+)?"
        r"(?:circle|square|triangle|diamond)\b|\b(?:cross|plus|star)\b)",
        re.IGNORECASE,
    )
    line = re.compile(r"\b(?:solid|dashed|dotted|dash-dot)\b", re.IGNORECASE)
    for index, dataset in enumerate(proposal.datasets):
        dataset_id = f"dataset_{index:03d}"
        name = dataset.name.strip()
        normalized = "".join(name.split()).casefold()
        if not name or re.search(
            r"\b(?:axis|dataset|curve)[_ ]\d+\b|<[^>]+>|\.\.\.|…",
            name,
            re.IGNORECASE,
        ):
            issue(
                "dataset_name_invalid",
                "Use one complete image-truth name without placeholders",
                dataset_id,
            )
        if normalized in names:
            issue(
                "duplicate_dataset_name",
                "Dataset names must be unique across the whole Figure project",
                dataset_id,
            )
        names.add(normalized)
        if re.search(r"\brun[ -]?out\b", name, re.IGNORECASE):
            issue(
                "arrow_dataset",
                "Arrowed markers remain in their marker Dataset; runout is not a Dataset",
                dataset_id,
            )
        if dataset.kind == "scatter" and not marker.search(name):
            issue(
                "marker_name_incomplete",
                "Scatter name must contain the visible fill and marker shape",
                dataset_id,
            )
        elif dataset.kind == "curve" and (
            not line.search(name) or "curve" not in name.casefold()
        ):
            issue(
                "curve_name_incomplete",
                "Curve name must contain its visible line style and curve role",
                dataset_id,
            )
        elif dataset.kind == "point_group" and not name.endswith(
            "Average Value + Spread of Value"
        ):
            issue(
                "point_group_name_invalid",
                "Vertical spread uses '<property> Average Value + Spread of Value'",
                dataset_id,
            )
        elif dataset.kind == "distribution_boundary" and (
            not line.search(name)
            or "frequency distribution boundary" not in name.casefold()
        ):
            issue(
                "distribution_name_incomplete",
                "Distribution name must contain line style and 'frequency distribution boundary'",
                dataset_id,
            )
        elif dataset.kind == "range_boundary":
            match = re.search(r"\b(upper|lower) boundary\b", name, re.IGNORECASE)
            if not line.search(name) or not match:
                issue(
                    "range_name_incomplete",
                    "Range boundary name must contain line style and upper/lower boundary",
                    dataset_id,
                )
            else:
                identity = "".join(
                    re.sub(
                        r"\b(?:upper|lower) boundary\b",
                        "boundary",
                        name.casefold(),
                    ).split()
                )
                bands.setdefault((dataset.axis, identity), set()).add(
                    match.group(1).casefold()
                )
    for (axis_id, _), roles in bands.items():
        if roles != {"upper", "lower"}:
            issue(
                "incomplete_range_band",
                "A range band needs one upper and one lower boundary",
                axis_id,
            )
    try:
        ChartPlan(source_asset=asset, axes=axes, datasets=proposal.expand())
    except (ValidationError, ValueError):
        issue(
            "dataset_structure_invalid",
            "Expanded Dataset names or Axis references violate ChartPlan rules",
        )
    return issues
