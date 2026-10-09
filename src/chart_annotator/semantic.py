"""Versioned JSON protocols over the existing credential-isolated transport."""

import json
import re
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
)

PROMPTS = Path(__file__).parent / "prompts/v2"

# Use the complete Axis curriculum on the first pass: structural validation
# cannot detect a plausible but semantically wrong scale or mapping.
AXIS_EXAMPLE_FILES = (
    "3_2_1_1.json",  # broken numeric X and side-specific Y mappings
    "3_2_1_7.json",  # categorical X and a shared 110/30 boundary
    "3_2_1_11.json",  # same-unit left/right scales with different origins
    "3_2_1_6.json",  # four spatially independent coordinate frames
    "3_5_1_1.json",  # logarithmic X with vertically separate Y mappings
    "3_2_7_2_3.json",  # two complete unit systems in one physical frame
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
    if task not in {"axes", "grounding", "datasets"}:
        raise ValueError(f"Unknown model stage: {task}")
    if task == "axes":
        return {
            "visible_text": list(
                dict.fromkeys(
                    text.normalized_text
                    for text in evidence.texts
                    if text.normalized_text
                )
            )
        }
    if task == "datasets":
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


def select_examples(evidence: Evidence, *, task: str) -> list[dict]:
    """Select reviewed examples for the requested protocol node."""
    if task not in {"axes", "grounding", "datasets"}:
        raise ValueError(f"Unknown model stage: {task}")
    text = " ".join(t.normalized_text for t in evidence.texts)
    normalized = "".join(text.casefold().split())
    if task == "datasets":
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
        if task not in example["tasks"]:
            continue
        score = sum(
            "".join(cue.casefold().split()) in normalized
            for cue in example["selection_cues"]
        )
        examples.append((score, path.name, example))
    if task == "axes":
        by_filename = {filename: example for _, filename, example in examples}
        return [
            by_filename[filename]
            for filename in AXIS_EXAMPLE_FILES
            if filename in by_filename
        ]

    examples.sort(key=lambda item: (-item[0], item[1]))
    # Grounding keeps both reviewed coordinate examples.
    return [example for _, _, example in examples]


def _record_check(checks, rule, paths, passed, requirement):
    if checks is not None:
        checks.append(
            {
                "rule": rule,
                "paths": paths,
                "status": "passed" if passed else "failed",
                "requirement": requirement,
            }
        )
    return passed


def validate_axes(
    structure: AxisStructure,
    evidence: Evidence,
    asset: SourceAsset,
    *,
    node: str = "validate_axes_plan",
    checks: list[dict] | None = None,
) -> list[ValidationIssue]:
    issues = []

    def check(rule, paths, passed, message):
        if not _record_check(checks, rule, paths, passed, message):
            issues.append(
                ValidationIssue(
                    code="axis_structure_invalid",
                    message=f"{', '.join(paths)}: {message}",
                    node=node,
                    repair="semantic",
                )
            )

    check(
        "axis_inventory",
        ["groups"],
        bool(structure.groups),
        "At least one Axis group is required",
    )
    placeholder = re.compile(
        r"\b(?:axis|dataset|curve)[_ ]\d+\b|<[^>]+>|\.\.\.|…|图片文字不清",
        re.IGNORECASE,
    )
    unit_only = {"%", "percent", "pa", "kpa", "mpa", "gpa", "ksi"}
    pair_names = {}
    for gi, group in enumerate(structure.groups):
        root = f"groups[{gi}]"
        names = {}
        for yi, y in enumerate(group.ys):
            names.setdefault(y.name.strip(), []).append(f"{root}.ys[{yi}].name")
            pair_names.setdefault(f"{group.x.name} - {y.name}", []).append(
                f"{root}.ys[{yi}]"
            )
            check(
                "numeric_direction_required",
                [f"{root}.x.scale", f"{root}.ys[{yi}].scale"],
                not (group.x.scale == y.scale == "categorical"),
                "An X-Y pair cannot have two categorical directions",
            )
        for paths in names.values():
            check(
                "distinct_y_names",
                paths,
                len(paths) == 1,
                "Y mappings in one group must have distinct names",
            )
        directions = [
            (f"{root}.x", group.x),
            *((f"{root}.ys[{yi}]", y) for yi, y in enumerate(group.ys)),
        ]
        for path, direction in directions:
            check(
                "axis_name_placeholder",
                [f"{path}.name"],
                not placeholder.search(direction.name),
                "Axis names must not contain placeholder patterns",
            )
            normalized = "".join(
                unicodedata.normalize("NFKC", direction.name).split()
            ).casefold()
            check(
                "axis_name_quantity",
                [f"{path}.name"],
                normalized not in unit_only,
                "A unit alone is not a numerical quantity",
            )
    for paths in pair_names.values():
        check(
            "distinct_axis_names",
            paths,
            len(paths) == 1,
            "Expanded X-Y Axis names must be unique across the Figure",
        )
    valid = True
    try:
        ChartPlan(source_asset=asset, axes=structure.expand(), datasets=[])
    except (ValidationError, ValueError):
        valid = False
    check(
        "axis_chart_plan",
        ["$"],
        valid,
        "Expanded Axis names or mappings must satisfy ChartPlan structural rules",
    )
    return issues


def validate_datasets(
    proposal: DatasetStructure,
    axes: list[AxisPlan],
    asset: SourceAsset,
    *,
    node: str = "validate_dataset_plan",
    checks: list[dict] | None = None,
) -> list[ValidationIssue]:
    """Report existing structural rules without tightening naming requirements."""
    issues = []

    def check(code, paths, passed, message, *ids):
        if not _record_check(checks, code, paths, passed, message):
            issues.append(
                ValidationIssue(
                    code=code,
                    message=f"{', '.join(paths)}: {message}",
                    node=node,
                    evidence_ids=list(ids),
                    repair="semantic",
                )
            )

    check(
        "dataset_inventory_empty",
        ["datasets"],
        bool(proposal.datasets),
        "At least one visible Dataset is required",
    )
    axis_ids = {axis.id for axis in axes}
    used_axes = {dataset.axis for dataset in proposal.datasets}
    for axis in axes:
        check(
            "axis_without_dataset",
            ["datasets"],
            axis.id in used_axes,
            f"Approved Axis {axis.id} needs at least one visible Dataset",
            axis.id,
        )

    names = {}
    bands = {}
    marker = re.compile(
        r"(?:\b(?:open|filled|half-filled)\s+(?:inverted\s+)?"
        r"(?:circle|square|triangle|diamond)\b|\b(?:cross|plus|star)\b)",
        re.IGNORECASE,
    )
    line = re.compile(r"\b(?:solid|dashed|dotted|dash-dot)\b", re.IGNORECASE)
    for index, dataset in enumerate(proposal.datasets):
        dataset_id = f"dataset_{index:03d}"
        root = f"datasets[{index}]"
        name_paths = [f"{root}.name"]
        check(
            "dataset_axis_unknown",
            [f"{root}.axis"],
            dataset.axis in axis_ids,
            "Dataset must reference one of the approved Axis IDs",
            dataset_id,
        )
        name = dataset.name.strip()
        normalized = "".join(name.split()).casefold()
        check(
            "dataset_name_invalid",
            name_paths,
            bool(name)
            and not re.search(
                r"\b(?:axis|dataset|curve)[_ ]\d+\b|<[^>]+>|\.\.\.|…",
                name,
                re.IGNORECASE,
            ),
            "Dataset name must be nonempty and contain no placeholder patterns",
            dataset_id,
        )
        names.setdefault(normalized, []).append((f"{root}.name", dataset_id))
        check(
            "arrow_dataset",
            name_paths,
            not re.search(r"\brun[ -]?out\b", name, re.IGNORECASE),
            "Arrowed markers remain in their marker Dataset; runout is not a Dataset",
            dataset_id,
        )
        if dataset.kind == "scatter":
            check(
                "marker_name_incomplete",
                name_paths,
                bool(marker.search(name)),
                "Scatter name must contain a recognized fill/shape descriptor",
                dataset_id,
            )
        elif dataset.kind == "curve":
            check(
                "curve_name_incomplete",
                name_paths,
                bool(line.search(name)) and "curve" in name.casefold(),
                "Curve name must contain a recognized line style and the word curve",
                dataset_id,
            )
        elif dataset.kind == "point_group":
            check(
                "point_group_name_invalid",
                name_paths,
                name.endswith("Average Value + Spread of Value"),
                "Vertical spread uses '<property> Average Value + Spread of Value'",
                dataset_id,
            )
        elif dataset.kind == "distribution_boundary":
            check(
                "distribution_name_incomplete",
                name_paths,
                bool(line.search(name))
                and "frequency distribution boundary" in name.casefold(),
                "Distribution name must contain line style and 'frequency distribution boundary'",
                dataset_id,
            )
        elif dataset.kind == "range_boundary":
            match = re.search(r"\b(upper|lower) boundary\b", name, re.IGNORECASE)
            valid = bool(line.search(name) and match)
            check(
                "range_name_incomplete",
                name_paths,
                valid,
                "Range boundary name must contain line style and upper/lower boundary",
                dataset_id,
            )
            if valid:
                identity = "".join(
                    re.sub(
                        r"\b(?:upper|lower) boundary\b", "boundary", name.casefold()
                    ).split()
                )
                bands.setdefault((dataset.axis, identity), []).append(
                    (match.group(1).casefold(), f"{root}.name")
                )
    for entries in names.values():
        check(
            "duplicate_dataset_name",
            [path for path, _ in entries],
            len(entries) == 1,
            "Dataset names must be unique across the whole Figure project",
            *(identifier for _, identifier in entries),
        )
    for (axis_id, _), entries in bands.items():
        check(
            "incomplete_range_band",
            [path for _, path in entries],
            {role for role, _ in entries} == {"upper", "lower"},
            "A range band needs one upper and one lower boundary",
            axis_id,
        )
    valid = True
    try:
        ChartPlan(source_asset=asset, axes=axes, datasets=proposal.expand())
    except (ValidationError, ValueError):
        valid = False
    check(
        "dataset_structure_invalid",
        ["$"],
        valid,
        "Expanded Dataset names or Axis references must satisfy ChartPlan structural rules",
    )
    return issues
