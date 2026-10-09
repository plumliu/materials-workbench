"""Grounding supplies search windows; only locally observed ink supplies pixels."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from chart_annotator import geometry, tick_ocr
from chart_annotator.calibration import _label_spacing_mapping, outside_spine
from chart_annotator.domain.models import (
    TextObservation,
    TickCandidate,
    ValidationIssue,
)
from chart_annotator.domain.workflow import (
    AxisBinding,
    Bindings,
    DirectionBinding,
    Evidence,
    Geometry,
    Grounding,
    ResolvedPoint,
    Structure,
    TickSelection,
)
from chart_annotator.intake import write_json
from chart_annotator.text_evidence import numeric_value


def to_pixels(point, image_size):
    """Qwen normalized coordinates refer to the full transmitted render, not ViT patches."""
    if any(not math.isfinite(v) or not 0 <= v <= 1000 for v in point):
        raise ValueError("Grounding must be normalized_0_1000")
    return tuple(
        float(v) * size / 1000 for v, size in zip(point, image_size, strict=True)
    )


def _merge_collinear_spines(found, orientation, center, radius):
    """Join raster-fragmented pieces of one locally grounded axis line."""
    along = 0 if orientation == "horizontal" else 1
    candidates = [
        spine
        for spine in found.spines
        if spine.orientation == orientation and abs(spine.coordinate - center) <= radius
    ]
    candidates.sort(key=lambda spine: spine.bbox[along])
    clusters = []
    for spine in candidates:
        if not clusters:
            clusters.append([spine])
            continue
        previous = clusters[-1]
        end = max(item.bbox[along + 2] for item in previous)
        coordinate = sum(item.coordinate * item.strength for item in previous) / sum(
            item.strength for item in previous
        )
        if (
            spine.bbox[along] <= end + radius
            and abs(spine.coordinate - coordinate)
            <= max(spine.width, *(item.width for item in previous)) + 2
        ):
            previous.append(spine)
        else:
            clusters.append([spine])

    merged = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        ids = {spine.id for spine in cluster}
        strength = sum(spine.strength for spine in cluster)
        first = cluster[0]
        combined = first.model_copy(
            update={
                "bbox": (
                    min(spine.bbox[0] for spine in cluster),
                    min(spine.bbox[1] for spine in cluster),
                    max(spine.bbox[2] for spine in cluster),
                    max(spine.bbox[3] for spine in cluster),
                ),
                "coordinate": sum(
                    spine.coordinate * spine.strength for spine in cluster
                )
                / strength,
                "width": max(spine.width for spine in cluster),
                "strength": strength,
            }
        )
        for tick in found.ticks:
            if tick.spine_id in ids:
                tick.spine_id = combined.id
        merged.append(combined)
    untouched = [
        spine
        for spine in found.spines
        if spine.orientation != orientation or spine not in candidates
    ]
    found.spines = [*untouched, *merged]


def _interval_distance(value: float, interval: tuple[float, float]) -> float:
    """Distance from one coordinate to a closed interval."""
    low, high = interval
    return max(low - value, 0.0, value - high)


def _text_axis_interval(text: TextObservation, direction: str):
    if text.bbox_px is None:
        return None
    box = text.bbox_px
    return (box[0], box[2]) if direction == "x" else (box[1], box[3])


def _nearby_value_texts(
    texts: list[TextObservation],
    value: float,
    direction: str,
    spine,
    side: int,
    hinted_pixel,
    radius: float,
):
    """Find local printed evidence without assuming its center is the axis point.

    A tick label may be centered, left/right aligned, or displaced around a shared
    boundary.  Its bounding interval is therefore evidence for an axis coordinate,
    while its center is never treated as the coordinate itself.
    """
    along = 0 if direction == "x" else 1
    matches = []
    for text in texts:
        if (
            text.numeric_value is None
            or text.bbox_px is None
            or not math.isclose(text.numeric_value, value, rel_tol=1e-9, abs_tol=1e-12)
            or not outside_spine(text, direction, spine.coordinate, side)
        ):
            continue
        interval = _text_axis_interval(text, direction)
        box = text.bbox_px
        glyph_height = max(1.0, box[3] - box[1])
        across_interval = (box[1], box[3]) if direction == "x" else (box[0], box[2])
        # Stay local to both Qwen's semantic hint and the observed axis spine.
        if _interval_distance(hinted_pixel[along], interval) > radius:
            continue
        if _interval_distance(spine.coordinate, across_interval) > max(
            radius, 4 * glyph_height
        ):
            continue
        matches.append(text)
    return sorted(
        matches,
        key=lambda text: (
            text.source != "pdf_text_layer",
            _interval_distance(
                hinted_pixel[along], _text_axis_interval(text, direction)
            ),
            text.id,
        ),
    )


def _project_labels_from_mapping(
    *,
    anchored,
    already_selected,
    texts,
    direction,
    spine,
    side,
    scale,
    slope,
    intercept,
    radius,
    prefix,
):
    """Project missing printed labels onto a fitted spine without using centers.

    The mapping must already be established by two ink intersections.  A missing
    label is accepted only when the mapping predicts a coordinate inside (or within
    raster tolerance of) that label's own bounding interval.  Thus these points
    validate an existing map; they cannot move it.
    """
    along = 0 if direction == "x" else 1
    selected_values = {item[0].value for item in already_selected.values()}
    projected = []
    tolerance = max(2.0, 2 * spine.width)
    for anchor, hinted_pixel in anchored:
        if anchor.value in selected_values or (scale == "log" and anchor.value <= 0):
            continue
        transformed = math.log10(anchor.value) if scale == "log" else anchor.value
        predicted = float(slope * transformed + intercept)
        candidates = _nearby_value_texts(
            texts,
            anchor.value,
            direction,
            spine,
            side,
            hinted_pixel,
            radius,
        )
        supported = [
            text
            for text in candidates
            if _interval_distance(predicted, _text_axis_interval(text, direction))
            <= tolerance
        ]
        if not supported or abs(predicted - hinted_pixel[along]) > radius:
            continue
        text = min(
            supported,
            key=lambda item: (
                item.source != "pdf_text_layer",
                _interval_distance(predicted, _text_axis_interval(item, direction)),
                item.id,
            ),
        )
        point = (
            (predicted, spine.coordinate)
            if direction == "x"
            else (spine.coordinate, predicted)
        )
        tick = TickCandidate(
            id=f"{prefix}_label_{len(projected)}",
            spine_id=spine.id,
            intersection_px=point,
            provenance="label_projection",
            text_observation_ids=[text.id],
            status=("pdf_only" if text.source == "pdf_text_layer" else "ocr_only"),
        )
        projected.append((tick, anchor, hinted_pixel, text.id))
        selected_values.add(anchor.value)
    return projected


def _label_spacing_selection(
    anchored,
    ticks,
    texts,
    direction,
    spine,
    radius,
    prefix,
    *,
    scale="linear",
    rejections=None,
):
    """Select one real station and same-column raw labels; fitting stays in calibration."""
    along = 0 if direction == "x" else 1
    tolerance = max(2.0, 2 * spine.width)
    valid = [(a, p) for a, p in anchored if numeric_value(a.visible_label) == a.value]
    if len({a.value for a, _ in valid}) < 2:
        return None
    sources = sorted(
        {t.source for t in texts}, key=lambda s: (s != "pdf_text_layer", s)
    )
    failures = []
    for source in sources:
        choices = []
        for side in (0, 1):
            pool = [
                t
                for t in texts
                if t.source == source
                and t.numeric_value is not None
                and t.bbox_px
                and outside_spine(t, direction, spine.coordinate, side)
                and _interval_distance(
                    spine.coordinate,
                    (t.bbox_px[1], t.bbox_px[3])
                    if direction == "x"
                    else (t.bbox_px[0], t.bbox_px[2]),
                )
                <= radius
                and _text_axis_interval(t, direction)[1]
                >= min(p[along] for _, p in valid) - radius
                and _text_axis_interval(t, direction)[0]
                <= max(p[along] for _, p in valid) + radius
            ]
            # A column has a common across-axis interval; separate columns cannot vote together.
            columns = []
            for text in sorted(pool, key=lambda t: t.bbox_px[1 - along]):
                box = text.bbox_px
                interval = (box[1 - along], box[3 - along])
                matching = [
                    c
                    for c in columns
                    if max(c[0], interval[0]) <= min(c[1], interval[1])
                ]
                if len(matching) > 1:
                    raise ValueError("Competing label columns near grounding")
                if matching:
                    column = matching[0]
                    column[0], column[1] = (
                        max(column[0], interval[0]),
                        min(column[1], interval[1]),
                    )
                    column[2].append(text)
                else:
                    columns.append([*interval, [text]])
            for _, _, column in columns:
                # Equivalent overlapping readings from one source are one physical label.
                unique = []
                for text in sorted(column, key=lambda t: t.id):
                    if not any(
                        t.numeric_value == text.numeric_value
                        and max(
                            _text_axis_interval(t, direction)[0],
                            _text_axis_interval(text, direction)[0],
                        )
                        <= min(
                            _text_axis_interval(t, direction)[1],
                            _text_axis_interval(text, direction)[1],
                        )
                        for t in unique
                    ):
                        unique.append(text)
                claimed = []
                ambiguous = False
                for anchor, pixel in valid:
                    matches = [
                        t
                        for t in unique
                        if t.numeric_value == anchor.value
                        and _interval_distance(
                            pixel[along], _text_axis_interval(t, direction)
                        )
                        <= radius
                    ]
                    if len(matches) > 1:
                        ambiguous = True
                    elif matches:
                        claimed.append((anchor, pixel, matches[0]))
                if len({t.id for _, _, t in claimed}) < 2:
                    continue
                if ambiguous:
                    raise ValueError("A grounded value has competing label positions")
                centers = [
                    sum(_text_axis_interval(t, direction)) / 2 for _, _, t in claimed
                ]
                low, high = min(centers), max(centers)
                labels = [
                    t
                    for t in unique
                    if low - tolerance
                    <= sum(_text_axis_interval(t, direction)) / 2
                    <= high + tolerance
                ]
                if len({t.numeric_value for t in labels}) != len(labels):
                    raise ValueError(
                        "Repeated numeric values at different label positions"
                    )
                real = []
                for tick in ticks:
                    matches = [
                        t
                        for t in labels
                        if _interval_distance(
                            tick.intersection_px[along],
                            _text_axis_interval(t, direction),
                        )
                        <= tolerance
                    ]
                    if len(matches) > 1:
                        raise ValueError(
                            "A real intersection has competing visible values"
                        )
                    if matches:
                        real.append((tick, matches[0]))
                if len(real) > 1:
                    return None  # Real stations must use the original path, including its conflicts.
                if not real:
                    continue
                tick, text = real[0]
                if not any(
                    t.id == text.id and math.dist(pixel, tick.intersection_px) <= radius
                    for _, pixel, t in claimed
                ):
                    continue
                choices.append((labels, claimed, tick, text))
        if len(choices) > 1:
            raise ValueError("Competing sides or columns for label spacing")
        if choices:
            labels, claimed, real_tick, real_text = choices[0]
            centers = [
                sum(_text_axis_interval(t, direction)) / 2 for _, _, t in claimed
            ]
            low, high = min(centers), max(centers)
            pairs = [(real_tick, real_text)]
            for text in labels:
                if text.id == real_text.id:
                    continue
                center = sum(_text_axis_interval(text, direction)) / 2
                point = (
                    (center, spine.coordinate)
                    if direction == "x"
                    else (spine.coordinate, center)
                )
                pairs.append(
                    (
                        TickCandidate(
                            id=f"{prefix}_center_{len(pairs)}",
                            spine_id=spine.id,
                            intersection_px=point,
                            provenance="label_center",
                            text_observation_ids=[text.id],
                            status="pdf_only"
                            if source == "pdf_text_layer"
                            else "ocr_only",
                        ),
                        text,
                    )
                )
            by_text = {text.id: tick for tick, text in pairs}
            selection = DirectionBinding(
                spine_id=spine.id,
                ticks=[
                    TickSelection(tick_id=tick.id, text_id=text.id)
                    for tick, text in pairs
                ],
                calibration_tick_ids=(
                    real_tick.id,
                    max(
                        pairs[1:],
                        key=lambda p: abs(
                            p[0].intersection_px[along]
                            - real_tick.intersection_px[along]
                        ),
                    )[0].id,
                ),
                grounded_tick_ids=[by_text[t.id].id for _, _, t in claimed],
                interval_px=(low, high),
            )
            try:
                # Source checks share the fitter without replacing raw evidence with predicted pixels.
                _label_spacing_mapping(
                    direction,
                    scale,
                    selection,
                    [
                        ResolvedPoint(
                            tick_id=t.id,
                            text_id=text.id,
                            pixel=t.intersection_px,
                            value=text.numeric_value,
                        )
                        for t, text in pairs
                    ],
                    {t.id: t for t, _ in pairs},
                    {text.id: text for _, text in pairs},
                    spine,
                )
            except ValueError as error:
                failure = {"source": source, "reason": str(error)}
                failures.append(failure)
                if rejections is not None:
                    rejections.append(failure)
                continue
            return selection, pairs, claimed
    if failures:
        raise ValueError(
            "No text source supports label spacing: "
            + "; ".join(f"{f['source']}: {f['reason']}" for f in failures)
        )
    return None


def snap_grounding(
    structure: Structure,
    grounding: Grounding,
    evidence: Evidence,
    image: Path,
    directory: Path,
    *,
    alternative=False,
) -> tuple[Bindings, Evidence, list[ValidationIssue]]:
    """Re-detect near every hint even when the global catalog missed the true T."""
    result = evidence.model_copy(deep=True)
    bindings = Bindings(axes=[])
    issues, events, label_source_rejections = [], [], []

    def problem(code, message, ids, repair="semantic"):
        issues.append(
            ValidationIssue(
                code=code,
                message=message,
                node="fit_and_validate_calibration",
                evidence_ids=ids,
                repair=repair,
            )
        )

    if grounding.unresolved:
        problem("grounding_unresolved", "Model reported unresolved grounding", [])
    by_axis = {g.axis_id: g for g in grounding.axes}
    if len(by_axis) != len(grounding.axes) or set(by_axis) != {
        a.id for a in structure.axes
    }:
        problem(
            "axis_grounding_set",
            "Grounding must cover each approved Axis exactly once",
            [],
        )
        return bindings, result, issues
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(image) as source:
        if source.size != evidence.geometry.image_size:
            raise ValueError("Grounding image dimensions differ from evidence")
        width, height = source.size
        # Search radii are retrieval parameters, never acceptance tolerances.
        radius = max(12, max(source.size) * (0.08 if alternative else 0.04))
        axis_order = sorted(
            enumerate(structure.axes),
            key=lambda item: item[1].shared_x_axis_id is not None,
        )
        binding_by_axis = {
            axis.id: AxisBinding(axis_id=axis.id) for axis in structure.axes
        }
        bindings.axes = [binding_by_axis[axis.id] for axis in structure.axes]
        for axis_index, axis in axis_order:
            hints = by_axis[axis.id]
            binding = binding_by_axis[axis.id]
            for direction in ("x", "y"):
                hint = getattr(hints, direction)
                scale = getattr(axis, f"{direction}_scale")
                if scale == "categorical":
                    if hint is not None:
                        problem(
                            "categorical_numeric_binding",
                            "Categorical direction cannot be calibrated",
                            [axis.id],
                        )
                    continue
                if direction == "x" and axis.shared_x_axis_id:
                    shared = binding_by_axis[axis.shared_x_axis_id].x
                    if shared is not None:
                        binding.x = shared.model_copy(deep=True)
                        shared_ticks = {item.tick_id: item for item in shared.ticks}
                        tick_catalog = {t.id: t for t in result.geometry.ticks}
                        text_catalog = {t.id: t for t in result.texts}
                        for tick_hint in hint.anchors:
                            pixel = to_pixels(tick_hint.point_2d, source.size)
                            matches = [
                                (item, tick_catalog[item.tick_id])
                                for item in shared_ticks.values()
                                if item.tick_id in tick_catalog
                                and item.text_id in text_catalog
                                and text_catalog[item.text_id].numeric_value is not None
                                and math.isclose(
                                    text_catalog[item.text_id].numeric_value,
                                    tick_hint.value,
                                    rel_tol=1e-9,
                                    abs_tol=1e-12,
                                )
                            ]
                            if matches:
                                item, tick = min(
                                    matches,
                                    key=lambda match: math.dist(
                                        pixel, match[1].intersection_px
                                    ),
                                )
                                events.append(
                                    {
                                        "axis_id": axis.id,
                                        "direction": direction,
                                        "visible_label": tick_hint.visible_label,
                                        "value": tick_hint.value,
                                        "normalized_hint": tick_hint.point_2d,
                                        "hint_px": pixel,
                                        "snapped_px": tick.intersection_px,
                                        "tick_id": tick.id,
                                        "text_id": item.text_id,
                                        "crop_bbox": None,
                                        "reused_from_axis_id": axis.shared_x_axis_id,
                                        "method": (
                                            tick.provenance
                                            if tick.provenance != "tick_intersection"
                                            else "shared_axis_reuse"
                                        ),
                                    }
                                )
                        continue
                if hint is None:
                    problem(
                        "missing_grounding",
                        "Numeric direction requires grounding",
                        [axis.id],
                    )
                    continue
                along, across = (0, 1) if direction == "x" else (1, 0)
                anchored = [
                    (anchor, to_pixels(anchor.point_2d, source.size))
                    for anchor in hint.anchors
                ]
                center = float(np.median([p[across] for _, p in anchored]))
                anchored = [
                    item for item in anchored if abs(item[1][across] - center) <= radius
                ]
                anchor_pixels = [p for _, p in anchored]
                if len(anchor_pixels) < 2:
                    problem(
                        "invalid_anchor_geometry",
                        "Need two grounding anchors near one numerical axis",
                        [axis.id],
                    )
                    continue
                low, high = (
                    min(p[along] for p in anchor_pixels),
                    max(p[along] for p in anchor_pixels),
                )
                if high - low <= max(2, radius / 4):
                    problem(
                        "invalid_anchor_geometry",
                        "Need two sufficiently separated grounding anchors",
                        [axis.id],
                    )
                    continue
                box = (
                    (low - radius, center - radius, high + radius, center + radius)
                    if direction == "x"
                    else (center - radius, low - radius, center + radius, high + radius)
                )
                x0, y0, x1, y1 = (
                    max(0, math.floor(box[0])),
                    max(0, math.floor(box[1])),
                    min(width, math.ceil(box[2])),
                    min(height, math.ceil(box[3])),
                )
                prefix = f"local_{axis_index}_{direction}"
                crop_path = directory / f"{prefix}.png"
                source.crop((x0, y0, x1, y1)).save(crop_path)
                local = geometry.detect(crop_path, [], alternative)
                expected_orientation = "horizontal" if direction == "x" else "vertical"
                _merge_collinear_spines(
                    local,
                    expected_orientation,
                    center - (y0 if direction == "x" else x0),
                    radius,
                )
                for spine in local.spines:
                    spine.bbox = (
                        spine.bbox[0] + x0,
                        spine.bbox[1] + y0,
                        spine.bbox[2] + x0,
                        spine.bbox[3] + y0,
                    )
                    spine.coordinate += y0 if spine.orientation == "horizontal" else x0
                candidates = [
                    s
                    for s in local.spines
                    if s.orientation == expected_orientation
                    and abs(s.coordinate - center) <= radius
                    and s.bbox[along] <= low + radius
                    and s.bbox[along + 2] >= high - radius
                ]
                candidates.sort(key=lambda s: abs(s.coordinate - center))
                if not candidates:
                    problem(
                        "local_spine_missing",
                        "No ink-supported axis spine near grounding",
                        [axis.id],
                        "semantic" if alternative else "evidence",
                    )
                    continue
                spine = candidates[0]
                ticks = [t for t in local.ticks if t.spine_id == spine.id]
                spine.id = prefix
                for i, tick in enumerate(ticks):
                    tick.id, tick.spine_id = f"{prefix}_t{i}", prefix
                    tick.intersection_px = (
                        tick.intersection_px[0] + x0,
                        tick.intersection_px[1] + y0,
                    )
                found = Geometry(
                    image_size=source.size,
                    preprocessing="grounding-local",
                    spines=[spine],
                    ticks=ticks,
                )
                geometry.associate(found, result.texts)
                known_texts = {t.id: t for t in result.texts}
                supported_ticks = [
                    t
                    for t in ticks
                    if any(
                        known_texts[i].numeric_value in {a.value for a, _ in anchored}
                        for i in t.text_observation_ids
                    )
                ]
                ocr = tick_ocr.read_ticks(
                    image,
                    found,
                    result.texts,
                    evidence.source_id,
                    directory / prefix,
                    search_unlabelled=True,
                    grounded_range=(direction, low, high)
                    if len(supported_ticks) < 2
                    else None,
                )
                for i, text in enumerate(ocr.observations):
                    text.id = f"{prefix}_ocr_{i}"
                result.texts.extend(ocr.observations)
                geometry.associate(found, result.texts)
                result.geometry.spines.append(spine)
                result.geometry.ticks.extend(ticks)
                texts = {t.id: t for t in result.texts}
                # Infer the label side from grounded values. The model anchors,
                # rather than a separately detected plot rectangle, own locality.
                side_scores = []
                for candidate_side in (0, 1):
                    values = {
                        texts[i].numeric_value
                        for tick in ticks
                        for i in tick.text_observation_ids
                        if outside_spine(
                            texts[i], direction, spine.coordinate, candidate_side
                        )
                    }
                    side_scores.append(
                        sum(
                            any(
                                v is not None
                                and math.isclose(
                                    v, anchor.value, rel_tol=1e-9, abs_tol=1e-12
                                )
                                for v in values
                            )
                            for anchor, _ in anchored
                        )
                    )
                side = max(range(2), key=lambda i: side_scores[i])
                numbered = [
                    tick
                    for tick in ticks
                    if any(
                        outside_spine(texts[i], direction, spine.coordinate, side)
                        for i in tick.text_observation_ids
                    )
                ]
                candidate_ticks = [
                    tick
                    for tick in ticks
                    if low - radius <= tick.intersection_px[along] <= high + radius
                ]

                def try_label_spacing():
                    rejected = []
                    try:
                        selected = _label_spacing_selection(
                            anchored,
                            candidate_ticks,
                            result.texts,
                            direction,
                            spine,
                            radius,
                            prefix,
                            scale=scale,
                            rejections=rejected,
                        )
                    except ValueError as error:
                        problem("label_spacing_ambiguous", str(error), [axis.id])
                        return True
                    finally:
                        label_source_rejections.extend(
                            {"axis_id": axis.id, "direction": direction, **r}
                            for r in rejected
                        )
                    if selected is None:
                        return False
                    selection, pairs, claimed = selected
                    setattr(binding, direction, selection)
                    for tick, text in pairs:
                        if tick.provenance == "label_center":
                            result.geometry.ticks.append(tick)
                        elif text.id not in tick.text_observation_ids:
                            tick.text_observation_ids.append(text.id)
                    by_text = {text.id: tick for tick, text in pairs}
                    for anchor, pixel, text in claimed:
                        tick = by_text[text.id]
                        events.append(
                            {
                                "axis_id": axis.id,
                                "direction": direction,
                                "visible_label": anchor.visible_label,
                                "value": anchor.value,
                                "normalized_hint": anchor.point_2d,
                                "hint_px": pixel,
                                "snapped_px": tick.intersection_px,
                                "tick_id": tick.id,
                                "text_id": text.id,
                                "crop_bbox": (x0, y0, x1, y1),
                                "method": tick.provenance,
                            }
                        )
                    return True

                if len(candidate_ticks) < 2:
                    if try_label_spacing():
                        continue
                    problem(
                        "local_ticks_missing",
                        "Need two real intersections, or one real intersection with two same-layout numeric labels",
                        [axis.id],
                        "semantic" if alternative else "evidence",
                    )
                    continue
                interval = (
                    min(tick.intersection_px[along] for tick in candidate_ticks),
                    max(tick.intersection_px[along] for tick in candidate_ticks),
                )
                if interval[0] >= interval[1]:
                    problem(
                        "local_range_missing",
                        "Need two distinct local tick positions",
                        [axis.id],
                        "semantic" if alternative else "evidence",
                    )
                    continue
                options = []
                for tick_hint, pixel in anchored:
                    parsed = numeric_value(tick_hint.visible_label)
                    if parsed is None or not math.isclose(
                        parsed, tick_hint.value, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        continue
                    matches = []
                    for tick in candidate_ticks:
                        if (
                            not interval[0] - spine.width
                            <= tick.intersection_px[along]
                            <= interval[1] + spine.width
                        ):
                            continue
                        ids = [
                            i
                            for i in tick.text_observation_ids
                            if texts[i].numeric_value is not None
                            and math.isclose(
                                texts[i].numeric_value,
                                tick_hint.value,
                                rel_tol=1e-9,
                                abs_tol=1e-12,
                            )
                            and outside_spine(
                                texts[i], direction, spine.coordinate, side
                            )
                        ]
                        # Shared panel boundaries can print two values around one
                        # geometric intersection (for example upper 110 and lower
                        # 30).  The glyph box need not overlap the line.  When the
                        # model has grounded the correct value beside a real local
                        # T, allow that nearby text to identify the T; the global
                        # multi-anchor fit below still has to validate it.
                        if not ids and math.dist(pixel, tick.intersection_px) <= radius:
                            nearby = _nearby_value_texts(
                                result.texts,
                                tick_hint.value,
                                direction,
                                spine,
                                side,
                                pixel,
                                radius,
                            )
                            along_interval_distance = [
                                (
                                    _interval_distance(
                                        tick.intersection_px[along],
                                        _text_axis_interval(text, direction),
                                    ),
                                    text,
                                )
                                for text in nearby
                            ]
                            close = [
                                (distance, text)
                                for distance, text in along_interval_distance
                                if distance
                                <= max(
                                    2 * spine.width,
                                    text.bbox_px[3] - text.bbox_px[1],
                                )
                            ]
                            if close:
                                close.sort(
                                    key=lambda item: (
                                        item[0],
                                        item[1].source != "pdf_text_layer",
                                        item[1].id,
                                    )
                                )
                                ids = [close[0][1].id]
                        if ids:
                            # Equivalent PDF/OCR readings stay in evidence; choose a stable ID.
                            ids.sort(
                                key=lambda i: (texts[i].source != "pdf_text_layer", i)
                            )
                            matches.append((tick, ids[0]))
                    if matches:
                        options.append((tick_hint, pixel, matches))

                hypotheses = []
                # A scanned line has uncertainty on both sides of its measured
                # center.  Two observed line widths admit small print/scan drift
                # while still rejecting a neighboring grid/data station.
                fit_tolerance = max(2.0, 2 * spine.width)
                for first in range(len(options)):
                    for second in range(first + 1, len(options)):
                        h0, _, matches0 = options[first]
                        h1, _, matches1 = options[second]
                        if scale == "log" and min(h0.value, h1.value) <= 0:
                            continue
                        v0 = math.log10(h0.value) if scale == "log" else h0.value
                        v1 = math.log10(h1.value) if scale == "log" else h1.value
                        if v0 == v1:
                            continue
                        for tick0, _ in matches0:
                            for tick1, _ in matches1:
                                if tick0.id == tick1.id:
                                    continue
                                p0 = tick0.intersection_px[along]
                                p1 = tick1.intersection_px[along]
                                if abs(p1 - p0) <= 2 * spine.width:
                                    continue
                                m = (p1 - p0) / (v1 - v0)
                                b = p0 - m * v0
                                chosen = {}
                                residual = 0.0
                                hint_distance = 0.0
                                for candidate, hinted_pixel, matches in options:
                                    value = (
                                        math.log10(candidate.value)
                                        if scale == "log" and candidate.value > 0
                                        else candidate.value
                                    )
                                    ranked = sorted(
                                        (
                                            abs(
                                                m * value
                                                + b
                                                - tick.intersection_px[along]
                                            ),
                                            math.dist(
                                                hinted_pixel, tick.intersection_px
                                            ),
                                            tick,
                                            text_id,
                                        )
                                        for tick, text_id in matches
                                        if tick.id not in chosen
                                    )
                                    if ranked and ranked[0][0] <= fit_tolerance:
                                        error, displacement, tick, text_id = ranked[0]
                                        chosen[tick.id] = (
                                            candidate,
                                            hinted_pixel,
                                            text_id,
                                        )
                                        residual += error
                                        hint_distance += displacement
                                hypotheses.append(
                                    (
                                        -len(chosen),
                                        residual,
                                        hint_distance,
                                        -abs(p1 - p0),
                                        chosen,
                                    )
                                )
                best = (
                    min(hypotheses, key=lambda item: item[:4]) if hypotheses else None
                )
                if best is None or -best[0] < 2:
                    if try_label_spacing():
                        continue
                    problem(
                        "insufficient_snapped_ticks",
                        "Need two matched real anchors, or one matched real anchor with two same-layout numeric labels",
                        [axis.id],
                        "evidence",
                    )
                    continue
                selected_hints = best[4]
                selected = {
                    tick_id: data[2] for tick_id, data in selected_hints.items()
                }
                # The widest two ink intersections determine the mapping.  Labels
                # without perpendicular tick ink may validate it, but never move it.
                seed_ticks = sorted(
                    (t for t in candidate_ticks if t.id in selected_hints),
                    key=lambda t: t.intersection_px[along],
                )
                calibration_tick_ids = (seed_ticks[0].id, seed_ticks[-1].id)
                seed0, seed1 = seed_ticks[0], seed_ticks[-1]
                seed_hint0 = selected_hints[seed0.id][0]
                seed_hint1 = selected_hints[seed1.id][0]
                seed_value0 = (
                    math.log10(seed_hint0.value) if scale == "log" else seed_hint0.value
                )
                seed_value1 = (
                    math.log10(seed_hint1.value) if scale == "log" else seed_hint1.value
                )
                seed_slope = (
                    seed1.intersection_px[along] - seed0.intersection_px[along]
                ) / (seed_value1 - seed_value0)
                seed_intercept = seed0.intersection_px[along] - seed_slope * seed_value0
                projected = _project_labels_from_mapping(
                    anchored=anchored,
                    already_selected=selected_hints,
                    texts=result.texts,
                    direction=direction,
                    spine=spine,
                    side=side,
                    scale=scale,
                    slope=seed_slope,
                    intercept=seed_intercept,
                    radius=radius,
                    prefix=prefix,
                )
                projected_ids = set()
                for tick, tick_hint, pixel, text_id in projected:
                    ticks.append(tick)
                    candidate_ticks.append(tick)
                    numbered.append(tick)
                    result.geometry.ticks.append(tick)
                    selected_hints[tick.id] = (tick_hint, pixel, text_id)
                    selected[tick.id] = text_id
                    projected_ids.add(tick.id)
                anchor_tick_ids = list(selected)
                for tick_id, (tick_hint, pixel, text_id) in selected_hints.items():
                    tick = next(t for t in candidate_ticks if t.id == tick_id)
                    events.append(
                        {
                            "axis_id": axis.id,
                            "direction": direction,
                            "visible_label": tick_hint.visible_label,
                            "value": tick_hint.value,
                            "normalized_hint": tick_hint.point_2d,
                            "hint_px": pixel,
                            "snapped_px": tick.intersection_px,
                            "tick_id": tick.id,
                            "text_id": text_id,
                            "crop_bbox": (x0, y0, x1, y1),
                            "method": (
                                "label_projection"
                                if tick.id in projected_ids
                                else "tick_intersection"
                            ),
                        }
                    )
                anchors = [t for t in candidate_ticks if t.id in selected]
                vals = [texts[selected[t.id]].numeric_value for t in anchors]
                if scale == "log" and any(v <= 0 for v in vals):
                    problem(
                        "nonpositive_log",
                        "Logarithmic labels must be positive",
                        [axis.id],
                    )
                    continue
                vals = [math.log10(v) if scale == "log" else v for v in vals]
                if len(set(vals)) != len(vals):
                    problem(
                        "singular_calibration",
                        "Grounded values must be distinct",
                        [axis.id],
                    )
                    continue
                m, b = np.linalg.lstsq(
                    np.column_stack((vals, np.ones(len(vals)))),
                    [t.intersection_px[along] for t in anchors],
                    rcond=None,
                )[0]
                for tick in numbered:
                    if (
                        tick.id in selected
                        or not interval[0] - spine.width
                        <= tick.intersection_px[along]
                        <= interval[1] + spine.width
                    ):
                        continue
                    supported = {}
                    for i in tick.text_observation_ids:
                        t = texts[i]
                        v = t.numeric_value
                        if (
                            v is None
                            or (scale == "log" and v <= 0)
                            or not outside_spine(t, direction, spine.coordinate, side)
                        ):
                            continue
                        if (
                            abs(
                                m * (math.log10(v) if scale == "log" else v)
                                + b
                                - tick.intersection_px[along]
                            )
                            <= spine.width
                        ):
                            supported.setdefault(v, i)
                    if len(supported) == 1:
                        selected[tick.id] = next(iter(supported.values()))
                setattr(
                    binding,
                    direction,
                    DirectionBinding(
                        spine_id=spine.id,
                        interval_px=(
                            min(t.intersection_px[along] for t in anchors),
                            max(t.intersection_px[along] for t in anchors),
                        ),
                        calibration_tick_ids=calibration_tick_ids,
                        grounded_tick_ids=anchor_tick_ids,
                        ticks=[
                            TickSelection(tick_id=i, text_id=t)
                            for i, t in selected.items()
                        ],
                    ),
                )
    # Later strips can observe a shared boundary too.  Reassociation may discover
    # extra labels, but it must not discard a model-grounded nearby boundary label
    # merely because that glyph box does not geometrically overlap the shared line.
    bound_text = {
        item.tick_id: item.text_id
        for axis in bindings.axes
        for direction in ("x", "y")
        for selection in [getattr(axis, direction)]
        if selection is not None
        for item in selection.ticks
    }
    geometry.associate(result.geometry, result.texts)
    tick_catalog = {tick.id: tick for tick in result.geometry.ticks}
    for tick_id, text_id in bound_text.items():
        tick = tick_catalog.get(tick_id)
        if tick is not None and text_id not in tick.text_observation_ids:
            tick.text_observation_ids.append(text_id)
    write_json(
        directory / "snap_events.json",
        {
            "coordinate_system": "normalized_0_1000",
            "image_size": result.geometry.image_size,
            "events": events,
            "label_source_rejections": label_source_rejections,
        },
    )
    write_json(directory / "local_evidence.json", result.model_dump(mode="json"))
    write_json(directory / "local_bindings.json", bindings.model_dump(mode="json"))
    return bindings, result, issues
