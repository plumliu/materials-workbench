"""Fit trustworthy endpoint mappings and treat other detected ticks as support."""

import math

import numpy as np

from chart_annotator.domain.models import ValidationIssue
from chart_annotator.domain.workflow import (
    Bindings,
    Calibration,
    Evidence,
    Fit,
    ResolvedPoint,
    Structure,
)


def outside_spine(text, direction, coordinate, side):
    if text.bbox_px is None:
        return False
    box = text.bbox_px
    center = (box[1] + box[3]) / 2 if direction == "x" else (box[0] + box[2]) / 2
    return center < coordinate if side == 0 else center > coordinate


def _coordinate_value(value: float, scale: str) -> float:
    return math.log10(value) if scale == "log" else value


def _label_spacing_mapping(direction, scale, selection, points, ticks, texts, spine):
    """Text spacing estimates slope; the only real station fixes absolute position."""
    index = 0 if direction == "x" else 1
    real = [p for p in points if ticks[p.tick_id].provenance == "tick_intersection"]
    if len(real) != 1 or len(points) < 2:
        raise ValueError(
            "Label spacing needs one real intersection and two independent labels"
        )
    if any(ticks[p.tick_id].provenance == "label_projection" for p in points):
        raise ValueError("Existing label projections cannot seed label spacing")
    if scale == "log" and any(p.value <= 0 for p in points):
        raise ValueError("Logarithmic labels must be positive")
    observations = [texts[p.text_id] for p in points]
    tolerance = max(2.0, 2 * spine.width)
    if len({p.text_id for p in points}) != len(points) or len(
        {p.value for p in points}
    ) != len(points):
        raise ValueError(
            "Label spacing requires independent labels and distinct values"
        )
    if (
        len({t.source for t in observations}) != 1
        or len({tuple(t.font_names) for t in observations}) != 1
    ):
        raise ValueError("Label spacing cannot mix text sources or layouts")
    heights = [t.bbox_px[3] - t.bbox_px[1] for t in observations if t.bbox_px]
    if len(heights) != len(points) or max(heights) - min(heights) > tolerance:
        raise ValueError("Label spacing requires consistent visible text boxes")
    grounded = set(selection.grounded_tick_ids)
    if (
        len(grounded.intersection(p.tick_id for p in points)) < 2
        or real[0].tick_id not in grounded
    ):
        raise ValueError("Two labels including the real station must be grounded")
    values = np.asarray([_coordinate_value(p.value, scale) for p in points])
    centers = np.asarray(
        [(t.bbox_px[index] + t.bbox_px[index + 2]) / 2 for t in observations]
    )
    order = np.argsort(values)
    differences = np.diff(centers[order])
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("Numeric labels must have one monotonic spatial order")
    slope, center_intercept = np.linalg.lstsq(
        np.column_stack((values, np.ones(len(values)))), centers, rcond=None
    )[0]
    if (
        not math.isfinite(slope)
        or slope == 0
        or abs(float(centers.max() - centers.min())) <= 2 * tolerance
    ):
        raise ValueError("Label spacing requires a finite well-separated mapping")
    origin = real[0]
    intercept = origin.pixel[index] - slope * _coordinate_value(origin.value, scale)
    residuals = list(abs(centers - (slope * values + center_intercept)))
    mapped = []
    for point, text, value in zip(points, observations, values, strict=True):
        coordinate = float(slope * value + intercept)
        low, high = text.bbox_px[index], text.bbox_px[index + 2]
        residuals.append(max(low - coordinate, 0.0, coordinate - high))
        pixel = (
            (coordinate, spine.coordinate)
            if direction == "x"
            else (spine.coordinate, coordinate)
        )
        mapped.append(
            point.model_copy(
                update={"pixel": origin.pixel if point is origin else pixel}
            )
        )
    if not math.isfinite(intercept) or max(residuals) > tolerance:
        raise ValueError(
            "Label spacing conflicts with original text boxes or intervals"
        )
    mapped.sort(key=lambda p: p.pixel[index])
    if mapped[-1].pixel[index] - mapped[0].pixel[index] <= 2 * tolerance:
        raise ValueError("Label spacing endpoints are too close")
    return {
        "calibration_method": "anchored_label_spacing",
        "spine_id": spine.id,
        "slope": float(slope),
        "intercept": float(intercept),
        "tolerance_px": tolerance,
        "max_residual_px": float(max(residuals)),
        "points": mapped,
        "endpoints": (mapped[0], mapped[-1]),
    }


def _fit_label_spacing(axis, direction, scale, selection, points, ticks, texts, spine):
    return Fit(
        axis_id=axis.id,
        direction=direction,
        scale=scale,
        **_label_spacing_mapping(
            direction, scale, selection, points, ticks, texts, spine
        ),
    )


def fit_bindings(structure: Structure, bindings: Bindings, evidence: Evidence):
    """Fit each numeric direction from two real, well-separated T intersections.

    Model anchors beyond the selected endpoints and automatically discovered ticks
    may add support. Missing, orphaned or contradictory auxiliary candidates never
    invalidate an otherwise self-consistent endpoint mapping.
    """
    issues, fits = [], []
    ticks = {tick.id: tick for tick in evidence.geometry.ticks}
    texts = {text.id: text for text in evidence.texts}
    spines = {spine.id: spine for spine in evidence.geometry.spines}
    by_axis = {binding.axis_id: binding for binding in bindings.axes}
    axes = {axis.id: axis for axis in structure.axes}

    def problem(code, message, ids=(), repair="semantic"):
        issues.append(
            ValidationIssue(
                code=code,
                message=message,
                node="fit_and_validate_calibration",
                evidence_ids=list(ids),
                repair=repair,
            )
        )

    if bindings.unresolved:
        problem("binding_unresolved", "Model reported unresolved bindings")
    if len(by_axis) != len(bindings.axes) or set(by_axis) != set(axes):
        problem("axis_binding_set", "Bindings must cover each Axis exactly once")
        return Calibration(fits=[]), issues

    for axis in structure.axes:
        binding = by_axis[axis.id]
        for direction in ("x", "y"):
            scale = getattr(axis, f"{direction}_scale")
            selection = getattr(binding, direction)
            if scale == "categorical":
                if selection is not None:
                    problem(
                        "categorical_numeric_binding",
                        "Categorical directions cannot have numeric bindings",
                        [axis.id],
                    )
                continue
            # A declared shared x mapping is fitted once at its root and cloned
            # below. It must not run local evidence gates once per owning Axis.
            if direction == "x" and axis.shared_x_axis_id:
                continue
            spine = spines.get(selection.spine_id) if selection else None
            expected = "horizontal" if direction == "x" else "vertical"
            if selection is None or spine is None or spine.orientation != expected:
                problem(
                    "spine_mismatch",
                    "Numeric direction needs an observed correctly oriented spine",
                    [axis.id],
                )
                continue

            index = 0 if direction == "x" else 1
            across = 1 - index
            selected_texts = [
                texts[item.text_id] for item in selection.ticks if item.text_id in texts
            ]
            side = max(
                (0, 1),
                key=lambda candidate: sum(
                    outside_spine(text, direction, spine.coordinate, candidate)
                    for text in selected_texts
                ),
            )
            points = []
            invalid = []
            nonpositive = []
            for item in selection.ticks:
                tick = ticks.get(item.tick_id)
                text = texts.get(item.text_id)
                if (
                    tick is None
                    or text is None
                    or tick.spine_id != spine.id
                    or text.id not in tick.text_observation_ids
                    or text.numeric_value is None
                    or not outside_spine(text, direction, spine.coordinate, side)
                    or abs(tick.intersection_px[across] - spine.coordinate)
                    > spine.width + 1
                ):
                    invalid.extend((item.tick_id, item.text_id))
                    continue
                if scale == "log" and text.numeric_value <= 0:
                    nonpositive.extend((item.tick_id, item.text_id))
                    continue
                points.append(
                    ResolvedPoint(
                        tick_id=tick.id,
                        text_id=text.id,
                        pixel=tick.intersection_px,
                        value=text.numeric_value,
                    )
                )
            if nonpositive:
                problem(
                    "nonpositive_log",
                    "Logarithmic calibration values must be positive",
                    [axis.id, *nonpositive],
                )
                continue
            if invalid:
                problem(
                    "invalid_tick_reference",
                    "Selected tick, text and observed spine do not agree",
                    [axis.id, *invalid],
                )
                continue

            # Snapping may discover a wider, text-confirmed T intersection after
            # its provisional pair. Re-select from all real intersections here;
            # projected labels validate the map but never define it.
            if any(ticks[p.tick_id].provenance == "label_center" for p in points):
                try:
                    fits.append(
                        _fit_label_spacing(
                            axis,
                            direction,
                            scale,
                            selection,
                            points,
                            ticks,
                            texts,
                            spine,
                        )
                    )
                except ValueError as error:
                    problem("label_spacing_invalid", str(error), [axis.id])
                continue
            real_points = [
                point
                for point in points
                if ticks[point.tick_id].provenance == "tick_intersection"
            ]
            endpoint_pairs = [
                (first, second)
                for position, first in enumerate(real_points)
                for second in real_points[position + 1 :]
                if _coordinate_value(first.value, scale)
                != _coordinate_value(second.value, scale)
            ]
            if not endpoint_pairs:
                problem(
                    "singular_calibration",
                    "Two observed calibration endpoints are required",
                    [axis.id, spine.id],
                )
                continue
            endpoints = sorted(
                max(
                    endpoint_pairs,
                    key=lambda pair: abs(pair[1].pixel[index] - pair[0].pixel[index]),
                ),
                key=lambda point: point.pixel[index],
            )
            endpoint_ids = tuple(point.tick_id for point in endpoints)
            endpoint_values = [
                _coordinate_value(point.value, scale) for point in endpoints
            ]
            endpoint_pixels = [point.pixel[index] for point in endpoints]
            # Scanned line centers and OCR overlays can drift across both sides of
            # the observed stroke.  Scale residual tolerance from actual ink width.
            tolerance = max(2.0, 2 * spine.width)
            interval = selection.interval_px or tuple(endpoint_pixels)
            minimum_span = 2 * tolerance
            if (
                endpoint_values[0] == endpoint_values[1]
                or abs(endpoint_pixels[1] - endpoint_pixels[0]) <= minimum_span
            ):
                problem(
                    "singular_calibration",
                    "Calibration endpoints need distinct values and sufficient separation",
                    [axis.id, *endpoint_ids],
                )
                continue
            slope = (endpoint_pixels[1] - endpoint_pixels[0]) / (
                endpoint_values[1] - endpoint_values[0]
            )
            intercept = endpoint_pixels[0] - slope * endpoint_values[0]
            if not math.isfinite(slope) or slope == 0 or not math.isfinite(intercept):
                problem(
                    "singular_calibration",
                    "Calibration mapping is not finite",
                    [axis.id, *endpoint_ids],
                )
                continue

            def residual(
                point,
                *,
                fitted_slope=slope,
                fitted_intercept=intercept,
                fitted_scale=scale,
                coordinate_index=index,
            ):
                return abs(
                    fitted_slope * _coordinate_value(point.value, fitted_scale)
                    + fitted_intercept
                    - point.pixel[coordinate_index]
                )

            grounded_ids = set(
                selection.grounded_tick_ids
                or [item.tick_id for item in selection.ticks]
            )
            grounded = [point for point in points if point.tick_id in grounded_ids]
            conflicting = [
                point.tick_id for point in grounded if residual(point) > tolerance
            ]
            if len(grounded) < 2 or conflicting:
                problem(
                    "grounded_anchor_conflict",
                    "At least two grounded anchors must support one endpoint mapping",
                    [axis.id, *conflicting],
                )
                continue

            # Keep snapped model anchors only when they validate the endpoint map.
            accepted = {
                point.tick_id: point for point in points if residual(point) <= tolerance
            }
            accepted.update((point.tick_id, point) for point in endpoints)
            used_values = {point.value for point in accepted.values()}
            used_pixels = {point.pixel[index] for point in accepted.values()}

            # Locally detected ticks are optional supporting evidence. Select a
            # candidate only when exactly one of its visible readings fits.
            for tick in ticks.values():
                coordinate = tick.intersection_px[index]
                if (
                    tick.spine_id != spine.id
                    or tick.id in accepted
                    or not interval[0] - tolerance
                    <= coordinate
                    <= interval[1] + tolerance
                ):
                    continue
                supported = []
                for text_id in tick.text_observation_ids:
                    text = texts.get(text_id)
                    if (
                        text is None
                        or text.numeric_value is None
                        or not outside_spine(text, direction, spine.coordinate, side)
                        or (scale == "log" and text.numeric_value <= 0)
                    ):
                        continue
                    point = ResolvedPoint(
                        tick_id=tick.id,
                        text_id=text.id,
                        pixel=tick.intersection_px,
                        value=text.numeric_value,
                    )
                    if residual(point) <= tolerance:
                        supported.append(point)
                unique = {point.value: point for point in supported}
                if len(unique) != 1:
                    continue
                point = next(iter(unique.values()))
                if point.value in used_values or point.pixel[index] in used_pixels:
                    continue
                accepted[point.tick_id] = point
                used_values.add(point.value)
                used_pixels.add(point.pixel[index])

            accepted_points = sorted(
                accepted.values(), key=lambda point: point.pixel[index]
            )
            fits.append(
                Fit(
                    axis_id=axis.id,
                    direction=direction,
                    scale=scale,
                    calibration_method="tick_intersections",
                    spine_id=spine.id,
                    slope=float(slope),
                    intercept=float(intercept),
                    tolerance_px=tolerance,
                    max_residual_px=max(residual(point) for point in accepted_points),
                    points=accepted_points,
                    endpoints=(endpoints[0], endpoints[1]),
                )
            )

    def shared_root(axis_id):
        seen = set()
        current = axes[axis_id]
        while current.shared_x_axis_id:
            if current.id in seen or current.shared_x_axis_id not in axes:
                return None
            seen.add(current.id)
            current = axes[current.shared_x_axis_id]
        return current.id

    root_fits = {fit.axis_id: fit for fit in fits if fit.direction == "x"}
    for axis in structure.axes:
        if not axis.shared_x_axis_id or axis.x_scale == "categorical":
            continue
        root = shared_root(axis.id)
        fit = root_fits.get(root) if root else None
        if fit is None or fit.scale != axis.x_scale:
            problem(
                "shared_x_missing",
                "Shared x mapping could not reuse its declared root Axis",
                [axis.id, axis.shared_x_axis_id],
            )
            continue
        fits.append(fit.model_copy(update={"axis_id": axis.id}, deep=True))

    return Calibration(fits=fits), issues
