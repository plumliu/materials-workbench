"""OpenCV line intersections and spatial text associations, never model pixels."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from chart_annotator.domain.models import TextObservation, TickCandidate
from chart_annotator.domain.workflow import Geometry, Spine


def groups(indices):
    return [
        g for g in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1) if len(g)
    ]


def associate(geometry: Geometry, texts: list[TextObservation]) -> None:
    spines = {s.id: s for s in geometry.spines}
    for tick in geometry.ticks:
        spine = spines[tick.spine_id]
        vertical = spine.orientation == "vertical"
        ids = []
        for text in texts:
            if text.numeric_value is None or text.bbox_px is None:
                continue
            x0, y0, x1, y1 = text.bbox_px
            x, y = tick.intersection_px
            height = y1 - y0
            uncertainty = spine.width + 1  # measured ink width plus raster quantization
            along = (
                y0 - uncertainty <= y <= y1 + uncertainty
                if vertical
                else x0 - uncertainty <= x <= x1 + uncertainty
            )
            gap = (
                min(abs(x - x0), abs(x - x1))
                if vertical
                else min(abs(y - y0), abs(y - y1))
            )
            # Search extent follows the observed glyph size; it is not a pass gate.
            if along and gap <= 3 * height:
                ids.append(text.id)
        tick.text_observation_ids = ids
        values = {t.numeric_value for t in texts if t.id in ids}
        sources = {t.source for t in texts if t.id in ids}
        tick.status = (
            "multiple_candidates"
            if len(values) > 1
            else "agreed"
            if len(sources) > 1
            else "pdf_only"
            if "pdf_text_layer" in sources
            else "ocr_only"
        )


def detect(image: Path, texts: list[TextObservation], alternative=False) -> Geometry:
    gray = np.asarray(Image.open(image).convert("L"))
    height, width = gray.shape
    # The verified manual drawings include light grey tick ink; Otsu can discard
    # it together with paper. This threshold proposes evidence, never passes a fit.
    dark = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    if alternative:
        dark = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
        )
    spines = []
    # Kernel lengths propose ink evidence; grounding can also trigger this same
    # detector on local crops independently of the global proposal catalog.
    for orientation, kernel in (
        ("horizontal", (max(21, width // 13), 1)),
        ("vertical", (1, max(21, height // 13))),
    ):
        mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones(kernel[::-1], np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            vertical = orientation == "vertical"
            ys, xs = np.where(labels[y : y + h, x : x + w] == i)
            coordinate = float(np.mean(xs) + x if vertical else np.mean(ys) + y)
            spines.append(
                Spine(
                    id=f"spine_{len(spines):03d}",
                    orientation=orientation,
                    bbox=(x, y, x + w, y + h),
                    coordinate=coordinate,
                    width=w if vertical else h,
                    strength=area,
                )
            )
    ticks = []
    for spine in spines:
        vertical = spine.orientation == "vertical"
        # A tick must include perpendicular ink extending beyond the spine.
        # Local search width scales with measured spine thickness.
        radius = max(6, round(spine.width * 6))
        c = round(spine.coordinate)
        x0, y0, x1, y1 = map(round, spine.bbox)
        if vertical:
            lo, hi = max(0, c - radius), min(width, c + radius + 1)
            region = dark[y0:y1, lo:hi]
        else:
            lo, hi = max(0, c - radius), min(height, c + radius + 1)
            region = dark[lo:hi, x0:x1].T
        # Rasterized T junctions can have 1-2 pixel gaps (including shared 110/30).
        # Never scale gap repair with spine thickness: that connects label glyphs.
        # Add ink only at the spine boundary; retain all original ink positions.
        edge0, edge1 = (x0, x1) if vertical else (y0, y1)
        positions = np.arange(region.shape[1]) + lo
        at_spine = (positions >= edge0 - 2) & (positions <= edge1 + 1)
        repaired = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
        region = region | (repaired * at_spine)
        scores = np.zeros(len(region))
        center = c - lo
        for row_index, row in enumerate(region):
            if not row[center]:
                continue
            # Count only contiguous ink through the spine. Nearby glyph strokes
            # separated by white pixels are not T intersections.
            left = np.flatnonzero(row[: center + 1][::-1] == 0)
            right = np.flatnonzero(row[center:] == 0)
            length = (
                (int(left[0]) if len(left) else center + 1)
                + (int(right[0]) if len(right) else len(row) - center)
                - 1
            )
            if length >= 2 * spine.width + 3:
                scores[row_index] = length
        for group in groups(np.flatnonzero(scores)):
            if len(group) > radius:
                continue
            along = float(np.average(group, weights=scores[group])) + (
                y0 if vertical else x0
            )
            point = (spine.coordinate, along) if vertical else (along, spine.coordinate)
            ticks.append(
                TickCandidate(
                    id=f"tick_{len(ticks):04d}",
                    spine_id=spine.id,
                    intersection_px=point,
                    text_observation_ids=[],
                    status="ocr_only",
                )
            )
    result = Geometry(
        image_size=(width, height),
        preprocessing="adaptive" if alternative else "gray<=210",
        spines=spines,
        ticks=ticks,
    )
    associate(result, texts)
    return result
