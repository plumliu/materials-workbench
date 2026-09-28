"""One visual feedback sheet from local evidence, including partial snaps."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from chart_annotator.domain.workflow import Bindings, Evidence, Grounding
from chart_annotator.intake import write_json
from chart_annotator.snapping import to_pixels

MAX_REVIEW_IMAGE_SIDE = 4096


def _limit_sheet_size(
    sheet: Image.Image, views: list[dict]
) -> tuple[Image.Image, dict]:
    """Cap the review image and keep every sheet-to-source transform exact."""
    original_size = sheet.size
    largest_side = max(original_size)
    if largest_side <= MAX_REVIEW_IMAGE_SIDE:
        scale_x = scale_y = 1.0
    else:
        ratio = MAX_REVIEW_IMAGE_SIDE / largest_side
        target = tuple(max(1, round(side * ratio)) for side in original_size)
        sheet = sheet.resize(target, Image.Resampling.LANCZOS)
        scale_x = target[0] / original_size[0]
        scale_y = target[1] / original_size[1]
        for view in views:
            view["sheet_bbox_px"] = [
                view["sheet_bbox_px"][0] * scale_x,
                view["sheet_bbox_px"][1] * scale_y,
                view["sheet_bbox_px"][2] * scale_x,
                view["sheet_bbox_px"][3] * scale_y,
            ]
            view["scale_xy"] = [
                view["scale_xy"][0] * scale_x,
                view["scale_xy"][1] * scale_y,
            ]
    return sheet, {
        "original_size": list(original_size),
        "sent_size": list(sheet.size),
        "scale_xy": [scale_x, scale_y],
        "max_side_px": MAX_REVIEW_IMAGE_SIDE,
    }


def contact_sheet(
    image: Path,
    grounding: Grounding,
    bindings: Bindings,
    evidence: Evidence,
    directory: Path,
    *,
    filename: str = "calibration_review.png",
    context_updates: dict | None = None,
) -> tuple[Path, dict]:
    with Image.open(image) as source:
        source = source.convert("RGB")
    width, height = source.size
    event_path = directory / "snap/snap_events.json"
    events = (
        json.loads(event_path.read_text(encoding="utf-8"))["events"]
        if event_path.exists()
        else []
    )
    ticks = {t.id: t for t in evidence.geometry.ticks}
    texts = {t.id: t for t in evidence.texts}
    points = []
    for axis in grounding.axes:
        for direction in ("x", "y"):
            hints = getattr(axis, direction)
            if hints is None:
                continue
            for hint in hints.anchors:
                match = next(
                    (
                        e
                        for e in events
                        if e["axis_id"] == axis.axis_id
                        and e["direction"] == direction
                        and e["visible_label"] == hint.visible_label
                        and tuple(e["normalized_hint"]) == hint.point_2d
                    ),
                    None,
                )
                points.append(
                    {
                        "axis_id": axis.axis_id,
                        "direction": direction,
                        "visible_label": hint.visible_label,
                        "value": hint.value,
                        "hint_px": to_pixels(hint.point_2d, source.size),
                        "snapped_px": match["snapped_px"] if match else None,
                        "tick_id": match["tick_id"] if match else None,
                        "text_id": match["text_id"] if match else None,
                        "snap_method": match.get("method") if match else None,
                    }
                )
    for axis in bindings.axes:
        for direction in ("x", "y"):
            selection = getattr(axis, direction)
            if selection is None:
                continue
            for item in selection.ticks:
                if any(
                    p["axis_id"] == axis.axis_id
                    and p["direction"] == direction
                    and p["tick_id"] == item.tick_id
                    for p in points
                ):
                    continue
                points.append(
                    {
                        "axis_id": axis.axis_id,
                        "direction": direction,
                        "visible_label": texts[item.text_id].normalized_text,
                        "hint_px": None,
                        "snapped_px": ticks[item.tick_id].intersection_px,
                        "tick_id": item.tick_id,
                        "text_id": item.text_id,
                        "snap_method": "additional_tick",
                    }
                )
    for index, point in enumerate(points, 1):
        point["number"] = index
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    rows, views = [], []
    top = 0

    def add_view(bbox, selected, title, zoom):
        nonlocal top
        crop = source.crop(bbox)
        scale = min(2 if zoom else 1, 680 / crop.width)
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        clean = crop.resize(size, Image.Resampling.LANCZOS)
        marked = clean.copy()
        draw = ImageDraw.Draw(marked)
        sx, sy = size[0] / crop.width, size[1] / crop.height

        def position(point):
            return ((point[0] - bbox[0]) * sx, (point[1] - bbox[1]) * sy)

        for item in selected:
            hint, snap = item["hint_px"], item["snapped_px"]
            if hint is not None and snap is not None:
                draw.line((*position(hint), *position(snap)), fill="#666666", width=2)
            if hint is not None:
                x, y = position(hint)
                draw.line((x - 4, y - 4, x + 4, y + 4), fill="#ed8800", width=2)
                draw.line((x - 4, y + 4, x + 4, y - 4), fill="#ed8800", width=2)
            if snap is not None:
                x, y = position(snap)
                if item.get("snap_method") == "label_projection":
                    draw.polygon(
                        ((x, y - 9), (x + 9, y), (x, y + 9), (x - 9, y)),
                        outline="#8b3fd1",
                        width=2,
                    )
                else:
                    draw.ellipse(
                        (x - 9, y - 9, x + 9, y + 9),
                        outline="#0070d8",
                        width=2,
                    )
            x, y = position(snap if snap is not None else hint)
            draw.text(
                (x + 12, y - 16),
                str(item["number"]),
                fill="#004477",
                font=small,
                stroke_width=1,
                stroke_fill="white",
            )
        # Clean and marked views have identical transforms except horizontal offset.
        row = Image.new("RGB", (1420, 74 + size[1] + 24 * len(selected)), "white")
        labels = ImageDraw.Draw(row)
        labels.text((16, 4), title, fill="black", font=font)
        for left, view, kind in ((16, clean, "clean"), (726, marked, "annotated")):
            labels.text((left, 28), kind, fill="#444444", font=small)
            row.paste(view, (left, 52))
            views.append(
                {
                    "title": title,
                    "kind": kind,
                    "source_bbox_px": bbox,
                    "sheet_bbox_px": [
                        left,
                        top + 52,
                        left + size[0],
                        top + 52 + size[1],
                    ],
                    "scale_xy": [sx, sy],
                }
            )
        for index, item in enumerate(selected):
            label = f"{item['number']}: {item['axis_id']} / {item['direction']} / {item['visible_label']}"
            if item.get("value") is not None:
                label += f" -> {item['value']:g}"
            if item["snapped_px"] is None:
                label += "  NOT SNAPPED"
            elif item.get("snap_method") == "label_projection":
                label += "  label projection"
            elif item.get("snap_method") == "shared_axis_reuse":
                label += "  reused shared axis"
            else:
                label += "  local T"
            if item["hint_px"] is None:
                label += " (additional tick)"
            labels.text(
                (16, 58 + size[1] + 24 * index), label, fill="black", font=small
            )
        rows.append(row)
        top += row.height

    header = Image.new("RGB", (1420, 72), "white")
    draw = ImageDraw.Draw(header)
    draw.text(
        (16, 5),
        "Orange X: grounding | Blue ring: local T | Purple diamond: label projection | Gray: displacement",
        font=font,
        fill="black",
    )
    draw.text(
        (16, 32),
        f"Original chart: {width} x {height}. Corrections: 0-1000 in ORIGINAL chart, never this sheet.",
        font=font,
        fill="black",
    )
    rows.append(header)
    top = header.height
    add_view((0, 0, width, height), points, "FULL CHART OVERVIEW", False)
    for axis in grounding.axes:
        for direction in ("x", "y"):
            hint = getattr(axis, direction)
            if hint is None:
                continue
            selected = [
                p
                for p in points
                if p["axis_id"] == axis.axis_id and p["direction"] == direction
            ]
            coordinates = [to_pixels(p.point_2d, source.size) for p in hint.anchors]
            coordinates.extend(
                p[k]
                for p in selected
                for k in ("hint_px", "snapped_px")
                if p[k] is not None
            )
            # Keep both adjacent labels at shared boundaries and all chosen text boxes.
            for p in selected:
                if p["text_id"] in texts:
                    box = texts[p["text_id"]].bbox_px
                    if box is not None:
                        coordinates.extend([(box[0], box[1]), (box[2], box[3])])
            xs, ys = zip(*coordinates, strict=True)
            padding = max(50, round(max(source.size) * 0.04))
            box = (
                max(0, math.floor(min(xs) - padding)),
                max(0, math.floor(min(ys) - padding)),
                min(width, math.ceil(max(xs) + padding)),
                min(height, math.ceil(max(ys) + padding)),
            )
            add_view(
                box,
                selected,
                f"{axis.axis_id} / {direction} | original crop {box}",
                True,
            )
    sheet = Image.new("RGB", (1420, top), "white")
    top = 0
    for row in rows:
        sheet.paste(row, (0, top))
        top += row.height
    sheet, sheet_resize = _limit_sheet_size(sheet, views)
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    context = {
        "coordinate_system": "normalized_0_1000_original_chart",
        "original_image_size": [width, height],
        "contact_sheet_resize": sheet_resize,
        "views": views,
        "points": points,
    }
    if context_updates:
        context.update(context_updates)
    write_json(path.with_suffix(".json"), context)
    return path, context
