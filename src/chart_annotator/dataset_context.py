"""Render a compact Axis-ownership sheet for Dataset planning."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from chart_annotator.calibration_review import MAX_REVIEW_IMAGE_SIDE
from chart_annotator.domain.workflow import Calibration, Structure
from chart_annotator.intake import write_json

X_COLOR = "#ff7f0e"
COLORS = (
    "#0066cc",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#e377c2",
)


def render(
    image: Path, structure: Structure, calibration: Calibration, directory: Path
) -> tuple[Path, dict]:
    """Show the approved Axis spans without Dataset examples or sampled data."""
    with Image.open(image) as opened:
        chart = opened.convert("RGB")
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    fits_by_axis = {
        axis.id: [fit for fit in calibration.fits if fit.axis_id == axis.id]
        for axis in structure.axes
    }
    legend_height = 38 + 25 * len(structure.axes)
    sheet = Image.new("RGB", (chart.width, chart.height + legend_height), "white")
    sheet.paste(chart, (0, legend_height))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 8),
        "Orange = X direction; Axis legend colors = Y directions.",
        fill="black",
        font=font,
    )
    aliases = []
    for index, axis in enumerate(structure.axes, 1):
        alias = f"A{index}"
        axis_color = COLORS[(index - 1) % len(COLORS)]
        draw.text(
            (12, 34 + 25 * (index - 1)),
            f"{alias} = {axis.id} = {axis.display_name}",
            fill=axis_color,
            font=small,
            stroke_width=1,
            stroke_fill="white",
        )
        directions = []
        for fit in fits_by_axis[axis.id]:
            color = X_COLOR if fit.direction == "x" else axis_color
            points = [(p.pixel[0], p.pixel[1] + legend_height) for p in fit.points]
            if not points:
                continue
            if fit.direction == "x":
                coordinate = sum(y for _, y in points) / len(points)
                start, end = min(x for x, _ in points), max(x for x, _ in points)
                line = (start, coordinate, end, coordinate)
                label_at = (start + 5, coordinate + 5)
            else:
                coordinate = sum(x for x, _ in points) / len(points)
                start, end = min(y for _, y in points), max(y for _, y in points)
                line = (coordinate, start, coordinate, end)
                label_at = (coordinate + 5, start + 5)
            draw.line(line, fill=color, width=4)
            for x, y in points:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
            draw.text(
                label_at,
                f"{alias}-{fit.direction.upper()}",
                fill=color,
                font=small,
                stroke_width=2,
                stroke_fill="white",
            )
            directions.append(
                {
                    "direction": fit.direction,
                    "scale": fit.scale,
                    "color": color,
                    "points": [
                        {"pixel": list(point.pixel), "value": point.value}
                        for point in fit.points
                    ],
                }
            )
        aliases.append(
            {
                "alias": alias,
                "axis_id": axis.id,
                "display_name": axis.display_name,
                "directions": directions,
            }
        )

    original_size = sheet.size
    if max(sheet.size) > MAX_REVIEW_IMAGE_SIDE:
        ratio = MAX_REVIEW_IMAGE_SIDE / max(sheet.size)
        target = tuple(max(1, round(value * ratio)) for value in sheet.size)
        sheet = sheet.resize(target, Image.Resampling.LANCZOS)
    context = {
        "schema_version": "dataset-context/v1",
        "source_image_size": list(chart.size),
        "original_sheet_size": list(original_size),
        "sent_sheet_size": list(sheet.size),
        "axes": aliases,
        "instruction": (
            "Orange lines and rings are approved X mappings; each Axis legend color "
            "marks its approved Y mapping. They are not data marks. Return the actual axis_id."
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dataset_context.png"
    sheet.save(path)
    write_json(path.with_suffix(".json"), context)
    return path, context
