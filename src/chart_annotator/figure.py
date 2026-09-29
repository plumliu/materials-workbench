"""Figure source provenance and a fixed, reversible render coordinate system."""

import shutil
from pathlib import Path

import pymupdf
from PIL import Image

from chart_annotator.domain.models import SourceAsset
from chart_annotator.intake import image_dimensions, open_pdf, write_json

PDF_RENDER_SCALE = 3.0  # 216 dpi, in unrotated PDF points before page rotation.


def ingest_figure(source: Path, output: Path, *, context: dict | None = None) -> tuple[Path, SourceAsset]:
    source = source.resolve(strict=True)
    kind = "pdf" if source.suffix.lower() == ".pdf" else "image"
    rotation = 0
    if kind == "pdf":
        with open_pdf(source) as doc:
            if len(doc) != 1:
                raise ValueError(
                    "Figure input must be a single-page PDF; use intake for manuals"
                )
            page = doc[0]
            size = (page.cropbox.width, page.cropbox.height)
            rotation = page.rotation
    else:
        size = image_dimensions(source)
    digest = context["source_id"] if context else source.stem
    directory = output.resolve() if context else output.resolve() / source.stem
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"source{source.suffix.lower()}"
    if context:
        target = source
    else:
        shutil.copyfile(source, target)
    asset = SourceAsset(
        source_id=digest,
        path=str(target) if context else target.name,
        original_path=str(source),
        kind=kind,
        original_size=size,
        rotation=rotation,
        manual_id=context["manual_id"] if context else None,
        source_page=context["source_page"] if context else 1,
        figure_id=context["figure_id"] if context else None,
        figure_job_id=context["figure_job_id"] if context else None,
    )
    if context:
        asset = asset.model_copy(update={key: context[key] for key in ("manual_id", "source_page", "figure_id", "figure_job_id")})
    write_json(directory / "source.json", asset.model_dump(mode="json"))
    return directory, asset


def render_figure(directory: Path, asset: SourceAsset) -> tuple[SourceAsset, Path]:
    source = directory / asset.path
    render = directory / "render"
    render.mkdir(exist_ok=True)
    path = render / "figure.png"
    matrix = None
    if asset.kind == "pdf":
        with open_pdf(source) as doc:
            page = doc[0]
            scale = pymupdf.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
            pix = page.get_pixmap(matrix=scale, colorspace=pymupdf.csRGB, alpha=False)
            matrix = (
                page.rotation_matrix
                * scale
                * pymupdf.Matrix(1, 0, 0, 1, -pix.x, -pix.y)
            )
            pix.save(path)
            width, height = pix.width, pix.height
    else:
        with Image.open(source) as image:
            # Keep original pixel positions and EXIF orientation unapplied.
            # White compositing gives downstream OCR and Qwen the same pixels.
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", image.size, "white")
            canvas.alpha_composite(rgba)
            canvas.convert("RGB").save(path)
            width, height = image.size
    rendered = asset.model_copy(
        update={
            "render_width": width,
            "render_height": height,
            "pdf_to_pixel_transform": tuple(matrix) if matrix else None,
        }
    )
    write_json(render / "source_asset.json", rendered.model_dump(mode="json"))
    write_json(
        render / "transform.json",
        {
            "schema_version": "figure-render/v1",
            "image": "render/figure.png",
            "image_size": [width, height],
            "source_coordinates": "unrotated PDF page points"
            if matrix
            else "original image pixels; EXIF unapplied",
            "pdf_to_pixel": list(matrix) if matrix else None,
            "pixel_to_pdf": list(~matrix) if matrix else None,
            "pdf_render_scale": PDF_RENDER_SCALE if matrix else None,
        },
    )
    return rendered, path
