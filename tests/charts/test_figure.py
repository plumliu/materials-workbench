import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from chart_annotator.figure import ingest_figure, render_figure
from chart_annotator.graph import build_workflow
from chart_annotator.intake import run_intake
from test_workflow_e2e import tool_reply


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_crop_rotation_transform_matches_rendered_pixels(tmp_path, rotation):
    source = tmp_path / "crop.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=210.5, height=300.5)
        page.draw_rect(pymupdf.Rect(70, 90, 80, 100), color=(0, 0, 0), fill=(0, 0, 0))
        page.set_cropbox(pymupdf.Rect(20, 30, 190.5, 270.5))
        page.set_rotation(rotation)
        pdf.save(source)
    directory, asset = ingest_figure(source, tmp_path / "output")
    assert asset.original_size == (170.5, 240.5)
    rendered, image_path = render_figure(directory, asset)
    assert rendered.rotation == rotation
    matrix = pymupdf.Matrix(*rendered.pdf_to_pixel_transform)
    with pymupdf.open(source) as pdf:
        center = (
            pdf[0].get_drawings()[0]["rect"].tl + pdf[0].get_drawings()[0]["rect"].br
        ) / 2
    pixel = center * matrix
    recovered = pixel * ~matrix
    assert tuple(recovered) == pytest.approx(tuple(center), abs=1e-5)
    with Image.open(image_path) as image:
        assert image.size == (rendered.render_width, rendered.render_height)
        assert image.getpixel((round(pixel.x), round(pixel.y))) == (0, 0, 0)
    assert json.loads((directory / "render/transform.json").read_text())[
        "pixel_to_pdf"
    ] == list(~matrix)
    repeated, repeated_path = render_figure(directory, asset)
    assert repeated == rendered
    assert repeated_path == image_path


def test_image_preserves_pixels(tmp_path):
    source = tmp_path / "image.png"
    original = Image.new("RGB", (101, 203), (20, 40, 60))
    original.save(source)
    directory, asset = ingest_figure(source, tmp_path)
    rendered, path = render_figure(directory, asset)
    assert rendered.pdf_to_pixel_transform is None
    assert Image.open(path).tobytes() == original.tobytes()


def test_job_provenance_and_panel_free_figure_progress(tmp_path):
    class NoReadableAxisModel:
        def complete(self, messages, **kwargs):
            return tool_reply(
                "submit_axes",
                {"groups": [], "unresolved": ["No readable coordinate axes"]},
            )

    source = tmp_path / "page.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text(
            (30, 100), "Figure 1.2 A complete caption for a test chart"
        )
        pdf.save(source)
    run_intake(source, tmp_path / "assets", tmp_path / "run")
    job = tmp_path / "assets/page/Figure_1.2/Figure_1.2.pdf"
    for mode, status in [("render", "render_complete"), ("figure", "needs_resolution")]:
        result = build_workflow(
            model=NoReadableAxisModel() if mode == "figure" else None
        ).invoke(
            {
                "input_path": str(job),
                "output_dir": str(tmp_path / mode),
                "mode": mode,
                "source_context": {
                    "source_id": "page/Figure_1.2",
                    "manual_id": "page",
                    "figure_id": "1.2",
                    "source_page": 1,
                    "figure_job_id": "page/Figure_1.2",
                },
            }
        )
        assert result["status"] == status
        asset = result["source_asset"]
        assert asset.figure_id == "1.2" and asset.source_page == 1
        assert asset.manual_id in asset.figure_job_id
        assert Path(result["rendered_figure"]).exists()
        if mode == "figure":
            assert result["validation_issues"][0].node == "validate_axes_plan"
            assert Path(result["pdf_text_observations"]).exists()
            assert Path(result["ocr_text_observations"]).exists()


def test_multipage_input_requires_intake(tmp_path):
    source = tmp_path / "manual.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        pdf.save(source)
    with pytest.raises(ValueError, match="single-page"):
        ingest_figure(source, tmp_path)
