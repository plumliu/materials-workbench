import json
from pathlib import Path

import pytest
from PIL import Image
from test_workflow_e2e import FakeChartModel, chart_pdf

from chart_annotator import calibration_review, stages
from chart_annotator.runner import run_figure


def test_review_sheet_size_cap_preserves_view_transforms(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_review, "MAX_REVIEW_IMAGE_SIDE", 256)
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    model = FakeChartModel()

    result = run_figure(source, tmp_path / "out", model=model)

    assert result["status"] == "exported", result
    sheet = next(Path(result["run_dir"]).glob("grounding/calibration_review.png"))
    context = json.loads(sheet.with_suffix(".json").read_text())
    with Image.open(sheet) as image:
        assert max(image.size) == 256
        assert context["contact_sheet_resize"]["sent_size"] == list(image.size)
        assert context["contact_sheet_resize"]["original_size"] != list(image.size)
        for view in context["views"]:
            x0, y0, x1, y1 = view["sheet_bbox_px"]
            left, top, right, bottom = view["source_bbox_px"]
            sx, sy = view["scale_xy"]
            assert 0 <= x0 < x1 <= image.width
            assert 0 <= y0 < y1 <= image.height
            assert left + (x1 - x0) / sx == pytest.approx(right)
            assert top + (y1 - y0) / sy == pytest.approx(bottom)


def test_compile_rejects_unreviewed_or_stale_calibration():
    for state in (
        {},
        {
            "calibration_revision": 2,
            "confirmed_calibration_revision": 1,
        },
    ):
        with pytest.raises(ValueError, match="visual approval"):
            stages.compile_chart_plan(state)
