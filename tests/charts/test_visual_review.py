import base64
import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError
from test_workflow_e2e import FakeChartModel, chart_pdf

from chart_annotator import calibration_review, stages
from chart_annotator.domain.workflow import VisualReview
from chart_annotator.qwen import ModelReply
from chart_annotator.runner import run_figure


class ReviewModel(FakeChartModel):
    def __init__(self, verdicts=("consistent",), partial=False):
        super().__init__()
        self.verdicts = iter(verdicts)
        self.images = []
        self.partial = partial

    def complete(self, messages):
        reply = super().complete(messages)
        payload = self.calls[-1]
        output = json.loads(reply.text)
        if self.partial and len(self.calls) == 2:
            output["groups"][0]["x"][0]["visible_label"] = "999"
            output["groups"][0]["x"][0]["value"] = 999
        if payload["schema"]["title"] == "VisualReview":
            self.images.append(messages[-1]["content"][1]["image_url"]["url"])
            verdict = next(self.verdicts)
            if verdict == "missing":
                output["axes"] = []
            elif verdict == "duplicate":
                output["axes"] *= 2
            else:
                output["axes"][0]["status"] = verdict
                if verdict != "consistent":
                    output["axes"][0]["issues"] = [
                        "Wrong lower-axis range"
                        if verdict == "wrong"
                        else "Label unreadable"
                    ]
                if verdict == "wrong":
                    corrected = payload["grounding"]["axes"][0]
                    # A changed search center forces a genuinely new snap and review.
                    corrected["x"]["anchors"][0]["point_2d"][0] += 1
                    output["axes"][0]["corrected_grounding"] = corrected
        return ModelReply(json.dumps(output), "fake", 0, "stop")


@pytest.mark.parametrize(
    "verdicts,status,reviews",
    [
        (("consistent",), "exported", 1),
        (("unclear",), "needs_resolution", 1),
        (("wrong", "consistent"), "exported", 2),
        (("wrong", "wrong"), "needs_resolution", 2),
        (("missing",), "failed", 1),
        (("duplicate",), "failed", 1),
    ],
)
def test_visual_gate_and_one_correction(tmp_path, verdicts, status, reviews):
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    model = ReviewModel(verdicts)
    result = run_figure(source, tmp_path / "out", model=model)
    assert result["status"] == status, result
    assert len(model.images) == reviews
    directory = Path(result["run_dir"])
    assert bool(list(directory.rglob("chart.tar"))) == (status == "exported")
    sheets = sorted(directory.glob("attempts/*/calibration_review.png"))
    assert len(sheets) == reviews
    for sheet, sent in zip(sheets, model.images, strict=True):
        assert base64.b64decode(sent.split(",", 1)[1]) == sheet.read_bytes()
        assert sheet.read_bytes() != (directory / "render/figure.png").read_bytes()
        context = json.loads(sheet.with_suffix(".json").read_text())
        assert any(
            p["hint_px"] is None for p in context["points"]
        )  # extra ticks reviewed
        with Image.open(sheet) as image:
            for view in context["views"]:
                x0, y0, x1, y1 = view["sheet_bbox_px"]
                left, top, right, bottom = view["source_bbox_px"]
                sx, sy = view["scale_xy"]
                assert 0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height
                assert left + (x1 - x0) / sx == pytest.approx(right)
                assert top + (y1 - y0) / sy == pytest.approx(bottom)
    if reviews == 2:
        calls = [c for c in model.calls if c["schema"]["title"] == "VisualReview"]
        assert calls[0]["review_context"]["correction_available"] is True
        assert calls[1]["review_context"]["correction_available"] is False
        assert sheets[0].read_bytes() != sheets[1].read_bytes()


def test_one_bad_anchor_is_visible_but_does_not_block_two_point_mapping(tmp_path):
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    model = ReviewModel(partial=True)
    result = run_figure(source, tmp_path / "out", model=model)
    assert result["status"] == "exported", result
    assert len(model.images) == 1
    directory = Path(result["run_dir"])
    assert list(directory.rglob("chart.tar"))
    for path in directory.glob("attempts/*/calibration_review.json"):
        points = json.loads(path.read_text())["points"]
        assert any(p["snapped_px"] is None for p in points)
        assert any(p["snapped_px"] is not None for p in points)
        assert json.loads((path.parent / "calibration.json").read_text())["fits"]


def test_review_sheet_size_cap_preserves_view_transforms(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_review, "MAX_REVIEW_IMAGE_SIDE", 256)
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    model = ReviewModel()

    result = run_figure(source, tmp_path / "out", model=model)

    assert result["status"] == "exported", result
    sheet = next(Path(result["run_dir"]).glob("attempts/*/calibration_review.png"))
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
            "visual_review_status": "consistent",
            "reviewed_calibration": "old.json",
            "calibration_plan": "new.json",
        },
    ):
        with pytest.raises(ValueError, match="visual approval"):
            stages.compile_chart_plan(state)


@pytest.mark.parametrize(
    "axis",
    [
        {"status": "consistent", "issues": ["wrong axis"], "corrected_grounding": None},
        {"status": "wrong", "issues": ["wrong axis"], "corrected_grounding": None},
        {"status": "unclear", "issues": [], "corrected_grounding": None},
        {
            "status": "consistent",
            "issues": [],
            "corrected_grounding": None,
            "confidence": 0.9,
        },
    ],
)
def test_review_schema_rejects_contradictory_verdicts(axis):
    with pytest.raises(ValidationError):
        VisualReview.model_validate(
            {"axes": [{"axis_id": "a", **axis}], "unresolved": []}
        )
