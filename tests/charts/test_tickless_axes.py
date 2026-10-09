import json
import math
from pathlib import Path

import pytest
from PIL import Image
from test_calibration_export import evidence_case
from test_workflow_e2e import FakeChartModel, tool_reply

from chart_annotator import tick_ocr
from chart_annotator.calibration import fit_bindings
from chart_annotator.domain.workflow import Geometry, TickGrounding
from chart_annotator.runner import run_figure
from chart_annotator.snapping import _label_spacing_selection
from chart_annotator.wpd import read


def spacing_case(count=2, direction="y", scale="linear"):
    structure, bindings, evidence = evidence_case(
        categorical=direction == "y", scale=scale
    )
    axis = structure.axes[0]
    if direction == "x":
        axis.y_scale = "categorical"
        axis.axis_type = "numeric_x_categorical_y"
        bindings.axes[0].y = None
    axis.y_scale = scale if direction == "y" else axis.y_scale
    selection = getattr(bindings.axes[0], direction)
    selection.ticks = selection.ticks[:count]
    selection.grounded_tick_ids = [item.tick_id for item in selection.ticks[:2]]
    selection.calibration_tick_ids = (
        selection.ticks[0].tick_id,
        selection.ticks[-1].tick_id,
    )
    ticks = {t.id: t for t in evidence.geometry.ticks}
    texts = {t.id: t for t in evidence.texts}
    for index, item in enumerate(selection.ticks):
        tick, text = ticks[item.tick_id], texts[item.text_id]
        if scale == "log":
            text.numeric_value = 10**index
        coordinate = tick.intersection_px[0 if direction == "x" else 1]
        box = list(text.bbox_px)
        along = 0 if direction == "x" else 1
        box[along], box[along + 2] = coordinate - 3, coordinate + 7
        text.bbox_px = tuple(box)  # All label centers have the same +2 px offset.
        if index:
            tick.provenance = "label_center"
            pixel = list(tick.intersection_px)
            pixel[along] += 2
            tick.intersection_px = tuple(pixel)
    return structure, bindings, evidence


@pytest.mark.parametrize(
    "direction,scale,count",
    [("y", "linear", 2), ("y", "linear", 3), ("x", "linear", 2), ("y", "log", 3)],
)
def test_one_real_station_keeps_absolute_position_with_two_or_more_labels(
    direction, scale, count
):
    structure, bindings, evidence = spacing_case(count, direction, scale)
    calibrated, issues = fit_bindings(structure, bindings, evidence)
    assert not issues
    fit = calibrated.fits[0]
    assert fit.calibration_method == "anchored_label_spacing"
    assert fit.intercept == pytest.approx(300 if direction == "y" else 50)
    assert fit.slope == pytest.approx(
        (-80 if scale == "log" else -8) if direction == "y" else 10
    )
    real = next(
        t
        for t in evidence.geometry.ticks
        if t.id == getattr(bindings.axes[0], direction).ticks[0].tick_id
    )
    point = next(p for p in fit.points if p.tick_id == real.id)
    assert point.pixel == real.intersection_px
    index = 0 if direction == "x" else 1
    for point in fit.endpoints:
        value = math.log10(point.value) if scale == "log" else point.value
        assert point.pixel[index] == pytest.approx(fit.slope * value + fit.intercept)


@pytest.mark.parametrize(
    "case",
    [
        "no_real",
        "one_label",
        "mixed_source",
        "duplicate",
        "extra_conflict",
        "log_zero",
        "two_real_conflict",
    ],
)
def test_label_spacing_rejects_insufficient_or_conflicting_evidence(case):
    structure, bindings, evidence = spacing_case(
        3 if case in {"extra_conflict", "two_real_conflict"} else 2
    )
    selection = bindings.axes[0].y
    ticks = {t.id: t for t in evidence.geometry.ticks}
    texts = {t.id: t for t in evidence.texts}
    if case == "no_real":
        ticks[selection.ticks[0].tick_id].provenance = "label_center"
    elif case == "one_label":
        selection.ticks = selection.ticks[1:]
    elif case == "mixed_source":
        texts[selection.ticks[1].text_id].source = "rapidocr_tick_crop"
    elif case == "duplicate":
        texts[selection.ticks[1].text_id].numeric_value = 0
    elif case == "extra_conflict":
        text = texts[selection.ticks[-1].text_id]
        box = list(text.bbox_px)
        box[1] += 25
        box[3] += 25
        text.bbox_px = tuple(box)
    elif case == "log_zero":
        structure.axes[0].y_scale = "log"
    elif case == "two_real_conflict":
        ticks[selection.ticks[1].tick_id].provenance = "tick_intersection"
    _, issues = fit_bindings(structure, bindings, evidence)
    assert issues


@pytest.mark.parametrize(
    "case",
    [
        "two_labels",
        "both_sides",
        "two_columns",
        "source_duplicates",
        "outside_range",
        "no_real",
        "other_unbound_column",
    ],
)
def test_label_selection_preserves_ownership_and_physical_label_count(case):
    _, bindings, evidence = spacing_case(2)
    selection = bindings.axes[0].y
    ticks = {t.id: t for t in evidence.geometry.ticks}
    texts = {t.id: t for t in evidence.texts}
    labels = [texts[item.text_id] for item in selection.ticks]
    real = ticks[selection.ticks[0].tick_id]
    spine = next(s for s in evidence.geometry.spines if s.id == "sy")
    anchored = [
        (
            TickGrounding(visible_label=str(i * 10), value=i * 10, point_2d=(0, 0)),
            (50, 300 - 80 * i),
        )
        for i in range(2)
    ]
    if case in {"both_sides", "two_columns"}:
        labels += [
            text.model_copy(
                update={
                    "id": text.id + "other",
                    "bbox_px": (60, text.bbox_px[1], 70, text.bbox_px[3])
                    if case == "both_sides"
                    else (5, text.bbox_px[1], 15, text.bbox_px[3]),
                }
            )
            for text in labels
        ]
    elif case == "source_duplicates":
        labels = [
            labels[0],
            labels[0].model_copy(
                update={"id": "ocr_duplicate", "source": "rapidocr_tick_crop"}
            ),
        ]
    elif case == "outside_range":
        labels.append(
            labels[1].model_copy(
                update={
                    "id": "outside",
                    "numeric_value": 150,
                    "bbox_px": (25, 100, 40, 110),
                }
            )
        )
    elif case == "other_unbound_column":
        labels += [
            text.model_copy(
                update={
                    "id": text.id + "unbound",
                    "bbox_px": (41, text.bbox_px[1] - 20, 45, text.bbox_px[3] - 20),
                }
            )
            for text in labels
        ]
    if case in {"both_sides", "two_columns"}:
        with pytest.raises(ValueError, match="Competing"):
            _label_spacing_selection(anchored, [real], labels, "y", spine, 60, "local")
    else:
        result = _label_spacing_selection(
            anchored,
            [] if case == "no_real" else [real],
            labels,
            "y",
            spine,
            60,
            "local",
        )
        if case in {"source_duplicates", "no_real"}:
            assert result is None
        else:
            assert len(result[0].ticks) == 2
            assert result[0].interval_px == (222, 302)


def test_tick_ocr_reads_full_grounded_range_once(tmp_path, monkeypatch):
    _, bindings, evidence = spacing_case(2)
    real = next(
        t
        for t in evidence.geometry.ticks
        if t.id == bindings.axes[0].y.ticks[0].tick_id
    )
    spine = next(s for s in evidence.geometry.spines if s.id == "sy")
    calls = []

    class Reader:
        def __call__(self, crop, **kwargs):
            calls.append(crop.shape)
            return type("Result", (), {"word_results": []})()

    monkeypatch.setattr(tick_ocr, "RapidOCR", Reader)
    image = tmp_path / "image.png"
    Image.new("RGB", (400, 400), "white").save(image)
    found = Geometry(
        image_size=(400, 400), preprocessing="test", spines=[spine], ticks=[real]
    )
    tick_ocr.read_ticks(
        image,
        found,
        evidence.texts,
        "test",
        tmp_path / "ocr",
        grounded_range=("y", 100, 300),
    )
    raw = json.loads((tmp_path / "ocr/raw.json").read_text())
    assert len(calls) == 1
    assert raw[0]["crop_bbox"][1] <= 100 and raw[0]["crop_bbox"][3] >= 300
    assert raw[0]["crop_bbox"][2] - raw[0]["crop_bbox"][0] < 100


@pytest.mark.parametrize("case", ["layout", "numeric_fit"])
def test_unusable_pdf_group_uses_complete_ocr_group_without_mixing(case):
    _, bindings, evidence = spacing_case(3)
    selection = bindings.axes[0].y
    ticks = {t.id: t for t in evidence.geometry.ticks}
    texts = {t.id: t for t in evidence.texts}
    labels = [texts[item.text_id] for item in selection.ticks]
    ocr = [
        text.model_copy(
            update={"id": text.id + "ocr", "source": "rapidocr_tick_crop"}, deep=True
        )
        for text in labels
    ]
    box = list(labels[1].bbox_px)
    if case == "layout":
        box[3] += 12
    else:
        box[1] += 25
        box[3] += 25
    labels[1].bbox_px = tuple(box)
    spine = next(s for s in evidence.geometry.spines if s.id == "sy")
    anchored = [
        (
            TickGrounding(visible_label=str(i * 10), value=i * 10, point_2d=(0, 0)),
            (50, 300 - 80 * i),
        )
        for i in range(3)
    ]
    rejected = []
    result = _label_spacing_selection(
        anchored,
        [ticks[selection.ticks[0].tick_id]],
        labels + ocr,
        "y",
        spine,
        60,
        "local",
        rejections=rejected,
    )
    assert result is not None
    assert {text.source for _, text in result[1]} == {"rapidocr_tick_crop"}
    assert [entry["source"] for entry in rejected] == ["pdf_text_layer"]
    with pytest.raises(ValueError, match="No text source supports"):
        _label_spacing_selection(
            anchored,
            [ticks[selection.ticks[0].tick_id]],
            labels,
            "y",
            spine,
            60,
            "local",
        )


def test_real_tickless_figure_fake_model_exports_both_axes_with_confirmation(tmp_path):
    def transform(model, call, reply):
        name = call["tools"][0]["function"]["name"]
        if name == "submit_axes":
            return tool_reply(
                name,
                {
                    "groups": [
                        {
                            "x": {
                                "name": "Property",
                                "scale": "categorical",
                                "categories": ["Ftu", "Fty", "e", "RA"],
                            },
                            "ys": [
                                {"name": "Strength (ksi)", "scale": "linear"},
                                {"name": "Ductility (percent)", "scale": "linear"},
                            ],
                        }
                    ]
                },
                f"call_{len(model.calls)}",
            )
        if name == "submit_grounding" and not any(
            m["role"] == "tool" for m in call["messages"]
        ):

            def anchor(value, x, y):
                return {"visible_label": str(value), "value": value, "point_2d": [x, y]}

            return tool_reply(
                name,
                {
                    "groups": [
                        {
                            "x": [],
                            "ys": [
                                [
                                    anchor(150, 98.5, 33.1),
                                    anchor(100, 98.5, 186.1),
                                    anchor(50, 98.5, 346.9),
                                    anchor(0, 98.5, 506.7),
                                ],
                                [
                                    anchor(0, 897.5, 506.7),
                                    anchor(10, 897.5, 427.9),
                                    anchor(20, 897.5, 346.9),
                                    anchor(30, 897.5, 266.3),
                                ],
                            ],
                        }
                    ]
                },
                f"call_{len(model.calls)}",
            )
        if name == "submit_datasets":
            return tool_reply(
                name,
                {
                    "datasets": [
                        {
                            "axis": f"axis_000_00{i}",
                            "name": f"{property_} | Condition {j} | solid bar",
                            "kind": "bar",
                        }
                        for i, property_ in enumerate(
                            ("Strength (ksi)", "Ductility (percent)")
                        )
                        for j in range(5)
                    ]
                },
                f"call_{len(model.calls)}",
            )
        return reply

    model = FakeChartModel(transform)
    source = Path(__file__).parent / "fixtures/charts/Figure_3.2.1.19/source.pdf"
    result = run_figure(source, tmp_path / "out", model=model, datasets=True)
    assert result["status"] == "exported", result
    assert len(model.calls) == 4 and result["grounding_submissions"] == 1
    root = Path(result["run_dir"])
    calibration = json.loads((root / "plan/calibration.json").read_text())
    right = next(f for f in calibration["fits"] if f["axis_id"] == "axis_000_001")
    assert right["calibration_method"] == "anchored_label_spacing"
    assert right["intercept"] == pytest.approx(640, abs=1)
    assert {p["value"] for p in right["points"]} >= {0, 10, 20, 30}
    assert not {100, 125, 150}.intersection(p["value"] for p in right["points"])
    context = json.loads((root / "grounding/calibration_review.json").read_text())
    fit_points = {p["tick_id"]: p["pixel"] for p in right["points"]}
    for point in context["points"]:
        if point["axis_id"] == right["axis_id"] and point["tick_id"] in fit_points:
            assert point["snapped_px"] == fit_points[point["tick_id"]]
    assert any(p["snap_method"] == "anchored_label_spacing" for p in context["points"])
    assert any(
        m["method"] == "anchored_label_spacing" for m in context["calibration_methods"]
    )
    project, _ = read(root / "output/chart.tar")
    assert len(project["axesColl"]) == 2 and len(project["datasetColl"]) == 10
