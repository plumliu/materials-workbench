import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from test_workflow_e2e import FakeChartModel, chart_pdf

from chart_annotator.calibration import fit_bindings
from chart_annotator.domain.models import AxisPlan, DatasetPlan, TickCandidate
from chart_annotator.domain.workflow import (
    Evidence,
    Geometry,
    Grounding,
    GroundingResponse,
    Spine,
    Structure,
)
from chart_annotator.figure import ingest_figure, render_figure
from chart_annotator.geometry import detect
from chart_annotator.semantic import expand_grounding
from chart_annotator.snapping import _merge_collinear_spines, snap_grounding, to_pixels
from chart_annotator.text_evidence import extract_pdf_text


def prepare(source, tmp_path):
    directory, asset = ingest_figure(source, tmp_path)
    asset, image = render_figure(directory, asset)
    texts = extract_pdf_text(directory, asset).observations
    found = detect(image, texts)
    return image, Evidence(source_id=asset.source_id, geometry=found, texts=texts)


def synthetic(tmp_path):
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    image, evidence = prepare(source, tmp_path / "input")
    structure = Structure(
        axes=[
            AxisPlan(
                id="a",
                display_name="Time - Stress",
                axis_type="xy",
                x_label="Time",
                y_label="Stress",
            )
        ],
        datasets=[
            DatasetPlan(id="d", axis_id="a", display_name="Stress", kind="scatter")
        ],
    )
    model = FakeChartModel()
    reply = model.complete(
        [
            {
                "content": [
                    {
                        "text": json.dumps(
                            {
                                "evidence": {},
                                "structure": structure.model_dump(),
                                "schema": Grounding.model_json_schema(),
                            }
                        )
                    }
                ]
            }
        ]
    )
    grounding = expand_grounding(
        GroundingResponse.model_validate_json(reply.text), structure
    )
    return image, evidence, structure, grounding


def test_normalized_conversion_uses_full_image_width_and_height():
    assert to_pixels((250, 750), (1600, 800)) == (400, 600)
    assert to_pixels((1000, 1000), (1600, 800)) == (1600, 800)
    assert to_pixels((250, 750), (800, 1600)) == (200, 1200)
    with pytest.raises(ValueError):
        to_pixels((1001, 500), (1600, 800))


def test_fragmented_local_axis_spine_is_merged_before_selection():
    found = Geometry(
        image_size=(80, 500),
        preprocessing="synthetic",
        spines=[
            Spine(
                id=f"s{i}",
                orientation="vertical",
                bbox=(30, start, 33, end),
                coordinate=31 + i * 0.2,
                width=3,
                strength=100,
            )
            for i, (start, end) in enumerate(((10, 150), (160, 300), (310, 490)))
        ],
        ticks=[
            TickCandidate(
                id=f"t{i}",
                spine_id=f"s{i}",
                intersection_px=(31, 100 + i * 150),
                text_observation_ids=[],
                status="ocr_only",
            )
            for i in range(3)
        ],
    )
    _merge_collinear_spines(found, "vertical", center=31, radius=20)
    vertical = [spine for spine in found.spines if spine.orientation == "vertical"]
    assert len(vertical) == 1
    assert vertical[0].bbox == (30, 10, 33, 490)
    assert {tick.spine_id for tick in found.ticks} == {vertical[0].id}


def test_local_snap_recovers_missing_global_catalog(tmp_path):
    image, evidence, structure, grounding = synthetic(tmp_path)
    evidence.geometry.spines = []
    evidence.geometry.ticks = []
    bindings, local, issues = snap_grounding(
        structure, grounding, evidence, image, tmp_path / "snap"
    )
    assert not issues
    calibrated, issues = fit_bindings(structure, bindings, local)
    assert not issues
    assert [len(f.points) for f in calibrated.fits] == [4, 4]
    events = json.loads((tmp_path / "snap/snap_events.json").read_text())["events"]
    assert all(e["hint_px"] != e["snapped_px"] for e in events)
    assert all(
        p.tick_id.startswith("local_") for f in calibrated.fits for p in f.points
    )
    assert {t.source for t in local.texts} >= {"pdf_text_layer", "rapidocr_tick_crop"}


def test_missing_tick_uses_label_projection_without_moving_two_real_intersections(
    tmp_path,
):
    image, evidence, structure, grounding = synthetic(tmp_path)
    with Image.open(image) as source:
        edited = source.copy()
    draw = ImageDraw.Draw(edited)
    draw.rectangle((64 * 3, 207 * 3, 78 * 3, 214 * 3), fill="white")
    draw.line((70 * 3, 207 * 3, 70 * 3, 214 * 3), fill="black", width=3)
    edited.save(image)
    grounding.axes[0].y.anchors[0].visible_label = "10"
    grounding.axes[0].y.anchors[0].value = 10
    grounding.axes[0].y.anchors[0].point_2d = (1000 * 73 / 440, 1000 * 213 / 390)
    bindings, local, issues = snap_grounding(
        structure, grounding, evidence, image, tmp_path / "snap"
    )
    assert not issues
    calibrated, issues = fit_bindings(structure, bindings, local)
    assert not issues
    y = next(fit for fit in calibrated.fits if fit.direction == "y")
    assert len(y.endpoints) == 2
    assert {point.value for point in y.endpoints} == {0, 20}
    assert {point.value for point in y.points} == {0, 10, 20}
    events = json.loads((tmp_path / "snap/snap_events.json").read_text())["events"]
    projected = next(
        event
        for event in events
        if event["direction"] == "y" and event["visible_label"] == "10"
    )
    assert projected["method"] == "label_projection"
    assert projected["snapped_px"][1] == pytest.approx(630, abs=2)


def test_one_model_value_mismatch_is_ignored_when_two_anchors_remain(tmp_path):
    image, evidence, structure, grounding = synthetic(tmp_path)
    grounding.axes[0].x.anchors[1].value = 11
    bindings, local, issues = snap_grounding(
        structure, grounding, evidence, image, tmp_path / "snap"
    )
    assert not issues
    _, issues = fit_bindings(structure, bindings, local)
    assert not issues


def test_shared_t_grounding_and_independent_reference(tmp_path):
    source = Path(__file__).parent / "fixtures/charts/Figure_3.2.1.7/source.pdf"
    image, evidence = prepare(source, tmp_path / "input")
    structure = Structure(
        axes=[
            AxisPlan(
                id=f"a{i}",
                display_name=f"Property {i}",
                axis_type="categorical_x_numeric_y",
                x_scale="categorical",
                categories=["A", "B", "C", "D", "E"],
            )
            for i in range(2)
        ],
        datasets=[],
    )
    # Historical raw Qwen points (not the corrected points or human TAR).
    pairs = [
        [
            ("140", [110, 240]),
            ("130", [110, 337]),
            ("120", [110, 427]),
            ("110", [110, 513]),
        ],
        [("30", [110, 513]), ("20", [110, 620]), ("10", [110, 705])],
    ]
    grounding = Grounding.model_validate(
        {
            "axes": [
                {
                    "axis_id": f"a{i}",
                    "y": {
                        "anchors": [
                            {
                                "visible_label": label,
                                "value": float(label),
                                "point_2d": point,
                            }
                            for label, point in pair
                        ],
                    },
                }
                for i, pair in enumerate(pairs)
            ]
        }
    )
    evidence.geometry.ticks = []
    bindings, local, issues = snap_grounding(
        structure, grounding, evidence, image, tmp_path / "snap"
    )
    assert not issues, issues
    calibrated, issues = fit_bindings(structure, bindings, local)
    assert not issues, issues
    assert {p.value for p in calibrated.fits[0].points} == {110, 120, 130, 140}
    assert {p.value for p in calibrated.fits[1].points} == {10, 20, 30}
    upper = {point.value: point.pixel for point in calibrated.fits[0].points}
    lower = {point.value: point.pixel for point in calibrated.fits[1].points}
    assert upper[110][1] == pytest.approx(lower[30][1], abs=1)
    # Read human calibration ONLY AFTER snapping/fitting; compare independent maps.
    reference = json.loads(source.with_name("reference.json").read_text())
    measurements = []
    for i, fit in enumerate(calibrated.fits):
        a, b = reference["axes"][i]["calibrationPoints"]
        ref_m = (b["py"] - a["py"]) / (float(b["dy"]) - float(a["dy"]))
        ref_b = a["py"] - ref_m * float(a["dy"])
        raw_values = np.array([float(x[0]) for x in pairs[i]])
        raw_y = np.array(
            [to_pixels(x[1], local.geometry.image_size)[1] for x in pairs[i]]
        )
        raw_m = (raw_y[1] - raw_y[0]) / (raw_values[1] - raw_values[0])
        raw_b = raw_y[0] - raw_m * raw_values[0]
        values = np.array([p.value for p in fit.points])
        before = float(np.mean(abs(raw_m * values + raw_b - (ref_m * values + ref_b))))
        after = float(
            np.mean(abs(fit.slope * values + fit.intercept - (ref_m * values + ref_b)))
        )
        assert after < before
        assert after < 3  # Independent human reference regression, not a runtime gate.
        measurements.append(
            {"axis": fit.axis_id, "raw_mae_px": before, "snapped_mae_px": after}
        )
    (tmp_path / "independent_reference.json").write_text(json.dumps(measurements))


def test_shared_x_is_snapped_once_and_reused(tmp_path):
    image, evidence, structure, grounding = synthetic(tmp_path)
    alias = structure.axes[0].model_copy(
        update={
            "id": "a2",
            "display_name": "Time - Other Stress",
            "shared_x_axis_id": "a",
        }
    )
    structure.axes.append(alias)
    grounding.axes.append(
        grounding.axes[0].model_copy(update={"axis_id": "a2"}, deep=True)
    )

    bindings, local, issues = snap_grounding(
        structure, grounding, evidence, image, tmp_path / "snap"
    )
    assert not issues
    assert bindings.axes[0].x == bindings.axes[1].x
    assert (tmp_path / "snap/local_0_x.png").exists()
    assert not (tmp_path / "snap/local_1_x.png").exists()
    events = json.loads((tmp_path / "snap/snap_events.json").read_text())["events"]
    assert any(event.get("reused_from_axis_id") == "a" for event in events)

    calibrated, issues = fit_bindings(structure, bindings, local)
    assert not issues
    x_fits = [fit for fit in calibrated.fits if fit.direction == "x"]
    assert {fit.axis_id for fit in x_fits} == {"a", "a2"}
    assert x_fits[0].model_copy(update={"axis_id": "a2"}) == x_fits[1]
