import io
import json
import tarfile

import pytest
from PIL import Image

from chart_annotator import wpd
from chart_annotator.calibration import fit_bindings
from chart_annotator.domain.models import (
    AxisPlan,
    CalibrationBinding,
    ChartPlan,
    DatasetPlan,
    SourceAsset,
    TextObservation,
    TickCandidate,
    ValidationIssue,
)
from chart_annotator.domain.workflow import (
    AxisBinding,
    Bindings,
    DirectionBinding,
    Evidence,
    Geometry,
    Grounding,
    Spine,
    Structure,
    TickSelection,
)
from chart_annotator.stages import retain_complete_axes


def evidence_case(scale="linear", categorical=False):
    spines = [
        Spine(
            id="sx",
            orientation="horizontal",
            bbox=(50, 299, 351, 302),
            coordinate=300,
            width=3,
            strength=900,
        ),
        Spine(
            id="sy",
            orientation="vertical",
            bbox=(49, 50, 52, 301),
            coordinate=50,
            width=3,
            strength=750,
        ),
    ]
    ticks, texts, selections = [], [], {}
    for direction in ("x", "y"):
        selected = []
        for i in range(4):
            value = 10**i if scale == "log" and direction == "x" else i * 10
            tid = f"t{direction}{i}"
            oid = f"o{direction}{i}"
            point = (50 + 100 * i, 300) if direction == "x" else (50, 300 - 80 * i)
            ticks.append(
                TickCandidate(
                    id=tid,
                    spine_id="s" + direction,
                    intersection_px=point,
                    text_observation_ids=[oid],
                    status="pdf_only",
                )
            )
            texts.append(
                TextObservation(
                    id=oid,
                    source="pdf_text_layer",
                    raw_text=str(value),
                    normalized_text=str(value),
                    method="synthetic",
                    numeric_value=value,
                    bbox_px=(point[0] - 5, 310, point[0] + 5, 320)
                    if direction == "x"
                    else (25, point[1] - 5, 40, point[1] + 5),
                )
            )
            selected.append(TickSelection(tick_id=tid, text_id=oid))
        selections[direction] = DirectionBinding(
            spine_id="s" + direction,
            ticks=selected,
            calibration_tick_ids=(selected[0].tick_id, selected[-1].tick_id),
        )
    axis = AxisPlan(
        id="a",
        display_name="axis",
        axis_type="categorical_x_numeric_y" if categorical else "xy",
        x_scale="categorical" if categorical else scale,
        categories=["A", "B"] if categorical else [],
    )
    structure = Structure(
        axes=[axis],
        datasets=[
            DatasetPlan(
                id="d", axis_id="a", display_name="observations", kind="scatter"
            )
        ],
    )
    bindings = Bindings(
        axes=[
            AxisBinding(
                axis_id="a",
                x=None if categorical else selections["x"],
                y=selections["y"],
            )
        ]
    )
    evidence = Evidence(
        source_id="synthetic",
        geometry=Geometry(
            image_size=(400, 400),
            preprocessing="synthetic",
            spines=spines,
            ticks=ticks,
        ),
        texts=texts,
    )
    return structure, bindings, evidence


@pytest.mark.parametrize(
    "scale,categorical", [("linear", False), ("log", False), ("linear", True)]
)
def test_fit_and_empty_tar_roundtrip(tmp_path, scale, categorical):
    structure, bindings, evidence = evidence_case(scale, categorical)
    calibrated, issues = fit_bindings(structure, bindings, evidence)
    assert issues == []
    assert all(f.max_residual_px < 1e-9 for f in calibrated.fits)
    axis = structure.axes[0]
    axis.calibration = {
        f.direction: [
            CalibrationBinding(
                tick_id=p.tick_id, value=p.value, text_observation_ids=[p.text_id]
            )
            for p in f.points
        ]
        for f in calibrated.fits
    }
    dataset = structure.datasets[0]
    dataset.kind = "point_group"
    dataset.group_names = ["Upper", "Average", "Lower"]
    plan = ChartPlan(
        source_asset=SourceAsset(source_id="synthetic", path="image.png", kind="image"),
        axes=[axis],
        datasets=[dataset],
    )
    image = tmp_path / "image.png"
    Image.new("RGB", (400, 400), "white").save(image)
    output = tmp_path / "chart.tar"
    wpd.export(output, plan, calibrated, image)
    assert wpd.validate(output, plan, calibrated, image)["points"] == 0
    loaded, pixels = wpd.read(output)
    assert loaded["datasetColl"][0]["groupNames"] == ["Upper", "Average", "Lower"]
    assert loaded["axesColl"][0]["type"] == ("BarAxes" if categorical else "XYAxes")
    assert pixels == image.read_bytes()
    Image.new("RGB", (400, 400), "black").save(image)
    with pytest.raises(ValueError, match="roundtrip"):
        wpd.validate(output, plan, calibrated, image)


def test_shared_tick_alternative_requires_unique_more_support():
    structure, bindings, evidence = evidence_case()
    duplicate = evidence.texts[0].model_copy(
        update={"id": "alternative", "numeric_value": 110.0}
    )
    evidence.texts.append(duplicate)
    evidence.geometry.ticks[0].text_observation_ids.append("alternative")
    calibrated, issues = fit_bindings(structure, bindings, evidence)
    assert not issues and len(calibrated.fits) == 2
    bindings.axes[0].x.ticks[0].text_id = "alternative"
    _, issues = fit_bindings(structure, bindings, evidence)
    assert any(i.code == "grounded_anchor_conflict" for i in issues)
    structure, bindings, evidence = evidence_case()
    bindings.axes[0].x.ticks.pop(1)
    _, issues = fit_bindings(structure, bindings, evidence)
    assert not issues


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda b, e: setattr(b.axes[0].x, "spine_id", "sy"), "spine_mismatch"),
        (
            lambda b, e: setattr(b.axes[0].x.ticks[0], "text_id", "oy0"),
            "invalid_tick_reference",
        ),
        (
            lambda b, e: setattr(e.geometry.spines[0], "coordinate", 200),
            "invalid_tick_reference",
        ),
        (
            lambda b, e: setattr(e.texts[1], "numeric_value", 1e7),
            "grounded_anchor_conflict",
        ),
        (
            lambda b, e: setattr(
                e.geometry.ticks[3],
                "intersection_px",
                e.geometry.ticks[0].intersection_px,
            ),
            "grounded_anchor_conflict",
        ),
    ],
)
def test_invalid_calibration_is_blocked(mutation, code):
    structure, bindings, evidence = evidence_case()
    mutation(bindings, evidence)
    _, issues = fit_bindings(structure, bindings, evidence)
    assert code in {i.code for i in issues}


def test_log_zero_is_blocked_and_selected_readings_disambiguate_shared_ticks():
    structure, bindings, evidence = evidence_case("log")
    evidence.texts[0].numeric_value = 0
    _, issues = fit_bindings(structure, bindings, evidence)
    assert "nonpositive_log" in {i.code for i in issues}
    structure, bindings, evidence = evidence_case()
    # Two equally supported complete numeric scales at the same physical ticks.
    for i in range(4):
        old = evidence.texts[i]
        new = old.model_copy(
            update={"id": "alt" + old.id, "numeric_value": old.numeric_value + 100}
        )
        evidence.texts.append(new)
        evidence.geometry.ticks[i].text_observation_ids.append(new.id)
    _, issues = fit_bindings(structure, bindings, evidence)
    assert not issues


def test_reader_rejects_traversal_and_links(tmp_path):
    for name in ("../bad", "/absolute", "chart/../bad"):
        path = tmp_path / "bad.tar"
        with tarfile.open(path, "w") as t:
            entry = tarfile.TarInfo(name)
            entry.size = 2
            t.addfile(entry, io.BytesIO(b"{}"))
        with pytest.raises(ValueError, match="member"):
            wpd.read(path)


def test_grounding_accepts_normalized_hints_but_rejects_raw_pixels_and_data():
    from pydantic import ValidationError

    from chart_annotator.domain.workflow import Grounding

    structure, bindings, _ = evidence_case()
    grounding = {
        "axes": [
            {
                "axis_id": "a",
                "y": {
                    "anchors": [
                        {"visible_label": "30", "value": 30, "point_2d": [100, 200]},
                        {"visible_label": "20", "value": 20, "point_2d": [100, 500]},
                        {"visible_label": "10", "value": 10, "point_2d": [100, 800]},
                    ],
                },
            }
        ]
    }
    assert Grounding.model_validate(grounding).coordinate_system == "normalized_0_1000"
    with pytest.raises(ValidationError):
        Grounding.model_validate_json(bindings.model_dump_json())
    grounding["axes"][0]["y"]["anchors"][0]["point_2d"] = [100, 1200]
    with pytest.raises(ValidationError):
        Grounding.model_validate(grounding)

    grounding["axes"][0]["y"]["anchors"][0]["point_2d"] = [100, 200]
    grounding["axes"][0]["y"]["anchors"] = grounding["axes"][0]["y"]["anchors"][:2]
    assert len(Grounding.model_validate(grounding).axes[0].y.anchors) == 2
    grounding["axes"][0]["y"]["anchors"] = grounding["axes"][0]["y"]["anchors"][:1]
    with pytest.raises(ValidationError):
        Grounding.model_validate(grounding)

    grounding["axes"][0]["y"]["anchors"] = [
        {"visible_label": str(i), "value": i, "point_2d": [100, i * 100]}
        for i in range(5)
    ]
    with pytest.raises(ValidationError):
        Grounding.model_validate(grounding)

    value = json.loads(structure.model_dump_json())
    value["datasets"][0]["data"] = [{"x": 1, "y": 2}]
    with pytest.raises(ValidationError):
        Structure.model_validate(value)


def test_unbound_tick_and_numeric_label_without_intersection_are_auxiliary():
    structure, bindings, evidence = evidence_case()
    bindings.axes[0].x.ticks.pop(1)
    _, issues = fit_bindings(structure, bindings, evidence)
    assert not issues
    structure, bindings, evidence = evidence_case()
    evidence.texts.append(
        TextObservation(
            id="orphan",
            source="pdf_text_layer",
            raw_text="15",
            normalized_text="15",
            numeric_value=15,
            method="synthetic",
            bbox_px=(190, 310, 210, 320),
        )
    )
    _, issues = fit_bindings(structure, bindings, evidence)
    assert not issues


def test_stale_provisional_endpoint_does_not_block_wider_real_ticks():
    structure, bindings, evidence = evidence_case()
    bindings.axes[0].x.ticks = bindings.axes[0].x.ticks[1:]
    calibrated, issues = fit_bindings(structure, bindings, evidence)
    assert not issues
    x_fit = next(fit for fit in calibrated.fits if fit.direction == "x")
    assert {point.value for point in x_fit.endpoints} == {10, 30}


def test_widest_real_ticks_replace_provisional_pair_and_exclude_projection():
    structure, bindings, evidence = evidence_case()
    bindings.axes[0].x.calibration_tick_ids = ("tx2", "tx3")
    calibrated, issues = fit_bindings(structure, bindings, evidence)
    assert not issues
    x_fit = next(fit for fit in calibrated.fits if fit.direction == "x")
    assert {point.value for point in x_fit.endpoints} == {0, 30}

    projected_id = "local_0_x_label_0"
    evidence.geometry.ticks[0].id = projected_id
    bindings.axes[0].x.ticks[0].tick_id = projected_id
    bindings.axes[0].x.interval_px = (0, 500)
    calibrated, issues = fit_bindings(structure, bindings, evidence)
    assert not issues
    x_fit = next(fit for fit in calibrated.fits if fit.direction == "x")
    assert {point.value for point in x_fit.endpoints} == {10, 30}


def test_grounded_validation_allows_small_scanned_line_drift():
    structure, bindings, evidence = evidence_case()
    # A scanned stroke has uncertainty on both sides of its measured centre.
    # Four pixels of endpoint-map residual is acceptable for a three-pixel line.
    for i, delta in enumerate([-2, 2, 2, -2]):
        tick = evidence.geometry.ticks[i]
        tick.intersection_px = (
            tick.intersection_px[0] + delta,
            tick.intersection_px[1],
        )
    _, issues = fit_bindings(structure, bindings, evidence)
    assert not issues


def test_grounded_validation_rejects_drift_beyond_two_line_widths():
    structure, bindings, evidence = evidence_case()
    for i, delta in enumerate([-4, 4, 4, -4]):
        tick = evidence.geometry.ticks[i]
        tick.intersection_px = (
            tick.intersection_px[0] + delta,
            tick.intersection_px[1],
        )
    _, issues = fit_bindings(structure, bindings, evidence)
    assert "grounded_anchor_conflict" in {i.code for i in issues}


def test_failed_axis_is_removed_without_blocking_independent_calibration():
    structure, bindings, evidence = evidence_case()
    calibrated, issues = fit_bindings(structure, bindings, evidence)
    assert not issues
    failed = structure.axes[0].model_copy(
        update={"id": "failed", "display_name": "failed axis"}, deep=True
    )
    structure.axes.append(failed)
    bindings.axes.append(
        bindings.axes[0].model_copy(update={"axis_id": "failed"}, deep=True)
    )
    grounding = Grounding.model_validate(
        {
            "axes": [
                {
                    "axis_id": axis_id,
                    "x": {
                        "anchors": [
                            {
                                "visible_label": str(value),
                                "value": value,
                                "point_2d": [100 + value, 800],
                            }
                            for value in (0, 10, 20)
                        ]
                    },
                    "y": {
                        "anchors": [
                            {
                                "visible_label": str(value),
                                "value": value,
                                "point_2d": [100, 800 - value],
                            }
                            for value in (0, 10, 20)
                        ]
                    },
                }
                for axis_id in ("a", "failed")
            ]
        }
    )
    failure = ValidationIssue(
        code="insufficient_snapped_ticks",
        message="failed",
        evidence_ids=["failed"],
    )
    unresolved = ValidationIssue(
        code="grounding_unresolved",
        message="one model Axis was unresolved",
    )

    kept, grounding, bindings, calibrated, blocking, skipped = retain_complete_axes(
        structure, grounding, bindings, calibrated, [failure, unresolved]
    )
    assert [axis.id for axis in kept.axes] == ["a"]
    assert [axis.axis_id for axis in grounding.axes] == ["a"]
    assert [axis.axis_id for axis in bindings.axes] == ["a"]
    assert {fit.axis_id for fit in calibrated.fits} == {"a"}
    assert not blocking
    assert skipped == [failure, unresolved]
