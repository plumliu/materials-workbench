import json

import pytest
from pydantic import ValidationError

from chart_annotator.domain.models import AxisPlan, SourceAsset, TextObservation
from chart_annotator.domain.workflow import (
    AxisStructure,
    DatasetStructure,
    DirectionGrounding,
    Evidence,
    Geometry,
    Grounding,
    GroundingResponse,
    Structure,
    TickGrounding,
)
from chart_annotator.qwen import ModelReply
from chart_annotator.semantic import (
    AXIS_EXAMPLE_FILES,
    DATASET_EXAMPLE_FILES,
    PROMPTS,
    dataset_example_output,
    request,
    select_examples,
    validate_axes,
    validate_datasets,
)


def load_example(name: str) -> dict:
    return json.loads(
        (PROMPTS / "examples" / f"{name}.json").read_text(encoding="utf-8")
    )


def evidence_for(example: dict) -> Evidence:
    visible = example.get("input", {}).get("visible_text", [])
    return Evidence(
        source_id="test-source",
        geometry=Geometry(
            image_size=(1200, 2200), preprocessing="test", spines=[], ticks=[]
        ),
        texts=[
            TextObservation(
                id=f"text_{index}",
                source="pdf_text_layer",
                raw_text=text,
                normalized_text=text,
                method="test",
            )
            for index, text in enumerate(visible)
        ],
    )


def test_exactly_eight_reviewed_examples_remain():
    examples = sorted((PROMPTS / "examples").glob("*.json"))
    assert len(examples) == 8
    assert all(
        (PROMPTS / "examples" / load_example(path.stem)["image"]).is_file()
        for path in examples
    )
    assert not (PROMPTS / "ontology.md").exists()
    assert not (PROMPTS / "marker-rules.md").exists()


@pytest.mark.parametrize("filename", AXIS_EXAMPLE_FILES)
def test_axis_few_shots_expand(filename):
    example = load_example(filename.removesuffix(".json"))
    axes = AxisStructure.model_validate(example["axis_output"])
    asset = SourceAsset(source_id="test-source", path=example["image"], kind="image")
    assert axes.expand()
    assert not validate_axes(axes, evidence_for(example), asset)


def test_dual_unit_example_is_two_independent_axes():
    axes = AxisStructure.model_validate(load_example("3_2_7_2_3")["axis_output"])
    assert len(axes.groups) == 2
    assert [group.x.name for group in axes.groups] == [
        "Tensile yield strength - MPa",
        "Tensile yield strength - ksi",
    ]
    assert all(axis.shared_x_axis_id is None for axis in axes.expand())


def test_protocols_reject_legacy_panel_ids():
    with pytest.raises(ValidationError):
        AxisStructure.model_validate(
            {
                "groups": [
                    {
                        "panel_id": "legacy",
                        "x": {"name": "X", "scale": "linear"},
                        "ys": [{"name": "Y", "scale": "linear"}],
                    }
                ]
            }
        )
    with pytest.raises(ValidationError):
        Grounding.model_validate(
            {
                "axes": [
                    {
                        "axis_id": "axis_000_000",
                        "panel_id": "legacy",
                        "x": {
                            "anchors": [
                                {"visible_label": "0", "value": 0, "point_2d": [0, 0]},
                                {
                                    "visible_label": "1",
                                    "value": 1,
                                    "point_2d": [1000, 0],
                                },
                            ]
                        },
                        "y": {
                            "anchors": [
                                {"visible_label": "0", "value": 0, "point_2d": [0, 0]},
                                {
                                    "visible_label": "1",
                                    "value": 1,
                                    "point_2d": [0, 1000],
                                },
                            ]
                        },
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "task,expected_count",
    [("axes", 5), ("axes-repair", 5), ("datasets", 2), ("datasets-repair", 2)],
)
def test_requests_use_bounded_multimodal_examples(tmp_path, task, expected_count):
    example = load_example("3_2_1_7")
    evidence = evidence_for(example)
    image = PROMPTS / "examples" / example["image"]
    axes = AxisStructure.model_validate(example["axis_output"])
    datasets = DatasetStructure.model_validate(dataset_example_output(example))
    structure = Structure(axes=axes.expand(), datasets=[])
    expected = axes if task.startswith("axes") else datasets
    if task.startswith("datasets"):
        evidence.source_id = example["source_ids"][0]

    class RecordingModel:
        def complete(self, messages):
            assert len(messages) == 2 * expected_count + 2
            return ModelReply(expected.model_dump_json(), "fake", 0, "stop")

    result = request(
        RecordingModel(),
        task,
        evidence,
        image,
        tmp_path / task,
        axes
        if task == "axes-repair"
        else structure
        if task.startswith("datasets")
        else None,
    )
    assert result == expected
    snapshot = json.loads(
        (tmp_path / task / "request.json").read_text(encoding="utf-8")
    )
    assert len(snapshot["example_files"]) == expected_count
    assert "examples" not in snapshot
    assert "image_sha256" not in snapshot
    if task.startswith("axes"):
        assert "fig_3_2_1_10.png" not in snapshot["example_files"]
        assert tuple(snapshot["example_files"]) == tuple(
            load_example(name.removesuffix(".json"))["image"]
            for name in AXIS_EXAMPLE_FILES
        )
    if task.startswith("datasets"):
        assert set(snapshot["payload"]) >= {"axes", "schema"}
        assert "structure" not in snapshot["payload"]


def test_dataset_protocol_is_minimal_and_expands_point_groups():
    schema = DatasetStructure.model_json_schema()
    assert set(schema["properties"]) == {"datasets", "unresolved"}
    assert set(schema["$defs"]["DatasetItem"]["properties"]) == {"axis", "name", "kind"}
    proposal = DatasetStructure.model_validate(
        {
            "datasets": [
                {
                    "axis": "axis_000_000",
                    "name": "Stress Average Value + Spread of Value",
                    "kind": "point_group",
                }
            ]
        }
    )
    expanded = proposal.expand()[0]
    assert expanded.group_names == ["upper", "average", "lower"]
    assert expanded.data == []
    assert set(expanded.model_dump()) == {
        "schema_version",
        "id",
        "axis_id",
        "display_name",
        "kind",
        "group_names",
        "data",
    }


def test_dataset_reference_images_match_fixtures():
    repository = PROMPTS.parents[3]
    for json_name, image_name in DATASET_EXAMPLE_FILES.items():
        figure = json_name.removesuffix(".json").replace("_", ".")
        source = (
            repository
            / "tests/charts/fixtures/charts"
            / f"Figure_{figure}"
            / "source.png"
        )
        assert (PROMPTS / "examples" / image_name).read_bytes() == source.read_bytes()


def test_dataset_validation_uses_only_axis_name_and_kind():
    axis = AxisPlan(
        id="axis_000_000",
        display_name="Time - Stress",
        axis_type="xy",
        x_label="Time",
        y_label="Stress",
    )
    asset = SourceAsset(source_id="test", path="chart.png", kind="image")
    good = DatasetStructure.model_validate(
        {
            "datasets": [
                {"axis": axis.id, "name": "Stress | filled circle", "kind": "scatter"}
            ]
        }
    )
    assert not validate_datasets(good, [axis], asset)
    bad = good.model_copy(deep=True)
    bad.datasets[0].name = "Stress"
    assert {issue.code for issue in validate_datasets(bad, [axis], asset)} == {
        "marker_name_incomplete"
    }


def test_binding_examples_keep_normalized_t_intersections(tmp_path):
    example = load_example("3_2_1_1")
    evidence = Evidence(
        source_id=example["source_ids"][0],
        geometry=Geometry(
            image_size=(657, 941), preprocessing="test", spines=[], ticks=[]
        ),
        texts=[],
    )
    structure = AxisStructure.model_validate(example["binding_input"]["structure"])
    expected = GroundingResponse.model_validate(example["binding_output"])

    class RecordingModel:
        def complete(self, messages):
            references = [
                GroundingResponse.model_validate_json(message["content"])
                for message in messages
                if message["role"] == "assistant"
            ]
            assert len(references) == 2
            return ModelReply(expected.model_dump_json(), "fake", 0, "stop")

    assert request(
        RecordingModel(),
        "binding",
        evidence,
        PROMPTS / "examples" / example["image"],
        tmp_path,
        structure,
    ) == __import__(
        "chart_annotator.semantic", fromlist=["expand_grounding"]
    ).expand_grounding(expected, structure)


def test_grounding_requires_two_to_four_anchors():
    anchors = [
        TickGrounding(visible_label="20", value=20, point_2d=(100, 200)),
        TickGrounding(visible_label="0", value=0, point_2d=(100, 800)),
    ]
    assert DirectionGrounding(anchors=anchors).anchors == anchors
    with pytest.raises(ValidationError):
        DirectionGrounding(anchors=anchors[:1])


def test_grouped_grounding_has_one_x_for_multiple_ys():
    structure = AxisStructure.model_validate(
        {
            "groups": [
                {
                    "x": {"name": "Temperature", "scale": "linear"},
                    "ys": [
                        {"name": "Ftu", "scale": "linear"},
                        {"name": "Fty", "scale": "linear"},
                    ],
                }
            ]
        }
    )
    anchors = [
        {"visible_label": "0", "value": 0, "point_2d": [100, 900]},
        {"visible_label": "10", "value": 10, "point_2d": [900, 900]},
    ]
    grouped = GroundingResponse.model_validate(
        {"groups": [{"x": anchors, "ys": [anchors, anchors]}]}
    )
    assert set(grouped.groups[0].model_dump()) == {"x", "ys"}
    assert len(grouped.groups[0].x) == 2
    assert len(grouped.groups[0].ys) == 2

    from chart_annotator.semantic import expand_grounding

    expanded = expand_grounding(grouped, structure)
    assert [axis.axis_id for axis in expanded.axes] == [
        "axis_000_000",
        "axis_000_001",
    ]
    assert expanded.axes[0].x == expanded.axes[1].x


def test_axis_validation_rejects_unit_only_names():
    axes = AxisStructure.model_validate(
        {
            "groups": [
                {
                    "x": {"name": "Time", "scale": "linear"},
                    "ys": [{"name": "ksi", "scale": "linear"}],
                }
            ]
        }
    )
    asset = SourceAsset(source_id="test", path="chart.png", kind="image")
    issues = validate_axes(
        axes,
        Evidence(
            source_id="test",
            geometry=Geometry(
                image_size=(10, 10), preprocessing="test", spines=[], ticks=[]
            ),
            texts=[],
        ),
        asset,
    )
    assert any("unit alone" in issue.message for issue in issues)


@pytest.mark.parametrize("method", ["source", "caption"])
def test_same_figure_is_excluded_without_image_hash(tmp_path, method):
    from PIL import Image

    example = load_example("3_2_1_7")
    evidence = evidence_for(example)
    image = tmp_path / "different.png"
    Image.new("RGB", (80, 80)).save(image)
    if method == "source":
        evidence.source_id = example["source_ids"][0]
    else:
        evidence.texts.append(
            TextObservation(
                id="caption",
                source="pdf_text_layer",
                raw_text="Figure 3.2.1.7",
                normalized_text="Figure 3.2.1.7",
                method="test",
            )
        )
    assert all(
        item["figure_id"] != "3.2.1.7"
        for item in select_examples(evidence, image, task="datasets")
    )


def test_axis_repair_receives_scoped_passes_and_failures(tmp_path):
    from chart_annotator.stages import plan_axes_node

    example = load_example("3_2_1_10")
    good = AxisStructure.model_validate(example["axis_output"])
    bad = good.model_copy(deep=True)
    bad.groups[0].ys[1].name = "ksi"
    evidence = tmp_path / "evidence.json"
    evidence.write_text(evidence_for(example).model_dump_json(), encoding="utf-8")
    calls = []

    class RepairModel:
        def complete(self, messages):
            prompt = messages[-1]["content"][0]["text"]
            payload = json.loads(prompt[prompt.index('{"evidence":') :])
            calls.append(payload)
            if len(calls) == 1:
                assert "validation_report" not in payload
                return ModelReply(bad.model_dump_json(), "fake", 0, "stop")
            report = payload["validation_report"]
            assert report["scope"] == "structural_checks_only"
            assert report["target"] == "structure"
            assert payload["structure"] == bad.model_dump(mode="json")
            failed = [c for c in report["checks"] if c["status"] == "failed"]
            assert [(c["rule"], c["paths"]) for c in failed] == [
                ("axis_name_quantity", ["groups[0].ys[1].name"])
            ]
            assert any(
                c["paths"] == ["groups[0].x.name"] and c["status"] == "passed"
                for c in report["checks"]
            )
            return ModelReply(good.model_dump_json(), "fake", 0, "stop")

    result = plan_axes_node(RepairModel())(
        {
            "run_dir": str(tmp_path),
            "axis_plan": "",
            "evidence_graph": str(evidence),
            "source_asset": SourceAsset(
                source_id="test", path="chart.png", kind="image"
            ),
            "rendered_figure": str(PROMPTS / "examples" / example["image"]),
        }
    )
    assert len(calls) == 2
    assert result["status"] == "running"
    assert all(
        c["status"] == "passed" for c in result["structure_validation_report"]["checks"]
    )
    snapshot = json.loads(
        (tmp_path / "attempts/e0_s0/model/axes-repair/request.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["payload"]["validation_report"] == calls[1]["validation_report"]


def test_axis_report_scopes_duplicate_names_to_both_directions():
    axes = AxisStructure.model_validate(
        {
            "groups": [
                {
                    "x": {"name": "Time", "scale": "linear"},
                    "ys": [{"name": "Stress", "scale": "linear"}] * 2,
                }
            ]
        }
    )
    checks = []
    asset = SourceAsset(source_id="test", path="chart.png", kind="image")
    validate_axes(axes, evidence_for({}), asset, checks=checks)
    duplicate = next(c for c in checks if c["rule"] == "distinct_y_names")
    assert duplicate["status"] == "failed"
    assert duplicate["paths"] == ["groups[0].ys[0].name", "groups[0].ys[1].name"]


def test_dataset_report_scopes_cross_item_failures_and_keeps_loose_names():
    axis = AxisPlan(id="a", display_name="Time - Stress", axis_type="xy")
    proposal = DatasetStructure.model_validate(
        {
            "datasets": [
                {
                    "axis": "a",
                    "name": "Stress Annealed filled circle",
                    "kind": "scatter",
                },
                {
                    "axis": "a",
                    "name": "Stress Annealed filled circle",
                    "kind": "scatter",
                },
                {
                    "axis": "a",
                    "name": "Stress | solid upper boundary",
                    "kind": "range_boundary",
                },
                {"axis": "missing", "name": "Stress | open circle", "kind": "scatter"},
            ]
        }
    )
    checks = []
    asset = SourceAsset(source_id="test", path="chart.png", kind="image")
    issues = validate_datasets(proposal, [axis], asset, checks=checks)
    failed = {c["rule"]: c["paths"] for c in checks if c["status"] == "failed"}
    assert failed["duplicate_dataset_name"] == ["datasets[0].name", "datasets[1].name"]
    assert failed["incomplete_range_band"] == ["datasets[2].name"]
    assert failed["dataset_axis_unknown"] == ["datasets[3].axis"]
    assert not any(i.code == "marker_name_incomplete" for i in issues)
    assert any(
        c["rule"] == "dataset_axis_unknown"
        and c["paths"] == ["datasets[0].axis"]
        and c["status"] == "passed"
        for c in checks
    )
    # Shortened conditions and missing separators remain accepted.
    loose = DatasetStructure(datasets=proposal.datasets[:1])
    assert not validate_datasets(loose, [axis], asset)
