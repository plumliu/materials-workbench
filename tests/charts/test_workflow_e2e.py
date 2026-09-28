import json
from pathlib import Path

import pymupdf
import pytest

from chart_annotator.domain.workflow import AxisStructure, DatasetStructure
from chart_annotator.qwen import ModelReply
from chart_annotator.runner import run_figure
from chart_annotator.wpd import read


class FakeChartModel:
    """Deterministic semantic answers for the synthetic chart; pixels from evidence."""

    def __init__(self):
        self.calls = []

    def complete(self, messages):
        prompt = messages[-1]["content"][0]["text"]
        payload = json.loads(prompt[prompt.index('{"evidence":') :])
        self.calls.append(payload)
        title = payload["schema"]["title"]
        if title == "AxisStructure":
            result = {
                "groups": [
                    {
                        "x": {"name": "Time", "scale": "linear"},
                        "ys": [{"name": "Stress", "scale": "linear"}],
                    }
                ],
            }
            result = AxisStructure.model_validate(result).model_dump(mode="json")
        elif title == "DatasetStructure":
            result = DatasetStructure.model_validate(
                {
                    "datasets": [
                        {
                            "axis": "axis_000_000",
                            "name": "Stress | filled circle",
                            "kind": "scatter",
                        }
                    ]
                }
            ).model_dump(mode="json")
        elif title == "VisualReview":
            result = {
                "axes": [
                    {
                        "axis_id": a["id"],
                        "status": "consistent",
                        "issues": [],
                        "corrected_grounding": None,
                    }
                    for a in payload["structure"]["axes"]
                ],
                "unresolved": [],
            }
        else:
            # Deliberately imprecise grounding, independent of local candidates.
            def point(x, y):
                return [1000 * x / 440, 1000 * y / 390]

            result = {
                "coordinate_system": "normalized_0_1000",
                "groups": [
                    {
                        "x": [
                            {
                                "visible_label": "0",
                                "value": 0,
                                "point_2d": point(73, 287),
                            },
                            {
                                "visible_label": "10",
                                "value": 10,
                                "point_2d": point(167, 287),
                            },
                            {
                                "visible_label": "30",
                                "value": 30,
                                "point_2d": point(367, 287),
                            },
                        ],
                        "ys": [
                            [
                                {
                                    "visible_label": "30",
                                    "value": 30,
                                    "point_2d": point(73, 53),
                                },
                                {
                                    "visible_label": "20",
                                    "value": 20,
                                    "point_2d": point(73, 127),
                                },
                                {
                                    "visible_label": "0",
                                    "value": 0,
                                    "point_2d": point(73, 287),
                                },
                            ]
                        ],
                    }
                ],
            }
        return ModelReply(json.dumps(result), "fake", 0, "stop")


def chart_pdf(path, figure_id="1.2"):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=440, height=390)
        page.draw_rect((70, 50, 370, 290), width=1)
        for i in range(4):
            x = 70 + 100 * i
            page.draw_line((x, 283), (x, 296), width=1)
            page.insert_text((x - 4, 312), str(i * 10), fontsize=11)
            y = 290 - 80 * i
            page.draw_line((64, y), (77, y), width=1)
            page.insert_text((43, y + 4), str(i * 10), fontsize=11)
        page.insert_text((190, 340), "Time", fontsize=12)
        page.insert_text((15, 170), "Stress", fontsize=12, rotate=90)
        for x, y in [(100, 240), (210, 150), (300, 90)]:
            page.draw_circle((x, y), 2, fill=(0, 0, 0))
        page.insert_text((70, 365), f"Figure {figure_id} Synthetic chart", fontsize=12)
        pdf.save(path)






















def test_real_local_evidence_fake_model_complete_graph(tmp_path):
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    model = FakeChartModel()
    result = run_figure(source, tmp_path / "output", model=model)
    assert result["status"] == "exported", result
    assert result["datasets_enabled"] is False
    assert len(model.calls) == 3
    assert list(result["nodes"]) == [
        "prepare_figure",
        "plan_axes",
        "ground_axes",
        "snap_and_fit",
        "review_calibration",
        "export_and_validate",
    ]
    assert set(model.calls[1]["evidence"]) == {"image_size", "numeric_text"}
    assert "Grounding" == model.calls[1]["schema"]["title"]
    tar = next(Path(p) for p in result["artifacts"] if p.endswith(".tar"))
    project, _ = read(tar)
    assert len(project["axesColl"]) == 1
    assert project["datasetColl"] == []
    assert len(project["axesColl"][0]["calibrationPoints"]) == 4
    assert result["checkpoint"] is None
    assert not list((tmp_path / "output/checkpoints").glob("*.sqlite"))


def test_dataset_planning_is_explicitly_enabled(tmp_path):
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    model = FakeChartModel()
    result = run_figure(source, tmp_path / "output", model=model, datasets=True)
    assert result["status"] == "exported", result
    assert result["datasets_enabled"] is True
    assert [call["schema"]["title"] for call in model.calls] == [
        "AxisStructure",
        "Grounding",
        "VisualReview",
        "DatasetStructure",
    ]
    dataset_call = model.calls[-1]
    assert dataset_call["review_context"]["schema_version"] == "dataset-context/v1"
    assert dataset_call["review_context"]["axes"][0]["alias"] == "A1"
    assert dataset_call["review_context"]["axes"][0]["axis_id"] == "axis_000_000"
    direction_colors = {
        item["direction"]: item["color"]
        for item in dataset_call["review_context"]["axes"][0]["directions"]
    }
    assert direction_colors["x"] != direction_colors["y"]
    assert "grounding" not in dataset_call["review_context"]
    assert dataset_call["axes"] == [
        {
            "id": "axis_000_000",
            "name": "Time - Stress",
            "x": {"name": "Time", "scale": "linear"},
            "y": {"name": "Stress", "scale": "linear"},
        }
    ]
    assert "structure" not in dataset_call
    assert (
        Path(result["run_dir"]) / "attempts/e0_s0/dataset-context/dataset_context.png"
    ).exists()
    project, _ = read(next(Path(p) for p in result["artifacts"] if p.endswith(".tar")))
    assert [dataset["name"] for dataset in project["datasetColl"]] == [
        "Stress | filled circle"
    ]


def test_model_failure_is_sanitized_and_never_exports(tmp_path):
    import pytest
    from PIL import Image

    from chart_annotator.domain.workflow import Evidence, Geometry
    from chart_annotator.semantic import request

    class FailedModel:
        def complete(self, messages):
            raise RuntimeError("provider echoed a secret")

    image = tmp_path / "blank.png"
    Image.new("RGB", (100, 100)).save(image)
    evidence = Evidence(
        source_id="test",
        geometry=Geometry(
            image_size=(100, 100), preprocessing="test", spines=[], ticks=[]
        ),
        texts=[],
    )
    with pytest.raises(RuntimeError, match="Model request failed"):
        request(FailedModel(), "axes", evidence, image, tmp_path / "model")
    assert "secret" not in (tmp_path / "model/failure.json").read_text()




def test_resume_from_completed_node_boundary(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from chart_annotator.graph import build_workflow

    source = tmp_path / "resume.pdf"
    chart_pdf(source)
    database = tmp_path / "checkpoint.sqlite"
    configuration = {"configurable": {"thread_id": "interrupted"}}
    model = FakeChartModel()
    with SqliteSaver.from_conn_string(str(database)) as saver:
        workflow = build_workflow(model=model, checkpointer=saver)
        list(
            workflow.stream(
                {
                    "input_path": str(source),
                    "output_dir": str(tmp_path / "runs"),
                    "mode": "figure",
                },
                configuration,
                interrupt_after=["prepare_figure"],
            )
        )
        snapshot = workflow.get_state(configuration)
        assert snapshot.next == ("plan_axes",)
        image = Path(snapshot.values["rendered_figure"])
        before = image.stat().st_mtime_ns
    result = run_figure(None, tmp_path, database, "interrupted", model=model)
    assert result["status"] == "exported", result
    assert result["checkpoint"] is None
    assert not database.exists()
    assert image.stat().st_mtime_ns == before
    assert len(model.calls) == 3


def test_failed_axis_visual_review_relocates_grounding_then_revalidates(tmp_path):
    class RepairModel(FakeChartModel):
        def complete(self, messages):
            reply = super().complete(messages)
            if len(self.calls) == 2:
                result = json.loads(reply.text)
                for anchor in result["groups"][0]["x"]:
                    anchor["point_2d"][1] = 500
                return ModelReply(json.dumps(result), "fake", 0, "stop")
            return reply

    source = tmp_path / "repair.pdf"
    chart_pdf(source)
    model = RepairModel()
    result = run_figure(source, tmp_path / "output", model=model)
    assert result["status"] == "exported", result
    assert len(model.calls) == 4
    assert result["nodes"]["snap_and_fit"] == "completed"
    assert model.calls[-2]["issues"]
    assert model.calls[-2]["grounding"]["coordinate_system"] == "normalized_0_1000"
    assert model.calls[-2]["review_context"]["mode"] == "failed_axes"
    assert model.calls[-2]["review_context"]["failed_axis_ids"] == ["axis_000_000"]
    assert model.calls[-2]["review_context"]["successful_fit_directions"] == [
        {"axis_id": "axis_000_000", "direction": "y"}
    ]
    assert model.calls[-2]["evidence"] == {"image_size": [1320, 1170]}
    directory = Path(result["run_dir"])
    assert (directory / "attempts/e1_s0/failed_axis_review.png").exists()
    assert (directory / "attempts/e1_s0_f1/calibration_review.png").exists()


@pytest.mark.parametrize("repair_succeeds,expected_axes", [(True, 2), (False, 1)])
def test_failed_axis_review_excludes_successful_sibling_axis(
    tmp_path, repair_succeeds, expected_axes
):
    class TwoAxisFailureModel(FakeChartModel):
        @staticmethod
        def grounding(group, *, bad_second_y=False):
            def point(x, y):
                return [1000 * x / 440, 1000 * y / 390]

            ys = []
            for index, _ in enumerate(group["ys"]):
                bad_x = (
                    220
                    if bad_second_y and (len(group["ys"]) == 1 or index == 1)
                    else 73
                )
                ys.append(
                    [
                        {
                            "visible_label": "30",
                            "value": 30,
                            "point_2d": point(bad_x, 53),
                        },
                        {
                            "visible_label": "20",
                            "value": 20,
                            "point_2d": point(bad_x, 127),
                        },
                        {
                            "visible_label": "0",
                            "value": 0,
                            "point_2d": point(bad_x, 287),
                        },
                    ]
                )
            return {
                "coordinate_system": "normalized_0_1000",
                "groups": [
                    {
                        "x": [
                            {
                                "visible_label": "0",
                                "value": 0,
                                "point_2d": point(73, 287),
                            },
                            {
                                "visible_label": "10",
                                "value": 10,
                                "point_2d": point(167, 287),
                            },
                            {
                                "visible_label": "30",
                                "value": 30,
                                "point_2d": point(367, 287),
                            },
                        ],
                        "ys": ys,
                    }
                ],
                "unresolved": [],
            }

        def complete(self, messages):
            prompt = messages[-1]["content"][0]["text"]
            payload = json.loads(prompt[prompt.index('{"evidence":') :])
            if payload["schema"]["title"] == "AxisStructure":
                self.calls.append(payload)
                result = AxisStructure.model_validate(
                    {
                        "groups": [
                            {
                                "x": {"name": "Time", "scale": "linear"},
                                "ys": [
                                    {"name": "Stress", "scale": "linear"},
                                    {"name": "Time", "scale": "linear"},
                                ],
                            }
                        ]
                    }
                ).model_dump(mode="json")
                return ModelReply(json.dumps(result), "fake", 0, "stop")
            if payload["schema"]["title"] == "Grounding":
                self.calls.append(payload)
                failed_review = (
                    payload.get("review_context", {}).get("mode") == "failed_axes"
                )
                result = self.grounding(
                    payload["structure"]["groups"][0],
                    bad_second_y=not (failed_review and repair_succeeds),
                )
                return ModelReply(json.dumps(result), "fake", 0, "stop")
            return super().complete(messages)

    source = tmp_path / "two-axis.pdf"
    chart_pdf(source)
    model = TwoAxisFailureModel()
    result = run_figure(source, tmp_path / "output", model=model)
    assert result["status"] == "exported", result
    correction_call = next(
        call
        for call in model.calls
        if call.get("review_context", {}).get("mode") == "failed_axes"
    )
    assert len(correction_call["structure"]["groups"]) == 1
    assert len(correction_call["structure"]["groups"][0]["ys"]) == 1
    assert len(correction_call["grounding"]["groups"]) == 1
    assert len(correction_call["grounding"]["groups"][0]["ys"]) == 1
    assert correction_call["review_context"]["successful_fit_directions"] == [
        {"axis_id": "axis_000_001", "direction": "x"}
    ]
    project, _ = read(next(Path(p) for p in result["artifacts"] if p.endswith(".tar")))
    assert len(project["axesColl"]) == expected_axes
    assert "targeted_repair" not in result["nodes"]
    if repair_succeeds:
        assert not result.get("skipped_axes")
    else:
        assert result["skipped_axes"]
        final_review = next(
            call for call in model.calls if call["schema"]["title"] == "VisualReview"
        )
        assert [axis["id"] for axis in final_review["structure"]["axes"]] == [
            "axis_000_000"
        ]


def test_global_fit_recovers_offset_anchor_without_retry(tmp_path):
    class OffsetModel(FakeChartModel):
        def complete(self, messages):
            reply = super().complete(messages)
            if len(self.calls) == 2:
                output = json.loads(reply.text)
                output["groups"][0]["x"][0]["point_2d"][0] += 50
                return ModelReply(json.dumps(output), "fake", 0, "stop")
            return reply

    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    model = OffsetModel()
    result = run_figure(source, tmp_path / "output", model=model)
    assert result["status"] == "exported", result
    assert (
        len(model.calls) == 3
    )  # Local retry adds no call; final visual review is mandatory.
    assert (
        Path(result["run_dir"]) / "attempts/e0_s0/snap/local_bindings.json"
    ).exists()
    assert not (Path(result["run_dir"]) / "attempts/e1_s0").exists()


def test_dataset_repair_revalidates_after_calibration_and_preserves_attempts(tmp_path):
    class NamingRepairModel(FakeChartModel):
        def complete(self, messages):
            prompt = messages[-1]["content"][0]["text"]
            payload = json.loads(prompt[prompt.index('{"evidence":') :])
            if (
                payload.get("issues")
                and payload["issues"][0]["node"] == "validate_dataset_plan"
            ):
                self.calls.append(payload)
                return self.correct
            reply = super().complete(messages)
            if payload["schema"]["title"] == "DatasetStructure":
                self.correct = reply
                bad = json.loads(reply.text)
                bad["datasets"][0]["name"] = "Stress"
                return ModelReply(json.dumps(bad), "fake", 0, "stop")
            return reply

    source = tmp_path / "repair.pdf"
    chart_pdf(source)
    model = NamingRepairModel()
    result = run_figure(source, tmp_path / "output", model=model, datasets=True)
    assert result["status"] == "exported", result
    assert len(model.calls) == 5
    assert not model.calls[3].get("issues")
    assert model.calls[4]["issues"][0]["code"] == "marker_name_incomplete"
    directory = Path(result["run_dir"])
    assert (directory / "attempts/e0_s0/dataset_validation.json").exists()
    assert not (directory / "attempts/e0_s0_d1/dataset_validation.json").exists()


def test_persistently_bad_dataset_naming_falls_back_to_axes_only(tmp_path):
    class BadNamingModel(FakeChartModel):
        def complete(self, messages):
            prompt = messages[-1]["content"][0]["text"]
            payload = json.loads(prompt[prompt.index('{"evidence":') :])
            if payload.get("issues"):
                self.calls.append(payload)
                return self.bad
            reply = super().complete(messages)
            if payload["schema"]["title"] != "DatasetStructure":
                return reply
            bad = json.loads(reply.text)
            bad["datasets"][0]["name"] = "Stress"
            self.bad = ModelReply(json.dumps(bad), "fake", 0, "stop")
            return self.bad

    source = tmp_path / "bad.pdf"
    chart_pdf(source)
    model = BadNamingModel()
    result = run_figure(source, tmp_path / "output", model=model, datasets=True)
    assert result["status"] == "exported", result
    assert len(model.calls) == 5
    assert result["nodes"]["ground_axes"] == "completed"
    assert result["skipped_datasets"]
    project, _ = read(next(Path(p) for p in result["artifacts"] if p.endswith(".tar")))
    assert project["datasetColl"] == []
