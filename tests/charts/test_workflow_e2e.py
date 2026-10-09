import copy
import json
from pathlib import Path

import pymupdf
import pytest

from chart_annotator.qwen import ModelReply
from chart_annotator.runner import run_figure
from chart_annotator.wpd import read


def axes_arguments():
    return {
        "groups": [
            {
                "x": {"name": "Time", "scale": "linear"},
                "ys": [{"name": "Stress", "scale": "linear"}],
            }
        ],
        "unresolved": [],
    }


def grounding_arguments():
    def anchor(value, x, y):
        return {
            "visible_label": str(value),
            "value": value,
            "point_2d": [1000 * x / 440, 1000 * y / 390],
        }

    return {
        "groups": [
            {
                "x": [anchor(0, 73, 287), anchor(10, 167, 287), anchor(30, 367, 287)],
                "ys": [[anchor(30, 73, 53), anchor(20, 73, 127), anchor(0, 73, 287)]],
            }
        ],
        "unresolved": [],
    }


def dataset_arguments():
    return {
        "datasets": [
            {
                "axis": "axis_000_000",
                "name": "Stress Annealed filled circle",
                "kind": "scatter",
            }
        ],
        "unresolved": [],
    }


def tool_reply(name, arguments, id_="call_1"):
    return ModelReply(
        None,
        "fake",
        0,
        "tool_calls",
        [
            {
                "id": id_,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments),
                },
            }
        ],
    )


class FakeChartModel:
    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def complete(self, messages, **kwargs):
        call = copy.deepcopy({"messages": messages, **kwargs})
        self.calls.append(call)
        name = kwargs["tools"][0]["function"]["name"]
        if name == "submit_grounding" and any(m["role"] == "tool" for m in messages):
            reply = ModelReply('{"status":"confirmed"}', "fake", 0, "stop")
        else:
            arguments = {
                "submit_axes": axes_arguments,
                "submit_grounding": grounding_arguments,
                "submit_datasets": dataset_arguments,
            }[name]()
            reply = tool_reply(name, arguments, f"call_{len(self.calls)}")
        return self.transform(self, call, reply) if self.transform else reply


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


def run(tmp_path, model=None, datasets=False):
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    return run_figure(
        source, tmp_path / "out", model=model or FakeChartModel(), datasets=datasets
    )


def test_real_local_evidence_fake_model_complete_graph(tmp_path):
    model = FakeChartModel()
    result = run(tmp_path, model)
    assert result["status"] == "exported", result
    assert result["model_turns"] == {"axes": 1, "grounding": 2}
    assert result["grounding_submissions"] == 1
    assert list(result["nodes"]) == [
        "prepare_figure",
        "plan_axes",
        "validate_axes_submission",
        "ground_axes",
        "snap_and_fit",
        "validate_grounding_completion",
        "export_and_validate",
    ]
    assert len(model.calls) == 3
    tar = next(Path(p) for p in result["artifacts"] if p.endswith(".tar"))
    project, _ = read(tar)
    assert len(project["axesColl"]) == 1 and project["datasetColl"] == []
    assert len(project["axesColl"][0]["calibrationPoints"]) == 4
    assert result["checkpoint"] is None


@pytest.mark.parametrize(
    "axes_opinions,dataset_opinions",
    [
        ([], []),
        (["Axis 标题待人工核对"], []),
        (["Axis 标题待人工核对"], ["系列归属待人工核对。\n<保留模型原话>"]),
    ],
)
def test_dataset_planning_is_explicitly_enabled(tmp_path, axes_opinions, dataset_opinions):
    def transform(model, call, reply):
        name = call["tools"][0]["function"]["name"]
        if name == "submit_axes":
            arguments = axes_arguments()
            arguments["unresolved"] = axes_opinions
            return tool_reply(name, arguments, f"call_{len(model.calls)}")
        if name == "submit_datasets":
            arguments = dataset_arguments()
            arguments["unresolved"] = dataset_opinions
            return tool_reply(name, arguments, f"call_{len(model.calls)}")
        return reply

    model = FakeChartModel(transform)
    result = run(tmp_path, model, datasets=True)
    assert result["status"] == "exported", result
    opinions = [*axes_opinions, *dataset_opinions]
    assert result["unresolved"] == opinions
    run_dir = Path(result["run_dir"])
    for path in ("plan/structure.json", "audit/summary.json"):
        record = json.loads((run_dir / path).read_text(encoding="utf-8"))
        assert record["unresolved"] == opinions
    assert len(model.calls) == 4
    assert all(call["tool_choice"] == "auto" for call in model.calls)
    assert model.calls[-1]["tools"][0]["function"]["name"] == "submit_datasets"
    assert not any(
        m["role"] in {"assistant", "tool"} for m in model.calls[-1]["messages"]
    )
    project, _ = read(next(Path(p) for p in result["artifacts"] if p.endswith(".tar")))
    assert project["datasetColl"][0]["name"] == "Stress Annealed filled circle"


@pytest.mark.parametrize(
    "bad",
    [
        "{not-json",
        {
            "groups": [
                {
                    "x": {"name": "Time", "scale": "categorical", "categories": ["A"]},
                    "ys": [
                        {"name": "Stress", "scale": "categorical", "categories": ["B"]}
                    ],
                }
            ]
        },
        {"groups": [{"x": {"name": "Time", "scale": "INVALID"}, "ys": []}]},
    ],
)
def test_invalid_axes_repairs_before_expanding(tmp_path, bad):
    def transform(model, call, reply):
        return tool_reply("submit_axes", bad) if len(model.calls) == 1 else reply

    model = FakeChartModel(transform)
    result = run(tmp_path, model)
    assert result["status"] == "exported", result
    assert result["model_turns"]["axes"] == 2
    first, second = model.calls[:2]
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert second["tools"] == first["tools"]
    assert second["messages"][-1]["role"] == "tool"
    assert json.loads(second["messages"][-1]["content"])["status"] == "needs_revision"


def test_axes_repair_does_not_spend_grounding_budget(tmp_path):
    def transform(model, call, reply):
        name = call["tools"][0]["function"]["name"]
        count = sum(c["tools"][0]["function"]["name"] == name for c in model.calls)
        if name == "submit_axes" and count == 1:
            return tool_reply(name, "{")
        if name == "submit_grounding" and count == 1:
            bad = grounding_arguments()
            for anchor in bad["groups"][0]["x"]:
                anchor["point_2d"][1] = 500
            return tool_reply(name, bad)
        if name == "submit_grounding" and count == 2:
            result = next(
                json.loads(m["content"])
                for m in reversed(call["messages"])
                if m["role"] == "tool"
            )
            assert result["default_repair_directions"] == ["groups[0].x"]
            return tool_reply(name, grounding_arguments())
        return reply

    model = FakeChartModel(transform)
    result = run(tmp_path, model)
    assert result["status"] == "exported", result
    assert result["model_turns"] == {"axes": 2, "grounding": 3}
    assert result["grounding_submissions"] == 2


@pytest.mark.parametrize("repair", [True, False])
def test_dataset_validation_repair_or_axes_only(tmp_path, repair):
    def transform(model, call, reply):
        if call["tools"][0]["function"]["name"] == "submit_datasets":
            count = sum(
                c["tools"][0]["function"]["name"] == "submit_datasets"
                for c in model.calls
            )
            if not repair or count == 1:
                bad = dataset_arguments()
                bad["datasets"][0]["name"] = "Stress"
                return tool_reply("submit_datasets", bad)
        return reply

    result = run(tmp_path, FakeChartModel(transform), datasets=True)
    assert result["status"] == "exported", result
    assert result["model_turns"]["datasets"] == 2
    project, _ = read(next(Path(p) for p in result["artifacts"] if p.endswith(".tar")))
    assert bool(project["datasetColl"]) == repair
    assert bool(result["skipped_datasets"]) != repair
