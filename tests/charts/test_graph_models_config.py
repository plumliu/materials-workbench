import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from chart_annotator import config
from chart_annotator.domain.models import (
    AxisPlan,
    ChartPlan,
    DatasetPlan,
    SourceAsset,
)
from chart_annotator.graph import FIGURE_NODES, build_workflow


def test_graph_compiles_and_fails_closed_on_missing_stage_input():
    graph = build_workflow()
    assert set(FIGURE_NODES) <= graph.nodes.keys()
    result = graph.invoke({"mode": "figure"})
    assert result["status"] == "failed"
    assert result["node_status"]["prepare_figure"] == "failed"
    assert result["validation_issues"][0].node == "prepare_figure"


def test_offline_figure_workflow_uses_tool_boundaries():
    called = []

    def offline(name):
        def node(state):
            called.append(name)
            return {
                "status": "exported" if name == "export_and_validate" else "running",
                "validation_issues": [],
                "pending_tool_call": {"fake": True}
                if name == "plan_axes"
                or (name == "ground_axes" and called.count(name) == 1)
                else None,
                "stage_outcome": "accepted"
                if name == "validate_axes_submission"
                or name == "validate_grounding_completion"
                else "completion",
            }

        return node

    overrides = {name: offline(name) for name in FIGURE_NODES}
    result = build_workflow(offline_nodes=overrides).invoke({"mode": "figure"})
    assert called == [
        "prepare_figure",
        "plan_axes",
        "validate_axes_submission",
        "ground_axes",
        "snap_and_fit",
        "ground_axes",
        "validate_grounding_completion",
        "export_and_validate",
    ]
    assert result["status"] == "exported"


def test_empty_datasets_unique_names_and_references():
    source = SourceAsset(source_id="test", path="image.png", kind="image")
    axis = AxisPlan(id="a", display_name="a", axis_type="xy")
    dataset = DatasetPlan(id="d", axis_id="a", display_name="d", kind="scatter")
    plan = ChartPlan(source_asset=source, axes=[axis], datasets=[dataset])
    assert ChartPlan.model_validate_json(plan.model_dump_json()) == plan
    with pytest.raises(ValidationError):
        DatasetPlan(**{**dataset.model_dump(), "data": [{"x": 1, "y": 2}]})
    with pytest.raises(ValidationError):
        ChartPlan(source_asset=source, axes=[axis], datasets=[dataset, dataset])
    with pytest.raises(ValidationError):
        ChartPlan(source_asset=source, axes=[], datasets=[dataset])
    with pytest.raises(ValidationError):
        DatasetPlan(id="d", axis_id="a", display_name="d", kind="point_group")
    with pytest.raises(ValidationError):
        AxisPlan(
            id="a",
            display_name="a",
            axis_type="xy",
            x_scale="log",
            calibration={"x": [{"tick_id": "t", "value": 0}]},
        )
    with pytest.raises(ValidationError):
        AxisPlan(
            id="a",
            display_name="a",
            axis_type="categorical_x_numeric_y",
            x_scale="categorical",
            calibration={"x": [{"tick_id": "invented-A", "value": 1}]},
        )


def test_root_dotenv_is_the_configuration_source(tmp_path, monkeypatch):
    for name in ("MODEL", "BASE_URL", "API_KEY", "EXTRA_BODY", "MAX_COMPLETION_TOKENS"):
        monkeypatch.delenv(f"CHART_ANNOTATOR_{name}", raising=False)
    (tmp_path / ".env").write_text(
        "CHART_ANNOTATOR_MODEL=file-model\nCHART_ANNOTATOR_API_KEY=test-secret-only\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv("CHART_ANNOTATOR_MODEL", "environment-model")
    settings = config.load_model_config()
    assert settings.model == "file-model"
    assert settings.api_key.get_secret_value() == "test-secret-only"
    assert settings.reasoning_effort == "xhigh"
    assert settings.max_completion_tokens == 65536
    env = tmp_path / ".env"
    env.write_text(env.read_text() + "CHART_ANNOTATOR_MAX_COMPLETION_TOKENS=32768\n")
    assert config.load_model_config().max_completion_tokens == 32768
    assert settings.extra_body == {}
    assert "test-secret-only" not in repr(settings) + settings.model_dump_json()
    assert "api_key" not in settings.model_dump()
    env.write_text(
        env.read_text()
        + 'CHART_ANNOTATOR_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":true}}\n'
    )
    assert config.load_model_config().extra_body["chat_template_kwargs"][
        "enable_thinking"
    ]
    env.write_text(
        env.read_text().replace(
            '{"chat_template_kwargs":{"enable_thinking":true}}',
            '{"api_key":"test-secret-only"}',
        )
    )
    with pytest.raises(ValueError, match="provider configuration") as error:
        config.load_model_config()
    assert "test-secret-only" not in str(error.value)


def test_production_has_no_historical_runtime_imports():
    root = Path(__file__).resolve().parents[2] / "src/chart_annotator"
    for path in root.rglob("*.py"):
        assert ".temp-codex" not in path.read_text(encoding="utf-8")


def test_all_eight_structural_goldens():
    paths = list((Path(__file__).parent / "fixtures/charts").glob("*/chart_plan.json"))
    assert len(paths) == 8
    for path in paths:
        plan = ChartPlan.model_validate_json(path.read_text(encoding="utf-8"))
        reference = json.loads(
            (path.parent / "reference.json").read_text(encoding="utf-8")
        )
        assert len(plan.axes) == reference["axis_count"]
        assert len(plan.datasets) == reference["dataset_count"]
        assert all(dataset.data == [] for dataset in plan.datasets)
