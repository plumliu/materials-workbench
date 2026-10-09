import copy
import json
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from PIL import Image
from test_workflow_e2e import (
    FakeChartModel,
    axes_arguments,
    chart_pdf,
    grounding_arguments,
    run,
    tool_reply,
)

from chart_annotator import tool_workflow as tools
from chart_annotator.domain.models import SourceAsset
from chart_annotator.domain.workflow import (
    AxisStructure,
    Evidence,
    Geometry,
    GroundingResponse,
    GroundingSubmission,
)
from chart_annotator.graph import build_workflow
from chart_annotator.qwen import ModelReply
from chart_annotator.runner import run_figure


def bare_state(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image)
    evidence = Evidence(
        source_id="test",
        geometry=Geometry(
            image_size=(100, 100), preprocessing="test", spines=[], ticks=[]
        ),
        texts=[],
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(evidence.model_dump_json(), encoding="utf-8")
    return {
        "workflow_protocol": tools.PROTOCOL,
        "run_dir": str(tmp_path),
        "rendered_figure": str(image),
        "evidence_graph": str(evidence_path),
        "source_asset": SourceAsset(source_id="test", path=str(image), kind="image"),
        "status": "running",
    }


@pytest.mark.parametrize(
    "stage,count", [("axes", 6), ("grounding", 2), ("datasets", 3)]
)
def test_prefix_has_one_system_reference_user_messages_and_minimal_input(
    tmp_path, stage, count
):
    state = bare_state(tmp_path)
    axes = AxisStructure.model_validate(axes_arguments())
    state["stage_data"] = {
        "axes": axes.model_dump(mode="json"),
        "structure": tools.stages.combine_structure(axes).model_dump(mode="json"),
    }
    state["dataset_context_sheet"] = state["rendered_figure"]
    messages = tools.initial_messages(stage, state)
    assert len(messages) == count + 2
    assert messages[0]["role"] == "system"
    assert all(m["role"] == "user" for m in messages[1:])
    assert "schema_version" not in messages[-1]["content"][0]["text"]
    assert '"schema"' not in messages[-1]["content"][0]["text"]
    assert all(
        "Reference arguments for submit_" in m["content"][0]["text"]
        for m in messages[1:-1]
    )
    schema = tools.tool_definition(stage)[0]["function"]["parameters"]
    assert "schema_version" not in schema["properties"]
    assert "coordinate_system" not in schema["properties"]
    if stage == "grounding":
        assert "enum" not in schema["properties"]["recheck_directions"]["items"]


@pytest.mark.parametrize(
    "case", ["unknown", "multiple", "duplicate", "missing", "text"]
)
def test_tool_envelopes_never_execute_illegal_proposals(tmp_path, case):
    state = bare_state(tmp_path)

    class Model:
        def complete(self, **kwargs):
            if case == "text":
                return ModelReply(json.dumps(axes_arguments()), "fake", 0, "stop")
            reply = tool_reply(
                "submit_other" if case == "unknown" else "submit_axes", axes_arguments()
            )
            if case == "multiple":
                reply.tool_calls.append(
                    tool_reply("submit_axes", axes_arguments(), "call_2").tool_calls[0]
                )
            if case == "duplicate":
                reply.tool_calls.append(copy.deepcopy(reply.tool_calls[0]))
            if case == "missing":
                reply.tool_calls[0].pop("id")
            return reply

    result = tools.model_node("axes", Model())(state)
    if case in {"duplicate", "missing"}:
        assert result["status"] == "failed"
        assert not result.get("pending_tool_call")
    elif case == "text":
        assert result["stage_messages"]["axes"][-1]["role"] == "user"
        assert not result.get("axis_plan")
    else:
        result = tools.validate_axes_submission(result)
        tail = [m for m in result["stage_messages"]["axes"] if m["role"] == "tool"]
        assert len(tail) == (2 if case == "multiple" else 1)
        assert not result.get("axis_plan")
        assert all(
            json.loads(m["content"])["issues"][0]["code"] == "tool_not_allowed"
            for m in tail
        )


def test_missing_fields_and_invalid_json_pair_with_original_call(tmp_path):
    for arguments in ("{bad", '{"groups":[{}]}'):

        class Model:
            def complete(self, **kwargs):
                return tool_reply("submit_axes", arguments, "original_call")

        state = tools.model_node("axes", Model())(bare_state(tmp_path))
        result = tools.validate_axes_submission(state)
        assert (
            result["stage_messages"]["axes"][-2]["tool_calls"][0]["function"][
                "arguments"
            ]
            == arguments
        )
        assert result["stage_messages"]["axes"][-1]["tool_call_id"] == "original_call"
        assert result["stage_outcome"] == "needs_revision"


def multi_axes():
    group = axes_arguments()["groups"][0]
    group["ys"] *= 3
    return AxisStructure.model_validate({"groups": [group, group, group]})


@pytest.mark.parametrize(
    "targets,mapping",
    [
        (["groups[2].ys[0]", "groups[2].ys[2]"], [(2, [0, 2])]),
        (["groups[2].x"], [(2, [0, 1, 2])]),
        (["groups[0].ys[1]", "groups[2].x"], [(0, [1]), (2, [0, 1, 2])]),
    ],
)
def test_repair_subsets_keep_full_addresses_and_ignore_context_changes(
    targets, mapping
):
    axes = multi_axes()
    anchors = grounding_arguments()["groups"][0]
    baseline = GroundingResponse.model_validate(
        {"groups": [{"x": anchors["x"], "ys": anchors["ys"] * 3}] * 3}
    )
    assert tools.subset_mapping(axes, targets) == mapping
    payload = tools.repair_payload(axes, baseline, targets)
    candidate = GroundingSubmission.model_validate(
        {**payload["current_repair_subset"], "recheck_directions": targets[1:]}
    )
    for group in candidate.groups:
        group.x[0].point_2d = (1, 1)
        for ys in group.ys:
            ys[0].point_2d = (2, 2)
    merged = tools.merge_grounding(axes, baseline, candidate, targets[:1])
    for i, group in enumerate(merged.groups):
        assert (group.x[0].point_2d == (1, 1)) == (f"groups[{i}].x" in targets)
        for j, ys in enumerate(group.ys):
            assert (ys[0].point_2d == (2, 2)) == (f"groups[{i}].ys[{j}]" in targets)
    expanded = tools.semantic.expand_grounding(merged, axes)
    assert expanded.axes[6].x == expanded.axes[7].x == expanded.axes[8].x
    with pytest.raises(ValueError):
        tools.subset_mapping(axes, ["groups[0].ys[99]"])


def test_grounding_history_frozen_images_and_no_independent_review(tmp_path):
    model = FakeChartModel()
    result = run(tmp_path, model)
    assert result["status"] == "exported"
    first, second = model.calls[1:]
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert second["tools"] == first["tools"]
    assert first["tool_choice"] == "auto"
    assert second["tool_choice"] == "auto"
    assert [m["role"] for m in second["messages"][-3:]] == ["assistant", "tool", "user"]
    attachment = second["messages"][-1]["content"]
    assert attachment[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "original_image_size" in attachment[0]["text"]
    assert "VisualReview" not in str(model.calls)


@pytest.mark.parametrize(
    "interrupt",
    ["plan_axes", "ground_axes", "snap_and_fit", "validate_grounding_completion"],
)
def test_sqlite_resume_uses_checkpoint_inputs_after_generated_paths_overwritten(
    tmp_path, interrupt
):
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    database = tmp_path / "checkpoint.sqlite"
    config = {"configurable": {"thread_id": "resume"}, "recursion_limit": 40}
    model = FakeChartModel()
    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_workflow(model=model, checkpointer=saver)
        list(
            graph.stream(
                {
                    "input_path": str(source),
                    "output_dir": str(tmp_path / "out"),
                    "mode": "figure",
                    "workflow_protocol": tools.PROTOCOL,
                },
                config,
                interrupt_after=[interrupt],
            )
        )
        snapshot = graph.get_state(config)
        history = copy.deepcopy(snapshot.values["stage_messages"])
        count = len(model.calls)
        # None of these generated paths may become a recovery input.
        for key in (
            "axis_plan",
            "structure_plan",
            "grounding_plan",
            "calibration_plan",
        ):
            if snapshot.values.get(key):
                Path(snapshot.values[key]).write_text('{"damaged":"uncommitted write"}')
        if snapshot.values.get("calibration_review_sheet"):
            Image.new("RGB", (20, 20), "red").save(
                snapshot.values["calibration_review_sheet"]
            )
        result = graph.invoke(None, config)
        assert result["status"] == "exported", result
        assert len(model.calls) == 3
        for stage, messages in history.items():
            assert result["stage_messages"][stage][: len(messages)] == messages
        if interrupt in {"plan_axes", "ground_axes"}:
            assert count in {1, 2}


@pytest.mark.parametrize("corrupt", ["revision", "sheet", "numeric", "source", "plain"])
def test_confirmation_rejects_stale_missing_evidence_or_invalid_conclusion(
    tmp_path, corrupt
):
    source = tmp_path / "chart.pdf"
    chart_pdf(source)
    config = {"configurable": {"thread_id": "gate"}, "recursion_limit": 40}
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_workflow(model=FakeChartModel(), checkpointer=saver)
        list(
            graph.stream(
                {
                    "input_path": str(source),
                    "output_dir": str(tmp_path / "out"),
                    "mode": "figure",
                },
                config,
                interrupt_before=["validate_grounding_completion"],
            )
        )
        state = copy.deepcopy(graph.get_state(config).values)
        if corrupt == "revision":
            state["calibration_revision"] += 1
        elif corrupt == "sheet":
            state["stage_data"].pop("sheet_index")
        elif corrupt == "numeric":
            state["stage_data"]["calibration"]["fits"] = []
        elif corrupt == "source":
            state["source_asset"] = state["source_asset"].model_copy(
                update={"source_id": "changed"}
            )
        else:
            state["pending_completion"] = "done"
        result = tools.validate_grounding_completion(state)
        assert result.get("confirmed_calibration_revision") is None
        assert result["stage_outcome"] != "accepted"


def test_auto_reserves_confirmation_and_over_budget_submission_is_rejected(
    tmp_path,
):
    def transform(model, call, reply):
        if call["tools"][0]["function"]["name"] == "submit_grounding":
            n = sum(
                c["tools"][0]["function"]["name"] == "submit_grounding"
                for c in model.calls
            )
            if n == 2:
                return ModelReply("invalid conclusion", "fake", 0, "stop")
            if n == 3:
                return ModelReply("invalid conclusion", "fake", 0, "stop")
            if n == 4:
                return ModelReply("invalid conclusion", "fake", 0, "stop")
            if n == 5:
                assert call["tool_choice"] == "auto"
                return tool_reply("submit_grounding", grounding_arguments())
        return reply

    result = run(tmp_path, FakeChartModel(transform))
    assert result["status"] == "needs_resolution"
    assert result["model_turns"]["grounding"] == 5
    assert result["grounding_submissions"] == 1
    assert not list(Path(result["run_dir"]).rglob("chart.tar"))


@pytest.mark.parametrize("stage", ["axes", "grounding", "datasets"])
@pytest.mark.parametrize("protocol", [None, "figure/old"])
def test_model_boundary_rejects_unversioned_and_old_states(tmp_path, stage, protocol):
    state = bare_state(tmp_path)
    if protocol is None:
        state.pop("workflow_protocol")
    else:
        state["workflow_protocol"] = protocol
    with pytest.raises(ValueError, match="Incompatible"):
        tools.model_node(stage, object())(state)


def test_old_checkpoint_refused_and_terminal_resume_stays_closed(tmp_path):
    database = tmp_path / "checkpoint.sqlite"
    config = {"configurable": {"thread_id": "old"}}
    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_workflow(checkpointer=saver)
        graph.update_state(
            config, {"mode": "figure", "status": "failed"}, as_node="prepare_figure"
        )
    with pytest.raises(ValueError, match="Incompatible"):
        run_figure(None, tmp_path, database, "old")


@pytest.mark.parametrize("mode", ["correct", "unchanged", "unresolved"])
def test_same_grounding_session_corrects_or_finishes(tmp_path, mode):
    def transform(model, call, reply):
        if call["tools"][0]["function"]["name"] != "submit_grounding":
            return reply
        count = sum(
            c["tools"][0]["function"]["name"] == "submit_grounding" for c in model.calls
        )
        if count == 2:
            if mode == "unresolved":
                return ModelReply(
                    '{"status":"unresolved","reason":"unreadable latest sheet"}',
                    "fake",
                    0,
                    "stop",
                )
            args = grounding_arguments()
            args["recheck_directions"] = ["groups[0].x"]
            if mode == "correct":
                args["groups"][0]["x"][0]["point_2d"][0] += 5
            return tool_reply("submit_grounding", args)
        return reply

    model = FakeChartModel(transform)
    result = run(tmp_path, model)
    assert result["status"] == (
        "needs_resolution" if mode == "unresolved" else "exported"
    ), result
    if mode != "unresolved":
        first_sheet = model.calls[2]["messages"][-1]["content"][1]["image_url"]["url"]
        sheets = [
            m["content"][1]["image_url"]["url"]
            for m in model.calls[3]["messages"]
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and m["content"][0]["text"].startswith("PROGRAM-GENERATED")
        ]
        assert sheets[0] == first_sheet
        assert len(sheets) == (2 if mode == "correct" else 1)
        assert result["grounding_submissions"] == 2


@pytest.mark.parametrize("repeat", [False, True])
def test_exhausted_failed_y_retains_complete_axes_and_gets_new_confirmation(
    tmp_path, repeat
):
    def transform(model, call, reply):
        name = call["tools"][0]["function"]["name"]
        if name == "submit_axes":
            args = axes_arguments()
            args["groups"][0]["ys"].append({"name": "Time", "scale": "linear"})
            return tool_reply(name, args)
        if name == "submit_grounding":
            count = sum(c["tools"][0]["function"]["name"] == name for c in model.calls)
            if count <= (2 if repeat else 3):
                args = grounding_arguments()
                bad_y = copy.deepcopy(args["groups"][0]["ys"][0])
                for anchor in bad_y:
                    anchor["point_2d"][0] = 510 if repeat else 500 + count * 10
                args["groups"][0]["ys"] = (
                    [args["groups"][0]["ys"][0], bad_y] if count == 1 else [bad_y]
                )
                return tool_reply(name, args)
            assert call["tool_choice"] == "auto"
            tool_result = next(
                json.loads(m["content"])
                for m in reversed(call["messages"])
                if m["role"] == "tool"
            )
            if repeat:
                assert tool_result["status"] == "needs_revision"
                text = call["messages"][-1]["content"][0]["text"]
                assert text.startswith("PROGRAM-GENERATED PARTIAL RESULT.")
                assert '"retained_axis_ids":["axis_000_000"]' in text
            else:
                assert tool_result["retained_axis_ids"] == ["axis_000_000"]
                assert tool_result["status"] == "awaiting_confirmation"
        return reply

    model = FakeChartModel(transform)
    result = run(tmp_path, model)
    assert result["status"] == "exported", result
    assert result["skipped_axes"]
    assert result["grounding_submissions"] == (2 if repeat else 3)
    assert result["model_turns"]["grounding"] == (3 if repeat else 4)


def test_local_schema_error_path_is_translated_to_full_direction(tmp_path):
    axes = multi_axes()
    state = bare_state(tmp_path) | {
        "active_stage": "grounding",
        "stage_data": {
            "axes": axes.model_dump(mode="json"),
            "grouped_grounding": {"present": True},
            "default_targets": ["groups[2].ys[0]"],
        },
    }
    args = {"recheck_directions": ["groups[2].ys[2]"]}
    assert (
        tools.argument_path(state, args, ("groups", 0, "ys", 1, 0, "value"))
        == "groups[2].ys[2][0].value"
    )


@pytest.mark.parametrize(
    "mode", ["submission_limit", "reserved_turn", "last_turn", "no_result"]
)
def test_grounding_rejection_feedback_agrees_with_remaining_actions(tmp_path, mode):
    def transform(model, call, reply):
        if call["tools"][0]["function"]["name"] != "submit_grounding":
            return reply
        count = sum(
            c["tools"][0]["function"]["name"] == "submit_grounding" for c in model.calls
        )
        if mode == "no_result":
            return tool_reply("submit_grounding", "{")
        if mode == "submission_limit" and count in {2, 3}:
            if count == 3:
                last = json.loads(call["messages"][-1]["content"])
                assert last["status"] == "awaiting_confirmation"
                assert "confirm" in last["next_action"]
                assert call["tool_choice"] == "auto"
            return tool_reply("submit_grounding", "{")
        if mode == "reserved_turn" and count in {2, 3}:
            return ModelReply("invalid conclusion", "fake", 0, "stop")
        if mode == "reserved_turn" and count == 4:
            return tool_reply("submit_grounding", "{")
        if mode == "last_turn" and count in {2, 3, 4}:
            return ModelReply("invalid conclusion", "fake", 0, "stop")
        if mode == "last_turn" and count == 5:
            assert call["tool_choice"] == "auto"
            return tool_reply("submit_grounding", "{")
        if count == (4 if mode == "submission_limit" else 5) and mode in {
            "submission_limit",
            "reserved_turn",
        }:
            assert call["tool_choice"] == "auto"
            last = json.loads(call["messages"][-1]["content"])
            assert last["status"] == "awaiting_confirmation"
            assert "no further submissions are allowed" in last["next_action"]
            assert "Confirm it if correct" in last["next_action"]
            sheets = [
                m
                for m in call["messages"]
                if m["role"] == "user"
                and isinstance(m["content"], list)
                and m["content"][0]["text"].startswith("PROGRAM-GENERATED")
            ]
            assert len(sheets) == 1
            return ModelReply('{"status":"confirmed"}', "fake", 0, "stop")
        return reply

    model = FakeChartModel(transform)
    result = run(tmp_path, model)
    assert result["status"] == (
        "exported"
        if mode in {"submission_limit", "reserved_turn"}
        else "needs_resolution"
    ), result
    diagnostics = json.loads(
        (Path(result["run_dir"]) / "tools/grounding.json").read_text(encoding="utf-8")
    )
    last = next(
        event["result"]
        for event in reversed(diagnostics["events"])
        if event["kind"] == "tool_result"
    )
    assert last["status"] == (
        "awaiting_confirmation"
        if mode in {"submission_limit", "reserved_turn"}
        else "blocked"
    )
    if mode == "no_result":
        assert result["model_turns"]["grounding"] == 3
        assert "no valid calibration" in last["next_action"]
    if mode == "last_turn":
        assert result["model_turns"]["grounding"] == 5
        assert "No model turns remain" in last["next_action"]


def test_same_rule_axes_errors_keep_each_actual_field_path(tmp_path):
    state = bare_state(tmp_path)
    arguments = axes_arguments()
    arguments["groups"][0]["x"]["name"] = "MPa"
    arguments["groups"][0]["ys"][0]["name"] = "ksi"

    class Model:
        def complete(self, **kwargs):
            return tool_reply("submit_axes", arguments)

    result = tools.validate_axes_submission(tools.model_node("axes", Model())(state))
    feedback = json.loads(result["stage_messages"]["axes"][-1]["content"])
    unit_errors = [
        issue for issue in feedback["issues"] if "unit alone" in issue["message"]
    ]
    assert [issue["paths"] for issue in unit_errors] == [
        ["groups[0].x.name"],
        ["groups[0].ys[0].name"],
    ]


def test_dataset_feedback_keeps_repeated_rule_and_cross_item_paths():
    from chart_annotator.domain.models import AxisPlan
    from chart_annotator.domain.workflow import DatasetStructure

    axes = [AxisPlan(id="a", display_name="Time - Stress", axis_type="xy")]
    proposal = DatasetStructure.model_validate(
        {
            "datasets": [
                {"axis": "a", "name": "Stress", "kind": "scatter"},
                {"axis": "a", "name": "Time", "kind": "scatter"},
                {"axis": "a", "name": "duplicate filled circle", "kind": "scatter"},
                {"axis": "a", "name": "duplicate filled circle", "kind": "scatter"},
                {
                    "axis": "a",
                    "name": "Stress solid upper boundary",
                    "kind": "range_boundary",
                },
            ]
        }
    )
    checks = []
    issues = tools.semantic.validate_datasets(
        proposal,
        axes,
        SourceAsset(source_id="test", path="image.png", kind="image"),
        checks=checks,
    )
    feedback = tools.checked_issues(issues, checks)
    assert [
        issue["paths"]
        for issue in feedback
        if issue["code"] == "marker_name_incomplete"
    ] == [["datasets[0].name"], ["datasets[1].name"]]
    assert next(
        issue["paths"]
        for issue in feedback
        if issue["code"] == "duplicate_dataset_name"
    ) == ["datasets[2].name", "datasets[3].name"]
    assert next(
        issue["paths"] for issue in feedback if issue["code"] == "incomplete_range_band"
    ) == ["datasets[4].name"]
