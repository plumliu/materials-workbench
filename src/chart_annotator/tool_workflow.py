"""The three Figure submission tools and their checkpoint-sized operations."""

import copy
import json
import time
from pathlib import Path

from pydantic import ValidationError

from chart_annotator import calibration, calibration_review, semantic, snapping, stages
from chart_annotator.config import load_model_config
from chart_annotator.domain.models import AxisPlan, ValidationIssue
from chart_annotator.domain.workflow import (
    AxisStructure,
    Bindings,
    Calibration,
    DatasetStructure,
    Evidence,
    Grounding,
    GroundingCompletion,
    GroundingResponse,
    GroundingSubmission,
    Structure,
)
from chart_annotator.intake import write_json
from chart_annotator.qwen import ModelCallError, QwenModel, image_message

PROTOCOL = "figure-tools/v1"
LIMITS = {"axes": 2, "grounding": 5, "datasets": 2}
SCHEMAS = {
    "axes": AxisStructure,
    "grounding": GroundingSubmission,
    "datasets": DatasetStructure,
}
DESCRIPTIONS = {
    "axes": "Submit the complete coordinate-structure proposal for Python validation.",
    "grounding": "Submit initial anchors or the current repair subset for Python detection and calibration; returns issues and visual evidence.",
    "datasets": "Submit the complete Dataset proposal against the accepted Axes for Python validation.",
}
SCOPES = {
    "axes": "structural_checks_only",
    "grounding": "geometry_and_numeric_fit",
    "datasets": "structural_and_name_checks",
}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def domain_arguments(value):
    return {
        key: item
        for key, item in value.items()
        if key not in {"schema_version", "coordinate_system"}
    }


def tool_definition(stage):
    schema = SCHEMAS[stage].model_json_schema()
    for key in ("schema_version", "coordinate_system"):
        schema["properties"].pop(key, None)
    return [
        {
            "type": "function",
            "function": {
                "name": f"submit_{stage}",
                "description": DESCRIPTIONS[stage],
                "parameters": schema,
            },
        }
    ]


def direction_addresses(axes):
    return [
        address
        for i, group in enumerate(axes.groups)
        for address in [
            f"groups[{i}].x",
            *[f"groups[{i}].ys[{j}]" for j in range(len(group.ys))],
        ]
    ]


def subset_mapping(axes, targets):
    """Only full accepted addresses select directions; local indices are output."""
    whitelist = direction_addresses(axes)
    if len(set(targets)) != len(targets) or not set(targets) <= set(whitelist):
        raise ValueError(
            "recheck_directions must be unique addresses from the initial list"
        )
    selected = set(targets)
    mapping = []
    for i, group in enumerate(axes.groups):
        x = f"groups[{i}].x"
        ys = [
            j
            for j in range(len(group.ys))
            if x in selected or f"groups[{i}].ys[{j}]" in selected
        ]
        if ys:
            mapping.append((i, ys))
    return mapping


def repair_payload(axes, grounding, targets):
    mapping = subset_mapping(axes, targets)
    groups, addresses = [], {}
    for local, (i, ys) in enumerate(mapping):
        group = grounding.groups[i]
        groups.append({"x": group.x, "ys": [group.ys[j] for j in ys]})
        addresses[f"groups[{local}].x"] = f"groups[{i}].x"
        for k, j in enumerate(ys):
            addresses[f"groups[{local}].ys[{k}]"] = f"groups[{i}].ys[{j}]"
    current = GroundingResponse(groups=groups).model_dump(mode="json")
    return {
        "current_repair_subset": domain_arguments(current),
        "direction_mapping": addresses,
    }


def merge_grounding(axes, baseline, submission, defaults):
    targets = list(dict.fromkeys([*defaults, *submission.recheck_directions]))
    mapping = subset_mapping(axes, targets)
    if not mapping:
        raise ValueError(
            "Select a direction with recheck_directions to correct a fitted result"
        )
    if len(submission.groups) != len(mapping):
        raise ValueError("Submit exactly the selected subset groups in original order")
    merged = baseline.model_copy(deep=True)
    for candidate, (i, ys) in zip(submission.groups, mapping, strict=True):
        if len(candidate.ys) != len(ys):
            raise ValueError("Submit selected/context Ys in original Y order and count")
        if f"groups[{i}].x" in targets:
            merged.groups[i].x = candidate.x
        for anchors, j in zip(candidate.ys, ys, strict=True):
            if f"groups[{i}].ys[{j}]" in targets:
                merged.groups[i].ys[j] = anchors
    merged.unresolved = submission.unresolved
    return merged


def initial_messages(stage, state):
    evidence = stages.read(state["evidence_graph"], Evidence)
    image = Path(
        state["dataset_context_sheet"]
        if stage == "datasets"
        else state["rendered_figure"]
    )
    data = state.get("stage_data", {})
    replacements = {"AXES_EVIDENCE_JSON": semantic.task_evidence("axes", evidence)}
    if stage == "grounding":
        axes = AxisStructure.model_validate(data["axes"])
        replacements = {
            "GROUNDING_STRUCTURE_JSON": semantic.compact_grounding_structure(axes),
            "DIRECTION_ADDRESSES_JSON": direction_addresses(axes),
            "GROUNDING_EVIDENCE_JSON": semantic.task_evidence("grounding", evidence),
        }
    elif stage == "datasets":
        replacements = {
            "COMPACT_AXES_JSON": semantic.compact_axes(
                Structure.model_validate(data["structure"]).axes
            ),
            "DATASETS_EVIDENCE_JSON": semantic.task_evidence("datasets", evidence),
        }
    prompt = (semantic.PROMPTS / f"{stage}-initial.txt").read_text(encoding="utf-8")
    for key, value in replacements.items():
        prompt = prompt.replace("{{" + key + "}}", dumps(value))
    messages = [
        {
            "role": "system",
            "content": (semantic.PROMPTS / f"{stage}-system.txt").read_text(
                encoding="utf-8"
            ),
        }
    ]
    for example in semantic.select_examples(evidence, task=stage):
        if stage == "axes":
            lesson, inputs, answer = (
                example["axis_lesson"],
                example["input"],
                example["axis_output"],
            )
        elif stage == "grounding":
            lesson, inputs, answer = (
                example["grounding_lesson"],
                example["grounding_input"],
                example["grounding_output"],
            )
        else:
            lesson = example["lesson"]
            inputs = {
                "axes": semantic.compact_axes(
                    [AxisPlan.model_validate(a) for a in example["output"]["axes"]]
                )
            }
            answer = semantic.dataset_example_output(example)
        messages.extend(
            image_message(
                semantic.PROMPTS / "examples" / example["image"],
                "REFERENCE EXAMPLE ONLY. This is reviewed reference material, not the current task or an executed tool result.\n"
                f"Lesson: {lesson}\nInput: {dumps(inputs)}\nReference arguments for submit_{stage}: {dumps(domain_arguments(answer))}",
            )
        )
    messages.extend(image_message(image, prompt))
    return messages


def record(state, event):
    """Bounded diagnostic events, never read as recovery state."""
    path = Path(state["run_dir"]) / "tools" / f"{state['active_stage']}.json"
    previous = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"protocol": PROTOCOL, "events": []}
    )
    previous["events"].append(event)
    write_json(path, previous)


def binding(state):
    return {
        "stage": state["active_stage"],
        "source": state["source_asset"].model_dump(mode="json"),
        "axes": state.get("stage_data", {}).get("axes"),
        "structure": state.get("stage_data", {}).get("structure"),
        "grounding": state.get("stage_data", {}).get("grounding"),
        "revision": state.get("calibration_revision", 0),
        "targets": state.get("stage_data", {}).get("default_targets", []),
    }


def feedback(state, text):
    stage = state["active_stage"]
    messages = copy.deepcopy(state["stage_messages"])
    messages[stage].append({"role": "user", "content": "PROGRAM FEEDBACK: " + text})
    return {
        "stage_messages": messages,
        "stage_outcome": "needs_revision",
        "status": "running",
    }


def stop(state, code, message, *, failed=False):
    return {
        "status": "failed" if failed else "needs_resolution",
        "stage_outcome": "blocked",
        "validation_issues": [
            *state.get("validation_issues", []),
            ValidationIssue(code=code, message=message, node=state["active_stage"]),
        ],
    }


def materialize(state):
    """Generated paths can be overwritten before a checkpoint; state wins."""
    data = state["stage_data"]
    directory = Path(state["run_dir"]) / "plan"
    update = {}
    for key, target in (
        ("axes", "axis_plan"),
        ("structure", "structure_plan"),
        ("grounding", "grounding_plan"),
        ("calibration", "calibration_plan"),
    ):
        if key in data:
            path = directory / f"{key}.json"
            write_json(path, data[key])
            update[target] = str(path)
    return update


def can_submit(state):
    # The next submission needs its own model turn and a later confirmation turn.
    return (
        state.get("grounding_submissions", 0) < 3
        and state.get("model_turns", {}).get("grounding", 0) < 4
        and not state.get("skipped_axes")
    )


def model_node(stage, model):
    def run(state):
        current = copy.deepcopy(state)
        current["active_stage"] = stage
        if current.get("workflow_protocol") != PROTOCOL:
            raise ValueError("Incompatible Figure checkpoint protocol")
        histories = current.setdefault("stage_messages", {})
        definitions = current.setdefault("stage_data", {}).setdefault(
            "tool_definitions", {}
        )
        if stage not in histories:
            current.update(materialize(current)) if current.get("stage_data") else None
            if stage == "datasets":
                current.update(stages.build_dataset_context(current))
            histories[stage] = initial_messages(stage, current)
            definitions[stage] = tool_definition(stage)
            current["stage_outcome"] = "initial"
        if current.get("pending_tool_call"):
            raise ValueError("Unpaired tool call at model boundary")
        turns = current.setdefault("model_turns", {})
        if turns.get(stage, 0) >= LIMITS[stage]:
            return {**current, **exhausted(current)}
        submit_allowed = stage != "grounding" or can_submit(current)
        if (
            stage == "grounding"
            and not submit_allowed
            and current["stage_outcome"] != "awaiting_confirmation"
        ):
            return {**current, **exhausted(current)}
        context = binding(current)
        context["sheet_index"] = current.get("stage_data", {}).get("sheet_index")
        context["images_sent"] = any(
            isinstance(m.get("content"), list)
            and any(
                p.get("type") == "text"
                and p.get("text", "").startswith("CURRENT FIGURE.")
                for p in m["content"]
            )
            and any(p.get("type") == "image_url" for p in m["content"])
            for m in histories[stage]
        )
        current["request_context"] = context
        started = time.perf_counter()
        request = {
            "messages": histories[stage],
            "tools": definitions[stage],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        record(
            current,
            {
                "kind": "request",
                "turn": turns.get(stage, 0) + 1,
                **request,
                "context": context,
            },
        )
        try:
            reply = (model or QwenModel(load_model_config())).complete(
                **copy.deepcopy(request)
            )
        except Exception as error:
            record(
                current,
                {
                    "kind": "failure",
                    "error": str(error)
                    if isinstance(error, ModelCallError)
                    else "Model adapter failed",
                    **(error.metadata if isinstance(error, ModelCallError) else {}),
                },
            )
            if isinstance(error, ModelCallError):
                raise
            raise RuntimeError(
                "Model request failed; inspect tool diagnostics"
            ) from None
        turns[stage] = turns.get(stage, 0) + 1
        calls = reply.tool_calls
        assistant = {
            **reply.assistant_fields,
            "role": "assistant",
            "content": reply.text,
        }
        if calls:
            assistant["tool_calls"] = calls
        histories[stage].append(assistant)
        record(
            current,
            {
                "kind": "response",
                "turn": turns[stage],
                "assistant": assistant,
                "model": reply.model,
                "finish_reason": reply.finish_reason,
                "total_tokens": reply.total_tokens,
                "http_attempts": reply.http_attempts,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )
        if reply.finish_reason not in {"stop", "tool_calls"} or (
            reply.finish_reason == "tool_calls" and not calls
        ):
            return {
                **current,
                **stop(
                    current,
                    "model_output_incomplete",
                    "Output truncated or refused",
                    failed=True,
                ),
            }
        if calls:
            ids = [call.get("id") for call in calls]
            if any(not isinstance(id_, str) or not id_.strip() for id_ in ids) or len(
                set(ids)
            ) != len(ids):
                return {
                    **current,
                    **stop(
                        current,
                        "tool_protocol_failed",
                        "Missing or duplicate tool call IDs",
                        failed=True,
                    ),
                }
            if any(
                call.get("type") != "function"
                or not isinstance(call.get("function"), dict)
                or not isinstance(call["function"].get("arguments"), str)
                or not isinstance(call["function"].get("name"), str)
                for call in calls
            ):
                return {
                    **current,
                    **stop(
                        current,
                        "tool_protocol_failed",
                        "Invalid tool call envelope",
                        failed=True,
                    ),
                }
            current["pending_tool_call"] = {
                "calls": calls,
                "binding": context,
                "submit_allowed": submit_allowed,
            }
            current["stage_outcome"] = "submitted"
        elif stage == "grounding":
            current["pending_completion"] = reply.text or ""
            current["stage_outcome"] = "completion"
        else:
            current.update(
                feedback(
                    current,
                    f"No submission was executed. Call submit_{stage} with the proposal; plain-text proposals are not accepted.",
                )
            )
            if turns[stage] >= LIMITS[stage]:
                current.update(exhausted(current))
        current.setdefault("status", "running")
        return current

    return run


def finish_tool(state, result, *, sheet=None, transform=None):
    stage = state["active_stage"]
    messages = copy.deepcopy(state["stage_messages"])
    calls = (
        state["pending_tool_call"]["calls"] if state.get("pending_tool_call") else []
    )
    for call in calls:
        messages[stage].append(
            {"role": "tool", "tool_call_id": call["id"], "content": dumps(result)}
        )
    data = copy.deepcopy(state.get("stage_data", {}))
    if sheet:
        source = (
            f"PROGRAM-GENERATED EVIDENCE ATTACHMENT for tool call {calls[0]['id']}. "
            if calls
            else "PROGRAM-GENERATED PARTIAL RESULT. " + dumps(result) + "\n"
        )
        text = (
            source
            + "This sheet shows the latest program result, not a new human request. "
            "Earlier sheets are historical; only the latest current result may be confirmed.\n"
            + dumps(transform)
        )
        messages[stage].extend(image_message(Path(sheet), text))
        data["sheet_index"] = len(messages[stage]) - 1
    record(
        state,
        {
            "kind": "tool_result",
            "calls": calls,
            "result": result,
            "calibration_revision": state.get("calibration_revision", 0),
            "sheet": str(sheet) if sheet else None,
        },
    )
    return {
        "stage_messages": messages,
        "stage_data": data,
        "pending_tool_call": None,
        "stage_outcome": result["status"],
        "status": "running",
    }


def reject(state, issues):
    result = {
        "status": "needs_revision",
        "validation_scope": SCOPES[state["active_stage"]],
        "issues": issues,
        "next_action": f"Correct these issues and call submit_{state['active_stage']}.",
    }
    if state["active_stage"] == "grounding":
        remaining_turn = (
            state.get("model_turns", {}).get("grounding", 0) < LIMITS["grounding"]
        )
        result["next_action"] = (
            "Correct the current repair subset; choose additional visibly wrong directions with recheck_directions, or report unresolved."
        )
        data = state.get("stage_data", {})
        if data.get("grouped_grounding"):
            axes = AxisStructure.model_validate(data["axes"])
            defaults = data.get("default_targets", [])
            result.update(
                default_repair_directions=defaults,
                **repair_payload(
                    axes,
                    GroundingResponse.model_validate(data["grouped_grounding"]),
                    defaults,
                ),
            )
        sheet_index = data.get("sheet_index")
        history = state["stage_messages"]["grounding"]
        valid_result = (
            not state.get("validation_issues")
            and data.get("calibration")
            and isinstance(sheet_index, int)
            and 0 <= sheet_index < len(history)
            and isinstance(history[sheet_index].get("content"), list)
            and any(
                part.get("type") == "image_url"
                for part in history[sheet_index]["content"]
            )
            and not stages.incomplete_axis_ids(
                Structure.model_validate(data["structure"]),
                Calibration.model_validate(data["calibration"]),
            )
        )
        if not remaining_turn:
            result.update(
                status="blocked",
                next_action="No model turns remain; automatic processing has stopped without confirmation.",
            )
        elif valid_result:
            result["status"] = "awaiting_confirmation"
            result["next_action"] = (
                "No new anchors were executed. Inspect the existing latest valid calibration sheet; "
                + (
                    "correct it with submit_grounding, confirm it if correct, or report unresolved."
                    if can_submit(state)
                    else "no further submissions are allowed. Confirm it if correct; otherwise report unresolved."
                )
            )
        elif not can_submit(state):
            result.update(
                status="blocked",
                next_action="No further submissions are allowed, and there is no valid calibration with a supplied sheet to confirm. Report unresolved.",
            )
    updated = {**state, **finish_tool(state, result)}
    if (
        updated.get("model_turns", {}).get(updated["active_stage"], 0)
        >= LIMITS[updated["active_stage"]]
    ):
        updated.update(exhausted(updated))
    return updated


def parse_submission(state):
    pending = state["pending_tool_call"]
    if pending["binding"] != state["request_context"] or binding(state) != {
        k: v
        for k, v in pending["binding"].items()
        if k not in {"sheet_index", "images_sent"}
    }:
        return None, [
            {
                "code": "stale_request",
                "paths": [],
                "message": "Request binding changed; submission rejected",
            }
        ]
    calls = pending["calls"]
    if (
        len(calls) != 1
        or calls[0]["function"]["name"] != f"submit_{state['active_stage']}"
    ):
        return None, [
            {
                "code": "tool_not_allowed",
                "paths": [],
                "message": "Use exactly one tool call for this stage",
            }
        ]
    if state["active_stage"] == "grounding" and not pending["submit_allowed"]:
        return None, [
            {
                "code": "submission_budget_exhausted",
                "paths": [],
                "message": "No submissions are allowed; confirm the current valid result or report unresolved",
            }
        ]
    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
        if (
            isinstance(arguments, dict)
            and {"schema_version", "coordinate_system"} & arguments.keys()
        ):
            return None, [
                {
                    "code": "python_owned_fields",
                    "paths": [],
                    "message": "Submit domain fields only",
                }
            ]
        return SCHEMAS[state["active_stage"]].model_validate(arguments), []
    except json.JSONDecodeError:
        return None, [
            {
                "code": "invalid_json",
                "paths": [],
                "message": "Tool arguments must be valid JSON",
            }
        ]
    except ValidationError as error:
        return None, [
            {
                "code": "invalid_arguments",
                "paths": [argument_path(state, arguments, e["loc"])],
                "message": e["msg"],
            }
            for e in error.errors(include_input=False, include_url=False)
        ]


def argument_path(state, arguments, location):
    path = ""
    for item in location:
        path += f"[{item}]" if isinstance(item, int) else ("." if path else "") + item
    data = state.get("stage_data", {})
    if (
        state["active_stage"] != "grounding"
        or not data.get("grouped_grounding")
        or not isinstance(arguments, dict)
    ):
        return path
    extra = arguments.get("recheck_directions", [])
    if not isinstance(extra, list) or any(not isinstance(item, str) for item in extra):
        return path
    targets = list(dict.fromkeys([*data.get("default_targets", []), *extra]))
    axes = AxisStructure.model_validate(data["axes"])
    if not set(targets) <= set(direction_addresses(axes)):
        return path
    mapping = subset_mapping(axes, targets)
    for local, (i, ys) in enumerate(mapping):
        prefixes = [(f"groups[{local}].x", f"groups[{i}].x")]
        prefixes += [
            (f"groups[{local}].ys[{k}]", f"groups[{i}].ys[{j}]")
            for k, j in enumerate(ys)
        ]
        for before, after in prefixes:
            if (
                path == before
                or path.startswith(before + "[")
                or path.startswith(before + ".")
            ):
                return after + path[len(before) :]
        if path == f"groups[{local}]":
            return f"groups[{i}]"
    return path


def checked_issues(issues, checks):
    failed = [check for check in checks if check["status"] == "failed"]
    return [
        {
            "code": issue.code,
            "message": issue.message,
            "paths": check["paths"],
        }
        for issue, check in zip(issues, failed, strict=True)
    ]


def validate_axes_submission(state):
    candidate, errors = parse_submission(state)
    if errors:
        return reject(state, errors)
    checks = []
    issues = semantic.validate_axes(
        candidate,
        stages.read(state["evidence_graph"], Evidence),
        state["source_asset"],
        checks=checks,
    )
    record(
        state,
        {
            "kind": "validation",
            "candidate": candidate.model_dump(mode="json"),
            "checks": checks,
        },
    )
    if issues:
        return reject(
            {**state, "validation_issues": issues}, checked_issues(issues, checks)
        )
    data = {
        "tool_definitions": state["stage_data"]["tool_definitions"],
        "axes": candidate.model_dump(mode="json"),
        "structure": stages.combine_structure(candidate).model_dump(mode="json"),
    }
    current = {**state, "stage_data": data, "validation_issues": []}
    current.update(materialize(current))
    return {
        **current,
        **finish_tool(
            current,
            {"status": "accepted", "validation_scope": SCOPES["axes"], "issues": []},
        ),
    }


def validate_dataset_submission(state):
    candidate, errors = parse_submission(state)
    if errors:
        return reject(state, errors)
    structure = Structure.model_validate(state["stage_data"]["structure"])
    checks = []
    issues = semantic.validate_datasets(
        candidate, structure.axes, state["source_asset"], checks=checks
    )
    record(
        state,
        {
            "kind": "validation",
            "candidate": candidate.model_dump(mode="json"),
            "checks": checks,
        },
    )
    if issues:
        return reject(
            {**state, "validation_issues": issues}, checked_issues(issues, checks)
        )
    structure.datasets = candidate.expand()
    axes = AxisStructure.model_validate(state["stage_data"]["axes"])
    structure.unresolved = [*axes.unresolved, *candidate.unresolved]
    data = {**state["stage_data"], "structure": structure.model_dump(mode="json")}
    current = {**state, "stage_data": data, "validation_issues": []}
    current.update(materialize(current))
    return {
        **current,
        **finish_tool(
            current,
            {
                "status": "accepted",
                "validation_scope": SCOPES["datasets"],
                "issues": [],
            },
        ),
    }


def snap_and_fit(state):
    current = copy.deepcopy(state)
    if current["pending_tool_call"]["submit_allowed"]:
        current["grounding_submissions"] = current.get("grounding_submissions", 0) + 1
    candidate, errors = parse_submission(current)
    if errors:
        return reject(current, errors)
    data = current["stage_data"]
    axes = AxisStructure.model_validate(data["axes"])
    try:
        if "grouped_grounding" in data:
            if len(set(candidate.recheck_directions)) != len(
                candidate.recheck_directions
            ):
                raise ValueError("recheck_directions must be unique")
            baseline = GroundingResponse.model_validate(data["grouped_grounding"])
            merged = merge_grounding(
                axes, baseline, candidate, data.get("default_targets", [])
            )
        else:
            if candidate.recheck_directions:
                raise ValueError(
                    "Initial submission must be complete; omit recheck_directions"
                )
            merged = GroundingResponse.model_validate(
                {
                    k: v
                    for k, v in candidate.model_dump(mode="json").items()
                    if k != "recheck_directions"
                }
            )
        grounded = semantic.expand_grounding(merged, axes)
    except ValueError as error:
        return reject(
            current,
            [
                {
                    "code": "grounding_structure_invalid",
                    "paths": ["groups"],
                    "message": str(error),
                }
            ],
        )
    if data.get("grouped_grounding") == merged.model_dump(mode="json"):
        if current.get("validation_issues"):
            current.update(
                reject(
                    current,
                    [
                        {
                            "code": "repeated_grounding",
                            "paths": [],
                            "message": "Identical proposal repeats the same unresolved issues",
                        }
                    ],
                )
            )
            return {**current, **exhausted(current)}
        return {
            **current,
            **finish_tool(
                current,
                {
                    "status": "awaiting_confirmation",
                    "validation_scope": SCOPES["grounding"],
                    "issues": [],
                    "next_action": "No anchors changed. Inspect the existing latest sheet and confirm it, or report unresolved.",
                },
            ),
        }
    data["grouped_grounding"] = merged.model_dump(mode="json")
    data["grounding"] = grounded.model_dump(mode="json")
    current["calibration_revision"] = current.get("calibration_revision", 0) + 1
    current["confirmed_calibration_revision"] = None
    data.pop("sheet_index", None)
    return execute_fit(current)


def execute_fit(state):
    current = copy.deepcopy(state)
    current.update(materialize(current))
    data = current["stage_data"]
    structure, grounding = (
        Structure.model_validate(data["structure"]),
        Grounding.model_validate(data["grounding"]),
    )
    directory = Path(current["run_dir"]) / "grounding"
    bindings, local, issues = snapping.snap_grounding(
        structure,
        grounding,
        stages.read(current["evidence_graph"], Evidence),
        Path(current["rendered_figure"]),
        directory / "snap",
        alternative=current.get("evidence_preprocessing") == "alternative",
    )
    result, fit_issues = calibration.fit_bindings(structure, bindings, local)
    failed_snap_ids = {id_ for issue in issues for id_ in issue.evidence_ids}
    issues.extend(
        issue
        for issue in fit_issues
        if not failed_snap_ids.intersection(issue.evidence_ids)
    )
    data.update(
        calibration=result.model_dump(mode="json"),
        bindings=bindings.model_dump(mode="json"),
        local_evidence=local.model_dump(mode="json"),
    )
    current["validation_issues"] = issues
    current.update(materialize(current))
    write_json(
        directory / "validation.json", [i.model_dump(mode="json") for i in issues]
    )
    if (
        issues
        and all(issue.repair == "evidence" for issue in issues)
        and not current.get("repair_attempts", {}).get("evidence")
    ):
        current.update(stage_outcome="retry_local_fit", status="running")
        return current
    return finish_fit(current)


def retry_local_fit(state):
    return execute_fit(
        {
            **state,
            "evidence_preprocessing": "alternative",
            "repair_attempts": {**state.get("repair_attempts", {}), "evidence": 1},
        }
    )


def fit_targets(axes, fitted, issues):
    targets = []
    keys = {(fit.axis_id, fit.direction) for fit in fitted.fits}
    for i, (x, ys, plans) in enumerate(semantic.grounding_groups(axes)):
        if x.scale != "categorical" and any(
            (axis.id, "x") not in keys for axis in plans
        ):
            targets.append(f"groups[{i}].x")
        for j, (y, plan) in enumerate(zip(ys, plans, strict=True)):
            if y.scale != "categorical" and (plan.id, "y") not in keys:
                targets.append(f"groups[{i}].ys[{j}]")
    if issues and not targets:
        targets = [address for address in direction_addresses(axes)]
    return targets


def image_transform(context, axes):
    directions = {}
    for i, (_, _, plans) in enumerate(semantic.grounding_groups(axes)):
        for j, axis in enumerate(plans):
            directions[(axis.id, "x")] = f"groups[{i}].x"
            directions[(axis.id, "y")] = f"groups[{i}].ys[{j}]"
    views = copy.deepcopy(context["views"])
    for view in views:
        parts = view["title"].split(" / ")
        if len(parts) >= 2:
            view["full_direction"] = directions.get((parts[0], parts[1].split(" |")[0]))
    return {"original_image_size": context["original_image_size"], "views": views}


def finish_fit(state):
    current = copy.deepcopy(state)
    data = current["stage_data"]
    structure, grounding = (
        Structure.model_validate(data["structure"]),
        Grounding.model_validate(data["grounding"]),
    )
    bindings, local, fitted = (
        Bindings.model_validate(data["bindings"]),
        Evidence.model_validate(data["local_evidence"]),
        Calibration.model_validate(data["calibration"]),
    )
    issues = current["validation_issues"]
    axes = AxisStructure.model_validate(data["axes"])
    targets = fit_targets(axes, fitted, issues)
    data["default_targets"] = targets
    partial = None
    if issues and (
        current.get("grounding_submissions", 0) >= 3
        or current["model_turns"]["grounding"] >= 4
        or current.get("stage_outcome") == "finalize_partial_axes"
    ):
        structure, grounding, bindings, fitted, issues, skipped = (
            stages.retain_complete_axes(structure, grounding, bindings, fitted, issues)
        )
        if skipped:
            current["calibration_revision"] += 1
            current["confirmed_calibration_revision"] = None
            partial = {"retained_axis_ids": [a.id for a in structure.axes]}
            data.update(
                structure=structure.model_dump(mode="json"),
                grounding=grounding.model_dump(mode="json"),
                calibration=fitted.model_dump(mode="json"),
            )
            skipped_path = Path(current["run_dir"]) / "grounding/skipped_axes.json"
            write_json(
                skipped_path,
                {**partial, "issues": [i.model_dump(mode="json") for i in skipped]},
            )
            current["skipped_axes"] = str(skipped_path)
            current.update(materialize(current))
    current["validation_issues"] = issues
    sheet, context = calibration_review.contact_sheet(
        Path(current["rendered_figure"]),
        grounding,
        bindings,
        local,
        Path(current["run_dir"]) / "grounding",
        calibration=fitted,
    )
    current["calibration_review_sheet"] = str(sheet)
    result = {
        "status": "needs_revision" if issues else "awaiting_confirmation",
        "validation_scope": SCOPES["grounding"],
        "issues": [
            {"code": issue.code, "paths": targets, "message": issue.message}
            for issue in issues
        ],
    }
    if partial:
        result.update(partial)
        result["direction_mapping"] = {
            a.id: [f"groups[{i}].x", f"groups[{i}].ys[{j}]"]
            for i, (_, _, plans) in enumerate(semantic.grounding_groups(axes))
            for j, a in enumerate(plans)
            if a.id in partial["retained_axis_ids"]
        }
    if issues:
        result.update(
            default_repair_directions=targets,
            **repair_payload(
                axes,
                GroundingResponse.model_validate(data["grouped_grounding"]),
                targets,
            ),
        )
        result["next_action"] = (
            "Correct the listed directions. You may also select a visibly wrong fitted direction with recheck_directions."
        )
    else:
        result["next_action"] = (
            "Inspect the attached latest calibration sheet. Correct it with submit_grounding, confirm it if correct, or report unresolved."
        )
    if not can_submit(current):
        result["next_action"] = (
            "No further submissions are allowed. Confirm the current valid result if its sheet is correct; otherwise report unresolved."
            if not issues
            else "No further submissions are allowed; the numerical result is invalid. Report unresolved."
        )
    current.update(
        finish_tool(
            current, result, sheet=sheet, transform=image_transform(context, axes)
        )
    )
    if issues and not can_submit(current):
        current.update(exhausted(current))
    return current


def validate_grounding_completion(state):
    try:
        completion = GroundingCompletion.model_validate_json(
            state["pending_completion"]
        )
    except ValidationError:
        completion = None
    if completion and completion.status == "unresolved":
        return stop(state, "grounding_unresolved", completion.reason)
    context = state["request_context"]
    data = state["stage_data"]
    sheet_index = data.get("sheet_index")
    messages = state["stage_messages"]["grounding"]
    sheet_sent = (
        isinstance(sheet_index, int)
        and sheet_index < len(messages) - 1
        and any(
            p.get("type") == "image_url"
            for p in messages[sheet_index].get("content", [])
            if isinstance(p, dict)
        )
    )
    valid = (
        completion
        and completion.status == "confirmed"
        and context["revision"] == state.get("calibration_revision")
        and {
            k: v for k, v in context.items() if k not in {"sheet_index", "images_sent"}
        }
        == binding(state)
        and context["images_sent"]
        and sheet_sent
        and context["sheet_index"] == sheet_index
        and not state.get("validation_issues")
        and data.get("calibration")
        and not state.get("pending_tool_call")
    )
    if valid:
        structure, fitted = (
            Structure.model_validate(data["structure"]),
            Calibration.model_validate(data["calibration"]),
        )
        valid = bool(structure.axes) and not stages.incomplete_axis_ids(
            structure, fitted
        )
    record(
        state,
        {
            "kind": "completion",
            "text": state["pending_completion"],
            "accepted": bool(valid),
            "context": context,
        },
    )
    if valid:
        return {
            "status": "running",
            "stage_outcome": "accepted",
            "confirmed_calibration_revision": state["calibration_revision"],
            "pending_completion": None,
            **materialize(state),
        }
    if state["model_turns"]["grounding"] >= 5:
        return exhausted(state)
    return feedback(
        state,
        "The current calibration is not eligible for confirmation: invalid conclusion, changed result, missing sheet, or numerical failure. Correct it if allowed, or report unresolved.",
    ) | {
        "stage_outcome": "awaiting_confirmation"
        if sheet_sent and not state.get("validation_issues")
        else "needs_revision"
    }


def exhausted(state):
    if state["active_stage"] == "datasets" and state.get(
        "confirmed_calibration_revision"
    ) == state.get("calibration_revision"):
        path = Path(state["run_dir"]) / "tools/skipped_datasets.json"
        write_json(
            path,
            {
                "reason": "Dataset submission budget exhausted",
                "issues": [
                    i.model_dump(mode="json")
                    for i in state.get("validation_issues", [])
                ],
            },
        )
        return {
            "status": "running",
            "stage_outcome": "accepted",
            "validation_issues": [],
            "skipped_datasets": str(path),
        }
    if (
        state["active_stage"] == "grounding"
        and state.get("model_turns", {}).get("grounding", 0) < 5
        and not state.get("skipped_axes")
    ):
        data = state.get("stage_data", {})
        if data.get("calibration") and data.get("bindings"):
            structure = Structure.model_validate(data["structure"])
            failed = stages.incomplete_axis_ids(
                structure, Calibration.model_validate(data["calibration"])
            )
            if failed and failed != {axis.id for axis in structure.axes}:
                return {"status": "running", "stage_outcome": "finalize_partial_axes"}
    return stop(
        state,
        "stage_budget_exhausted",
        "No accepted result or valid confirmation within this stage's budget",
    )


def nodes(model):
    return {
        "plan_axes": model_node("axes", model),
        "validate_axes_submission": validate_axes_submission,
        "ground_axes": model_node("grounding", model),
        "snap_and_fit": snap_and_fit,
        "retry_local_fit": retry_local_fit,
        "validate_grounding_completion": validate_grounding_completion,
        "finalize_partial_axes": finish_fit,
        "plan_datasets": model_node("datasets", model),
        "validate_dataset_submission": validate_dataset_submission,
        "export_and_validate": stages.export_and_validate,
    }
