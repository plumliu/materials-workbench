"""Shared operations for the local UI and Codex CLI. No model agent scheduler."""

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

from .storage import candidates, child, lock, read_json, write_json


def paths(root, name):
    return (
        child(root / "runs", name),
        child(root / "figure_assets", name),
        child(root / "pdfs", name + ".pdf"),
    )


def manual(root, name):
    run, assets, pdf = paths(root, name)
    data = read_json(run / "manual.json")
    if not data:
        raise ValueError("请先完成 intake")
    if data.get("schema_version") != 2:
        raise ValueError("不支持旧手册记录；请使用新工作台格式")
    stat = pdf.stat()
    if (
        data["source_size"] != stat.st_size
        or data["source_mtime_ns"] != stat.st_mtime_ns
    ):
        raise ValueError("来源 PDF 已变化，请让 Codex 检查后重新接收")
    return data


def intake(root, name):
    from chart_annotator.intake import run_intake

    run, assets, pdf = paths(root, name)
    if (run / "manual.json").exists() or assets.exists():
        raise ValueError("手册已有接收记录或资产，请让 Codex 检查；不会覆盖已有成果")
    data = run_intake(pdf, root / "figure_assets", run)
    return {
        "status": "needs_resolution" if data["issues"] else "completed",
        "pages": data["page_count"],
        "figures": len(data["figures"]),
        "issues": data["issues"],
    }


def figure_info(root, name, item):
    run, assets, _ = paths(root, name)
    work = run / "figures" / item["id"]
    archive = assets / item["id"] / (item["id"] + ".tar")
    saved = read_json(work / "review.json") or {}
    signature = tar_version(archive) if archive.exists() else None
    return {
        **item,
        "ready": archive.is_file(),
        "version": signature,
        "review_status": saved.get("status", "unreviewed")
        if saved.get("version") == signature
        else "unreviewed",
        "processing": read_json(work / "status.json") or {"status": "pending"},
    }


def tar_version(path):
    stat = path.stat()
    # JavaScript numbers cannot preserve a nanosecond timestamp's full precision.
    return [stat.st_size, str(stat.st_mtime_ns)]


def _figure(root, name, item, retry=False):
    from chart_annotator.runner import run_figure

    run, assets, _ = paths(root, name)
    work = run / "figures" / item["id"]
    model_dir = work / "model"
    prior = read_json(work / "status.json") or {}
    if prior.get("status") == "exported" and not retry:
        return prior
    if model_dir.exists():
        if not retry:
            return {
                "status": "needs_resolution",
                "id": item["id"],
                "message": "已有失败或中断记录，检查后用 --retry 重跑本图",
            }
        assert (
            model_dir.resolve().parent == work.resolve()
            and work.resolve().is_relative_to((root / "runs").resolve())
        )
        shutil.rmtree(model_dir)
        (work / "checkpoint.sqlite").unlink(missing_ok=True)
    write_json(work / "status.json", {"status": "running", "id": item["id"]})
    try:
        result = run_figure(
            assets / item["id"] / (item["id"] + ".pdf"),
            model_dir,
            checkpoint=work / "checkpoint.sqlite",
            thread_id=item["id"],
            datasets=True,
            source_context={
                "source_id": name + "/" + item["id"],
                "manual_id": name,
                "source_page": item["page"],
                "figure_id": item["id"].removeprefix("Figure_"),
                "figure_job_id": name + "/" + item["id"],
            },
        )
        if result["status"] == "exported":
            archive = assets / item["id"] / (item["id"] + ".tar")
            with lock(run, "publication"):
                if not archive.exists():
                    temporary = archive.with_suffix(".tmp")
                    shutil.copyfile(model_dir / "output/chart.tar", temporary)
                    temporary.replace(archive)
    except Exception as exc:
        result = {
            "status": "failed",
            "error": type(exc).__name__,
            "message": "处理失败，请检查本图 model 目录",
        }
    write_json(work / "status.json", result)
    return result


def figures(root, name, ids=None, workers=1, retry=False):
    data = manual(root, name)
    items = [item for item in data["figures"] if not ids or item["id"] in ids]
    if ids and set(ids) != {item["id"] for item in items}:
        raise ValueError("存在未知 Figure")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers 必须至少为 1")
    results = []
    if workers == 1:
        results = [_figure(root, name, item, retry) for item in items]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            futures = [pool.submit(_figure, root, name, item, retry) for item in items]
            for future in as_completed(futures):
                results.append(future.result())
    return {
        "status": "completed"
        if all(r["status"] == "exported" for r in results)
        else "needs_resolution",
        "exported": sum(r["status"] == "exported" for r in results),
        "total": len(items),
    }


def tables(root, name):
    from pdf_tree_workflow.luna_processing import apply_luna_results, plan_luna_batches
    from pdf_tree_workflow.mineru_runner import run_mineru
    from pdf_tree_workflow.preparation import prepare_run
    from pdf_tree_workflow.table_processing import process_mineru_results

    data = manual(root, name)
    run, assets, pdf = paths(root, name)
    table_dir = run / "tables"
    if not (table_dir / "manifest.json").exists():
        prepare_run(
            pdf_path=pdf,
            figure_assets=assets,
            run_dir=table_dir,
            figure_mappings=[
                {"physical_page": item["page"], "asset_unit": item["id"]}
                for item in data["figures"]
            ],
        )
    if not (table_dir / "index.json").exists():
        extracted = run_mineru(run_dir=table_dir, env_file=root / ".env")
        if extracted["failed"]:
            return {"status": "needs_resolution", "failed": extracted["failed"]}
        with lock(run, "publication"):
            process_mineru_results(run_dir=table_dir)
    if not (table_dir / "luna_batch_plan.json").exists():
        plan_luna_batches(
            run_dir=table_dir, contract=root / "LUNA_TABLE_REVIEW_CONTRACT.md"
        )
    with lock(run, "publication"):
        result = apply_luna_results(run_dir=table_dir)
    return {
        "status": "waiting_luna" if result["pending_batches"] else "completed",
        **result,
    }


def apply_luna(root, name):
    from pdf_tree_workflow.luna_processing import apply_luna_results

    run, _, _ = paths(root, name)
    with lock(run, "publication"):
        result = apply_luna_results(run_dir=run / "tables")
        result["status"] = "waiting_luna" if result["pending_batches"] else "completed"
        previous = read_json(run / "tasks/tables.json") or {}
        write_json(run / "tasks/tables.json", {**previous, **result})
        return result


def ready_tables(run):
    items = candidates(Path(run) / "tables")
    waiting_keys = {
        item.get("node_id_match_key")
        for item in items
        if item["data_status"] == "review_required"
    }
    return [
        item
        for item in items
        if item["data_status"] != "review_required"
        and (
            not item.get("node_id_match_key")
            or item["node_id_match_key"] not in waiting_keys
        )
    ]


def summary(root, name):
    from pdf_tree_workflow.human_review import read_review
    from pdf_tree_workflow.identifiers import parse_identifier_prefix

    run, assets, pdf = paths(root, name)
    data = read_json(run / "manual.json") or {}
    if data and data.get("schema_version") != 2:
        raise ValueError("不支持旧手册记录；请使用新工作台格式")
    image_items = [figure_info(root, name, item) for item in data.get("figures", [])]
    table_items = candidates(run / "tables")
    table_manifest = read_json(run / "tables/manifest.json") or {}
    figure_keys = {
        parse_identifier_prefix(item["id"].removeprefix("Figure_")).match_key
        for item in image_items
    }
    missing_figures = [
        node
        for node in table_manifest.get("nodes", [])
        if node["node_type"] == "figure"
        and node["node_id_match_key"] not in figure_keys
    ]
    artifact_names = {item["artifact_directory"] for item in table_items}
    missing_tables = [
        node
        for node in table_manifest.get("nodes", [])
        if node["node_type"] == "table"
        and node.get("table_artifact") not in artifact_names
    ]
    confirmed = 0
    for item in table_items:
        try:
            confirmed += (
                read_review(
                    run / "tables" / item["artifact_directory"] / "human_review.json",
                    item,
                )["status"]
                == "verified"
            )
        except ValueError:
            pass
    tasks = {path.stem: read_json(path) for path in (run / "tasks").glob("*.json")}
    # A crashed process releases its OS lock even if its last status said running.
    for action, state in tasks.items():
        if state.get("status") == "running":
            try:
                with lock(run, *operation_locks(action)):
                    state = {**state, "status": "interrupted"}
                    tasks[action] = state
            except ValueError:
                pass
    issues = [
        {
            **issue,
            "subject": "接收手册时发现的问题",
            "next_step": "请 Codex 对照来源 PDF 检查本页的识别结果与 Figure 映射。",
            "files": [str(pdf), str(run / "manual.json")],
        }
        for issue in data.get("issues", [])
    ]
    for node in missing_figures + missing_tables:
        is_figure = node["node_type"] == "figure"
        kind = "Figure" if is_figure else "Table"
        issues.append(
            {
                "page": node.get("target_page"),
                "catalog_pages": node.get("catalog_pages", []),
                "code": f"catalog_{node['node_type']}_missing",
                "subject": f"{kind} {node['node_id_raw']}",
                "title": node.get("title_raw", ""),
                "message": "目录列出了这张图，但接收清单中没有对应的 Figure 资产。"
                if is_figure
                else "目录列出了这张表，但尚无对应的 OCR / Luna 表格候选结果。",
                "next_step": "请 Codex 对照目录标题查找内容页，检查编号与接收清单的映射。"
                if is_figure
                else "请 Codex 检查本表来源页的 OCR 结果和表格编号匹配。",
                "files": [
                    str(pdf),
                    str(run / "tables/manifest.json"),
                    str(
                        run / "manual.json" if is_figure else run / "tables/index.json"
                    ),
                ],
            }
        )
    source_issue_count = len(issues)
    for item in image_items:
        if not item["ready"]:
            work = run / "figures" / item["id"]
            issues.append(
                {
                    "page": item["page"],
                    "code": "figure_tar_missing",
                    "subject": item["id"],
                    "message": "已有来源图像 PDF，但还没有可在 WPD 中打开的标注 TAR。",
                    "next_step": "模型记录已标记导出成功，请 Codex 检查 model/output/chart.tar 和 Figure 资产目录，确认 TAR 是否已正确发布。"
                    if item["processing"]["status"] == "exported"
                    else "请 Codex 检查本图的模型处理记录；未运行则启动识别，失败或中断则查明原因后重试。",
                    "processing_status": item["processing"]["status"],
                    "files": [
                        str(assets / item["id"] / (item["id"] + ".pdf")),
                        str(assets / item["id"] / (item["id"] + ".tar")),
                        str(work),
                    ],
                }
            )
    state = {
        "name": name,
        "intake": bool(data),
        "pages": data.get("page_count", 0),
        "issues": issues,
        "pdf": str(pdf),
        "figures": {
            "total": len(image_items) + len(missing_figures),
            "missing": len(missing_figures),
            "ready": sum(i["ready"] for i in image_items),
            "verified": sum(i["review_status"] == "verified" for i in image_items),
            "failed": sum(
                i["processing"]["status"] in {"failed", "needs_resolution"}
                for i in image_items
            ),
        },
        "tables": {
            "total": len(table_items) + len(missing_tables),
            "missing": len(missing_tables),
            "ready": len(ready_tables(run)),
            "verified": confirmed,
            "partial": sum(i["data_status"] == "partial" for i in table_items),
        },
        "tasks": tasks,
    }
    required, pending = [], []
    if not data:
        required.append("尚未接收手册，请先完成 intake。")
    if not (run / "tables/index.json").is_file():
        required.append("尚无表格处理清单，请让 Codex 完成表格 OCR 处理。")
    for key, label in (("figures", "图像"), ("tables", "表格")):
        counts = state[key]
        if counts["total"] > counts["verified"]:
            pending.append(
                f"{label}：共 {counts['total']} 项，已确认 {counts['verified']} 项，"
                f"还有 {counts['total'] - counts['verified']} 项待确认"
                f"（其中 {counts['total'] - counts['ready']} 项尚未就绪）。"
            )
    if source_issue_count:
        pending.append(
            f"另有 {source_issue_count} 项来源或目录匹配问题，详见下方问题详情。"
        )
    if state["tables"]["partial"]:
        pending.append(f"{state['tables']['partial']} 张表的内容仍不完整。")
    state["assembly"] = {"required": required, "pending": pending}
    return state


class AssemblyBlocked(ValueError):
    """The requested assembly mode has unmet prerequisites, not a processing error."""


def require_assembly(state, allow_incomplete=False):
    check = state["assembly"]
    reasons = check["required"] + ([] if allow_incomplete else check["pending"])
    if reasons:
        hint = (
            "请先完成上述准备。"
            if check["required"]
            else "请完成核验，或勾选“允许带未完成项组装”。"
        )
        raise AssemblyBlocked("暂不能组装：" + " ".join(reasons) + " " + hint)


def assemble(root, name, allow_incomplete=False):
    from pdf_tree_workflow.builder import build_tree

    state = summary(root, name)
    table_dir = root / "runs" / name / "tables"
    require_assembly(state, allow_incomplete)
    data = manual(root, name)
    incomplete = bool(state["assembly"]["pending"])
    run, assets, pdf = paths(root, name)
    result = build_tree(
        pdf_path=pdf,
        figure_assets=assets,
        figure_records=data["figures"],
        output_parent=root / "library",
        table_run_dir=table_dir,
        force=True,
    )
    write_json(
        root / "library" / name / "REVIEW_STATUS.json",
        {
            "incomplete": bool(incomplete),
            "figures": state["figures"],
            "tables": state["tables"],
            "issues": state["issues"],
        },
    )
    return {
        "status": "completed",
        "output": str(root / "library" / name),
        "incomplete": bool(incomplete),
        **result,
    }


def operation_locks(action):
    return (
        ("figures", "tables", "publication")
        if action in {"intake", "assemble"}
        else ("figures",)
        if action == "figures"
        else ("tables",)
    )


def execute(root, name, action, **options):
    root = Path(root).resolve()
    run, _, _ = paths(root, name)
    actions = {
        "intake": intake,
        "figures": figures,
        "tables": tables,
        "apply-luna": apply_luna,
        "assemble": assemble,
    }
    if action not in actions:
        raise ValueError("未知操作")
    with lock(run, *operation_locks(action)):
        record = {"status": "running", "started": datetime.now(UTC).isoformat()}
        write_json(run / "tasks" / (action + ".json"), record)
        try:
            result = actions[action](root, name, **options)
        except Exception as exc:
            from dotenv import dotenv_values

            error = str(exc)
            for key, value in dotenv_values(root / ".env").items():
                if value and any(part in key for part in ("KEY", "TOKEN", "SECRET")):
                    error = error.replace(value, "[credential]")
            write_json(
                run / "tasks" / (action + ".json"),
                {
                    **record,
                    "status": "blocked"
                    if isinstance(exc, AssemblyBlocked)
                    else "failed",
                    "error": error,
                    "finished": datetime.now(UTC).isoformat(),
                },
            )
            if isinstance(exc, AssemblyBlocked):
                raise AssemblyBlocked(error) from None
            raise ValueError(error) from None
        write_json(
            run / "tasks" / (action + ".json"),
            {**record, **result, "status": result.get("status", "completed")},
        )
        return result
