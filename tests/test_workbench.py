from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import tarfile
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pymupdf
import pytest

from materials_workbench import workflow
from materials_workbench.server import Handler
from materials_workbench.storage import candidates, lock, read_json, write_json
from chart_annotator.wpd import read as read_tar

FIXTURES = Path(__file__).parent / "charts/fixtures/charts"


def prepared(root):
    (root / "pdfs").mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 100), "1 Test materials", fontname="hebo", fontsize=14)
        page.insert_text((40, 150), "1.2 [Figure] Chart")
        page.insert_text((40, 200), "1.3 [Table] Measurement results")
        pdf.new_page().insert_text((40, 50), "REFERENCES\n1. Test handbook reference.")
        page = pdf.new_page()
        page.insert_text(
            (40, 50), "Figure 1.2 A sufficiently detailed chart caption for intake"
        )
        page.insert_text((40, 150), "Table 1.3 Measurement results")
        page = pdf.new_page()
        page.insert_text((40, 50), "Table 1.3 Measurement results (continued)")
        pdf.save(root / "pdfs/Manual.pdf")
    shutil.copy2(
        Path(__file__).parents[1] / "LUNA_TABLE_REVIEW_CONTRACT.md",
        root / "LUNA_TABLE_REVIEW_CONTRACT.md",
    )
    workflow.execute(root, "Manual", "intake")
    return root / "runs/Manual"


def archive_bytes(path, change=None):
    with tarfile.open(path) as archive:
        entries = {
            member.name: archive.extractfile(member).read()
            for member in archive
            if member.isfile()
        }
    for name in entries:
        if name.endswith("/wpd.json") and change:
            data = json.loads(entries[name])
            change(data)
            entries[name] = json.dumps(data).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


@contextmanager
def service(root):
    class TestHandler(Handler):
        pass

    TestHandler.root = root
    TestHandler.runs_dir = root / "runs"
    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_intake_parallel_table_processing_and_final_assembly(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    assert (tmp_path / "figure_assets/Manual/_page_review/page_0002.pdf").exists()
    assert read_json(run / "manual.json")["figures"][0]["id"] == "Figure_1.2"
    assert not list(run.rglob("*sha256*"))
    with pytest.raises(ValueError, match="已有"):
        workflow.execute(tmp_path, "Manual", "intake")

    def fake_ocr(*, run_dir, **kwargs):
        for segment in read_json(run_dir / "segments.json"):
            write_json(
                run_dir / "ocr" / segment["segment_id"] / "content_list.json",
                [
                    {
                        "type": "table",
                        "page_idx": index,
                        "table_caption": ["Table 1.3 Measurement results"],
                        "table_body": f"<table><tr><td>Property</td><td>Value</td></tr><tr><td>Row {index}</td><td>5</td></tr></table>",
                    }
                    for index in range(len(segment["physical_pages"]))
                ],
            )
        return {"failed": []}

    monkeypatch.setattr("pdf_tree_workflow.mineru_runner.run_mineru", fake_ocr)
    # Figure work and table OCR can coexist, while intake and assembly cannot.
    with lock(run, "figures"):
        result = workflow.execute(tmp_path, "Manual", "tables")
        with pytest.raises(ValueError, match="正在进行"):
            workflow.execute(tmp_path, "Manual", "assemble")
    assert result["status"] in {"waiting_luna", "completed"}
    table_dir = run / "tables"
    segments = read_json(table_dir / "segments.json")
    assert 3 in segments[0]["physical_pages"]  # same page contains a Figure and Table
    items = candidates(table_dir)
    assert any(item["source_physical_pages"] == [3, 4] for item in items)
    assert all("/" not in item["artifact_directory"] for item in items)
    assert not list(table_dir.glob("Table_*/source_pages"))
    with pytest.raises(ValueError, match="待确认"):
        workflow.execute(tmp_path, "Manual", "assemble")
    assert read_json(run / "tasks/assemble.json")["status"] == "blocked"
    # The website rejects unmet prerequisites before spawning any assembly process.
    with service(tmp_path) as url:
        with monkeypatch.context() as patch:

            def no_process(*args, **kwargs):
                pytest.fail("Blocked assembly must not start a subprocess")

            patch.setattr("materials_workbench.server.subprocess.Popen", no_process)
            with pytest.raises(HTTPError) as exc:
                urlopen(
                    Request(
                        url + "/api/action",
                        data=json.dumps(
                            {"manual": "Manual", "action": "assemble"}
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                )
            response = json.loads(exc.value.read())
            assert exc.value.code == 409
            assert response["status"] == "blocked"
            assert "暂不能组装" in response["error"]
            assert "待确认" in response["error"]
    result = workflow.execute(tmp_path, "Manual", "assemble", allow_incomplete=True)
    assert Path(result["output"]).is_dir()
    assert read_json(tmp_path / "library/Manual/REVIEW_STATUS.json")["incomplete"]


def test_overview_distinguishes_catalog_pages_and_content_pages(tmp_path):
    run = prepared(tmp_path)
    write_json(
        run / "tables/manifest.json",
        {
            "nodes": [
                {
                    "node_type": "figure",
                    "node_id_raw": "1.4",
                    "node_id_match_key": "1.4",
                    "title_raw": "Missing chart",
                    "catalog_pages": [1],
                    "target_page": None,
                },
                {
                    "node_type": "figure",
                    "node_id_raw": "1.5",
                    "node_id_match_key": "1.5",
                    "title_raw": "Known page",
                    "catalog_pages": [1],
                    "target_page": 4,
                },
            ]
        },
    )
    state = workflow.summary(tmp_path, "Manual")
    assert state["pages"] == 4
    missing = [i for i in state["issues"] if i["code"] == "catalog_figure_missing"]
    assert [i["page"] for i in missing] == [None, 4]
    assert missing[0]["catalog_pages"] == [1]
    assert str(run / "tables/manifest.json") in missing[0]["files"]
    tar_issue = next(i for i in state["issues"] if i["code"] == "figure_tar_missing")
    assert tar_issue["subject"] == "Figure_1.2" and tar_issue["page"] == 3
    assert state["figures"] == {
        "total": 3,
        "missing": 2,
        "ready": 0,
        "verified": 0,
        "failed": 0,
    }
    with pytest.raises(workflow.AssemblyBlocked, match="表格"):
        workflow.require_assembly(state, allow_incomplete=True)


def test_wpd_save_reopen_and_stale_tab_rejection(tmp_path):
    run = prepared(tmp_path)
    path = tmp_path / "figure_assets/Manual/Figure_1.2/Figure_1.2.tar"
    sample = FIXTURES / "Figure_3.2.1.7/human.tar"
    empty = archive_bytes(
        sample, lambda data: [d.update(data=[]) for d in data["datasetColl"]]
    )
    path.write_bytes(empty)
    with service(tmp_path) as base:
        query = urlencode({"manual": "Manual"})
        info = json.load(urlopen(base + "/api/figures?" + query))[0]
        assert isinstance(info["version"][1], str)  # survives a browser JSON round trip
        edited = archive_bytes(
            path,
            lambda data: data["datasetColl"][0]["data"].append({"x": 100, "y": 120}),
        )
        params = urlencode(
            {
                "manual": "Manual",
                "id": "Figure_1.2",
                "version": json.dumps(info["version"]),
                "status": "draft",
            }
        )
        request = Request(
            base + "/api/figure-save?" + params,
            data=edited,
            headers={"Content-Type": "application/x-tar"},
        )
        saved = json.load(urlopen(request))
        assert saved["review_status"] == "draft"
        assert saved["version"] != info["version"]
        assert read_tar(path)[0]["datasetColl"][0]["data"] == [{"x": 100, "y": 120}]
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 409
        bad_origin = Request(
            base + "/api/figure-save?" + params,
            data=b"{}",
            headers={
                "Content-Type": "application/x-tar",
                "Origin": "https://example.invalid",
            },
        )
        with pytest.raises(HTTPError):
            urlopen(bad_origin)
    assert read_json(run / "figures/Figure_1.2/review.json")["status"] == "draft"
    # A fresh server reads the persisted archive and review state.
    with service(tmp_path) as base:
        reopened = json.load(urlopen(base + "/api/figures?" + query))[0]
        assert reopened["version"] == saved["version"]
        assert reopened["review_status"] == "draft"


def test_model_retry_preserves_human_archive(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    source = FIXTURES / "Figure_3.2.1.7/human.tar"
    current = tmp_path / "figure_assets/Manual/Figure_1.2/Figure_1.2.tar"
    shutil.copy2(source, current)
    before = current.read_bytes()

    def fake_figure(source, output, **kwargs):
        (output / "output").mkdir(parents=True)
        (output / "output/chart.tar").write_bytes(b"new model output")
        return {"status": "exported", "run_dir": str(output)}

    monkeypatch.setattr("chart_annotator.runner.run_figure", fake_figure)
    workflow.execute(tmp_path, "Manual", "figures", retry=True)
    assert current.read_bytes() == before
    assert (
        run / "figures/Figure_1.2/model/output/chart.tar"
    ).read_bytes() == b"new model output"


def test_incremental_luna_application_keeps_prior_formula_results(
    tmp_path, monkeypatch
):
    from pdf_tree_workflow.luna_processing import plan_luna_batches
    from materials_workbench.storage import publish_candidate

    run = prepared(tmp_path)

    def fake_ocr(*, run_dir, **kwargs):
        segment = read_json(run_dir / "segments.json")[0]
        write_json(
            run_dir / "ocr" / segment["segment_id"] / "content_list.json",
            [
                {
                    "type": "table",
                    "page_idx": 0,
                    "table_caption": [f"Table 1.{n} Results"],
                    "bbox": [0, 100 * n, 500, 100 * n + 90],
                    "table_body": "<table><tr><td>Label</td><td>Value</td></tr><tr><td>A</td><td>5</td></tr></table>",
                }
                for n in (3, 4)
            ],
        )
        return {"failed": []}

    monkeypatch.setattr("pdf_tree_workflow.mineru_runner.run_mineru", fake_ocr)
    workflow.execute(tmp_path, "Manual", "tables")
    tables = run / "tables"
    items = candidates(tables)
    assert len(items) == 2
    for item in items:
        item["data_status"] = "review_required"
        item["issues"] = [{"code": "check", "severity": "review"}]
        publish_candidate(tables / item["artifact_directory"], item)
    plan = plan_luna_batches(
        run_dir=tables,
        contract=tmp_path / "LUNA_TABLE_REVIEW_CONTRACT.md",
        force=True,
        max_tables=1,
    )
    for number, batch in enumerate(plan["batches"], 1):
        assigned = read_json(Path(batch["input_manifest"]))["assigned_candidates"][0]
        output = Path(batch["output_directory"])
        patch = {
            "schema_version": 1,
            "contract_version": "2.0",
            "batch_id": batch["batch_id"],
            "candidate_id": assigned["candidate_id"],
            "table_directory": assigned["table_directory"],
            "decision": "accept",
            "reviewed_source_pages": assigned["source_physical_pages"],
            "issue_resolutions": [
                {
                    "issue_code": "check",
                    "resolution": "confirmed",
                    "reason": "Checked test source",
                }
            ],
            "operations": [],
            "unresolved": [],
        }
        write_json(output / "patches/check.json", patch)
        write_json(
            output / "batch_report.json",
            {
                "schema_version": 1,
                "contract_version": "2.0",
                "batch_id": batch["batch_id"],
                "results": [
                    {
                        "candidate_id": assigned["candidate_id"],
                        "patch_file": "patches/check.json",
                    }
                ],
                "formula_review": [
                    {"physical_page": page, "status": "clean", "findings": []}
                    for page in assigned["source_physical_pages"]
                ],
            },
        )
        result = workflow.execute(tmp_path, "Manual", "apply-luna")
        assert len(workflow.ready_tables(run)) == number
        assert len(result["pending_batches"]) == 2 - number
    versions = [item["revision"] for item in candidates(tables)]
    again = workflow.execute(tmp_path, "Manual", "apply-luna")
    assert again["formula_pages_reviewed"] == result["formula_pages_reviewed"] > 0
    assert [item["revision"] for item in candidates(tables)] == versions
    assert read_json(run / "tasks/tables.json")["status"] == "completed"


def test_old_contracts_are_rejected(tmp_path):
    from pdf_tree_workflow.human_review import editable_review

    run = prepared(tmp_path)
    record = read_json(run / "manual.json")
    record["schema_version"] = 1
    write_json(run / "manual.json", record)
    with pytest.raises(ValueError, match="旧手册"):
        workflow.summary(tmp_path, "Manual")
    review = tmp_path / "old_review.json"
    write_json(review, {"schema_version": 1, "candidate_id": "a", "revision": 1})
    with pytest.raises(ValueError):
        editable_review(review, {"candidate_id": "a", "revision": 1})


def test_catalog_omissions_do_not_drop_intake_figures(tmp_path):
    from pdf_tree_workflow.builder import _merge_intake_figures
    from pdf_tree_workflow.model import Node, ParsedManual

    parent = Node(
        sequence=0, node_id_raw="1", title_raw="Materials", node_type="section"
    )
    parsed = ParsedManual(tmp_path / "x.pdf", 3, [1], [2], [parent], {}, {})
    records = [{"id": "Figure_1.2", "page": 3, "caption": "Figure 1.2 Plot (Ref. 1)"}]
    _merge_intake_figures(parsed, records)
    _merge_intake_figures(parsed, records)
    assert len(parsed.nodes) == 2
    assert parsed.nodes[1].parent is parent
    assert parsed.nodes[1].source_pages == [3]
