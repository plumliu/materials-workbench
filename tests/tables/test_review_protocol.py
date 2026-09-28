import json
from pathlib import Path
from copy import deepcopy

import pytest
from openpyxl import load_workbook

from pdf_tree_workflow.luna_processing import (
    _apply_operation,
    _sync_component_grid,
    _validate_formula_review,
    _validate_patch,
    plan_luna_batches,
)
from pdf_tree_workflow.table_processing import _review_request, _write_workbook


def test_review_request_omits_repeated_cell_text():
    candidate = {
        "candidate_id": "table-1",
        "node_id_raw": "1.2",
        "synthetic_id": None,
        "caption": "Table 1.2",
        "footnotes": "",
        "source_physical_pages": [3],
        "source_fragments": [{"fragment_id": "f1", "segment_id": "s1"}],
        "grid": {
            "rows": [["A", "B"]],
            "cells": [
                {"cell_id": "R001C001", "rowspan": 1, "colspan": 2, "text": "A"},
                {"cell_id": "R001C002", "rowspan": 1, "colspan": 1, "text": "B"},
            ],
            "embedded_media": [],
        },
        "components": [],
        "issues": [{"code": "merged_region", "severity": "review"}],
    }
    packet = _review_request(candidate)
    assert packet["rows"] == [["A", "B"]]
    assert packet["merged_cells"] == [
        {"cell_id": "R001C001", "rowspan": 1, "colspan": 2}
    ]
    assert "cells" not in packet
    assert packet["issues"] == candidate["issues"]


def test_batch_manifest_contains_only_luna_inputs(tmp_path):
    table_dir = tmp_path / "Table_1"
    table_dir.mkdir(parents=True)
    (table_dir / "candidate.json").write_text(
        json.dumps(
            {
                "schema_version": 2, "revision": 1, "candidate_id": "table-1", "data_status": "review_required",
                "source_physical_pages": [3],
                "grid": {"cells": [{"cell_id": "R001C001"}]},
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    import pymupdf
    (tmp_path / "manifest.json").write_text(json.dumps({"page_review_dir": str(tmp_path)}))
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(tmp_path / "page_0003.pdf")
    contract = tmp_path / "contract.md"
    contract.write_text("contract", encoding="utf-8")
    plan = plan_luna_batches(run_dir=tmp_path, contract=contract)
    assert Path(plan["contract_file"]).read_text(encoding="utf-8") == "contract"
    manifest = json.loads(Path(plan["batches"][0]["input_manifest"]).read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "2.0"
    assert set(manifest["assigned_candidates"][0]) == {
        "candidate_id", "base_revision", "table_directory", "source_physical_pages", "sources"
    }


def test_formula_review_checks_unique_page_coverage():
    assigned = [
        {"candidate_id": "table-1", "source_physical_pages": [3, 4]},
        {"candidate_id": "table-2", "source_physical_pages": [4]},
    ]
    report = {
        "formula_review": [
            {"physical_page": 3, "status": "clean", "findings": []},
            {
                "physical_page": 4,
                "status": "issue",
                "findings": [{
                    "location": "Table 1.2, row 2",
                    "artifact": "review_request.json",
                    "observed": "10^3",
                    "source_visible": "10 to the power of minus 3",
                    "suggested_latex": "10^{-3}",
                    "kind": "exponent_sign",
                    "candidate_id": None,
                }],
            },
        ]
    }
    pages, exceptions = _validate_formula_review(report, assigned, "batch_001")
    assert pages == {3, 4}
    assert [item["physical_page"] for item in exceptions] == [4]
    invalid = deepcopy(report)
    invalid["formula_review"][1]["findings"][0]["candidate_id"] = "table-2"
    invalid["formula_review"][1]["physical_page"] = 3
    invalid["formula_review"][0]["physical_page"] = 4
    with pytest.raises(ValueError, match="finding"):
        _validate_formula_review(invalid, assigned, "batch_001")
    report["formula_review"][1]["findings"] = [{}]
    with pytest.raises(ValueError, match="finding"):
        _validate_formula_review(report, assigned, "batch_001")
    report["formula_review"][1]["physical_page"] = "4"
    with pytest.raises(ValueError, match="page"):
        _validate_formula_review(report, assigned, "batch_001")
    report["formula_review"].pop()
    with pytest.raises(ValueError, match="coverage mismatch"):
        _validate_formula_review(report, assigned, "batch_001")


def test_new_patch_rejects_untraceable_resolution_and_evidence(tmp_path):
    candidate = {
        "candidate_id": "table-1",
        "source_physical_pages": [3],
        "issues": [{"code": "numeric_ocr", "severity": "review"}],
    }
    patch = {
        "schema_version": 1,
        "contract_version": "2.0",
        "batch_id": "batch_001",
        "candidate_id": "table-1",
        "table_directory": str(tmp_path),
        "decision": "patch",
        "reviewed_source_pages": [3],
        "issue_resolutions": [{"issue_code": "numeric_ocr", "resolution": "corrected", "reason": "Source shows 5."}],
        "operations": [{"op_id": "op_001", "op": "set_cell", "target": {"cell_id": "R001C001"}, "before": "S", "after": "5", "confidence": "high", "reason": "Source shows 5.", "evidence": [{"physical_page": 3, "description": "first cell"}]}],
        "unresolved": [],
    }
    _validate_patch(candidate, patch, "batch_001", tmp_path, "2.0")
    invalid = deepcopy(patch)
    invalid["issue_resolutions"][0].update(resolution="potato", reason="")
    with pytest.raises(ValueError, match="issue resolution"):
        _validate_patch(candidate, invalid, "batch_001", tmp_path, "2.0")
    invalid = deepcopy(patch)
    invalid["issue_resolutions"][0]["resolution"] = "unresolved"
    with pytest.raises(ValueError, match="partial decision"):
        _validate_patch(candidate, invalid, "batch_001", tmp_path, "2.0")
    invalid["decision"] = "partial"
    invalid["unresolved"] = [{
        "code": "unrelated_issue", "description": "Still unclear",
        "consequence": "Value remains uncertain", "evidence_pages": [3],
    }]
    with pytest.raises(ValueError, match="missing entries"):
        _validate_patch(candidate, invalid, "batch_001", tmp_path, "2.0")
    invalid["unresolved"][0]["code"] = "numeric_ocr"
    _validate_patch(candidate, invalid, "batch_001", tmp_path, "2.0")
    invalid = deepcopy(patch)
    invalid["operations"][0]["evidence"] = [{"physical_page": 999999, "description": ""}]
    with pytest.raises(ValueError, match="source-page evidence"):
        _validate_patch(candidate, invalid, "batch_001", tmp_path, "2.0")


@pytest.mark.parametrize("component_count", [1, 2])
def test_cell_patch_reaches_component_workbook(tmp_path, component_count):
    def grid(value):
        return {
            "rows": [[value]],
            "cells": [{"cell_id": "R001C001", "row": 1, "column": 1, "rowspan": 1, "colspan": 1, "text": value}],
        }

    components = [{"component_id": "component_001", "grid": grid("old")}]
    rows = [["old"]]
    if component_count == 2:
        components.append({"component_id": "component_002", "grid": grid("second")})
        rows.append(["second"])
    candidate = {
        "node_id_raw": "1.2",
        "caption": "Table 1.2",
        "source_physical_pages": [3],
        "data_status": "review_required",
        "components": components,
        "grid": {"rows": rows, "cells": []},
    }
    target_row = component_count
    old_value = rows[target_row - 1][0]
    operation = {
        "op": "set_cell",
        "target": {"cell_id": f"R{target_row:03d}C001"},
        "before": old_value,
        "after": "new",
    }
    _apply_operation(candidate, operation)
    _sync_component_grid(candidate, operation)
    output = tmp_path / "table.xlsx"
    _write_workbook(output, candidate)
    workbook = load_workbook(output, read_only=True)
    try:
        assert workbook.worksheets[component_count - 1]["A1"].value == "new"
    finally:
        workbook.close()


def test_cell_patch_rejects_misaligned_component_rows():
    values = ["A", "X", "B", "Y", "B", "B", "Q"]
    candidate = {
        "grid": {"rows": [[value] for value in values], "cells": []},
        "components": [
            {"grid": {"rows": [[value] for value in part]}}
            for part in (["A", "X"], ["B", "Y", "B"], ["A", "Q"])
        ],
    }
    operation = {
        "op": "set_cell", "target": {"cell_id": "R005C001"},
        "before": "B", "after": "new",
    }
    _apply_operation(candidate, operation)
    with pytest.raises(ValueError, match="do not align"):
        _sync_component_grid(candidate, operation)
