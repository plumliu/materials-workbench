from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import re
import shutil
from typing import Any

from .table_processing import _notes, _render_candidate_preview, _write_workbook, _write_json
from materials_workbench.storage import candidates as load_candidates, write_index


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(candidate: dict[str, Any]) -> str:
    return candidate.get("node_id_match_key") or candidate.get("synthetic_id") or candidate["candidate_id"]


def _set_cell(candidate: dict[str, Any], cell_id: str, before: str, after: str) -> None:
    match = re.fullmatch(r"R(\d{3})C(\d{3})", cell_id)
    if not match:
        raise ValueError(f"Invalid cell id: {cell_id}")
    row, column = int(match.group(1)), int(match.group(2))

    def update(grid: dict[str, Any]) -> bool:
        rows = grid["rows"]
        if row > len(rows) or column > len(rows[row - 1]):
            return False
        if rows[row - 1][column - 1] != before:
            return False
        rows[row - 1][column - 1] = after
        for cell in grid.get("cells", []):
            if cell.get("cell_id") == cell_id:
                cell["text"] = after
        return True

    if not update(candidate["grid"]):
        raise ValueError(f"Cell old-value mismatch at {cell_id}")
    components = candidate.get("components") or []
    if len(components) == 1:
        if not update(components[0]["grid"]):
            raise ValueError(f"Component old-value mismatch at {cell_id}")


def reconcile_reparse(
    *,
    run_dir: Path,
    reparse_run_dir: Path,
    reviewed_overrides: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Reconcile a parser-only rerun with already-reviewed final artifacts.

    Existing logical Tables retain their accepted grids.  Their provenance pages
    are narrowed to the fragments found by the new parser.  Newly discovered
    unnumbered Tables are imported only when an explicit source-review override
    is supplied, so parser changes cannot silently promote OCR-only content.
    """

    run_dir = run_dir.expanduser().resolve()
    reparse_run_dir = reparse_run_dir.expanduser().resolve()
    reviewed_overrides = reviewed_overrides.expanduser().resolve()
    overrides = _read_json(reviewed_overrides)
    previous_summary_path = run_dir / "reconciliation_summary.json"
    previous_summary = (
        _read_json(previous_summary_path) if previous_summary_path.is_file() else None
    )
    snapshot = reparse_run_dir
    current_candidates = load_candidates(run_dir)
    reparse_candidates = load_candidates(reparse_run_dir)
    current_by_identity = {_identity(item): item for item in current_candidates}
    reparse_by_identity = {_identity(item): item for item in reparse_candidates}

    narrowed: list[dict[str, Any]] = []
    for identity, current in current_by_identity.items():
        replacement = reparse_by_identity.get(identity)
        if replacement is None:
            continue
        old_pages = list(current.get("source_physical_pages", []))
        new_pages = list(replacement.get("source_physical_pages", []))
        if old_pages == new_pages:
            continue
        if not set(new_pages).issubset(old_pages):
            raise ValueError(
                f"Reparse expands source pages for {identity}: {old_pages} -> {new_pages}; explicit structural review required"
            )
        current["source_physical_pages"] = new_pages
        current["source_fragments"] = copy.deepcopy(replacement.get("source_fragments", []))
        current["components"] = copy.deepcopy(replacement.get("components", []))
        artifact_dir = run_dir / current["artifact_directory"]
        artifact_candidate = _read_json(artifact_dir / "candidate.json")
        artifact_candidate["source_physical_pages"] = new_pages
        artifact_candidate["source_fragments"] = current["source_fragments"]
        artifact_candidate["components"] = current["components"]
        _write_json(artifact_dir / "candidate.json", artifact_candidate)
        provenance_path = artifact_dir / "table_provenance.json"
        provenance = _read_json(provenance_path)
        provenance["source_physical_pages"] = new_pages
        provenance.setdefault("reconciliation", []).append(
            {
                "reason": "fragment_boundary_reparse",
                "old_source_physical_pages": old_pages,
                "new_source_physical_pages": new_pages,
                "reparse_snapshot": str(snapshot),
            }
        )
        _write_json(provenance_path, provenance)
        narrowed.append(
            {"identity": identity, "old_pages": old_pages, "new_pages": new_pages}
        )

    imported: list[dict[str, Any]] = []
    for candidate in reparse_candidates:
        synthetic_id = candidate.get("synthetic_id")
        if not synthetic_id or synthetic_id in current_by_identity:
            continue
        override = overrides.get("tables", {}).get(synthetic_id)
        if not override or override.get("reviewed") is not True:
            raise ValueError(
                f"New unnumbered Table {synthetic_id} lacks an explicit reviewed override"
            )
        revised = copy.deepcopy(candidate)
        operations: list[dict[str, Any]] = []
        for item in override.get("set_cells", []):
            _set_cell(revised, item["cell_id"], item["before"], item["after"])
            operations.append(copy.deepcopy(item))
        revised["data_status"] = "source_reviewed"
        revised["validation_route"] = "mineru_plus_rules_plus_source_review"
        source_artifact = reparse_run_dir / candidate["artifact_directory"]
        destination = run_dir / source_artifact.name
        if destination.exists():
            if not force:
                raise FileExistsError(destination)
            shutil.rmtree(destination)
        shutil.copytree(source_artifact, destination)
        revised["artifact_directory"] = destination.relative_to(run_dir).as_posix()
        _write_json(destination / "candidate.json", revised)
        _write_workbook(destination / "table.xlsx", revised)
        _render_candidate_preview(revised, destination / "candidate_preview.png")
        note = _notes(revised)
        note += "\n## Source review\n\n" + str(override["review_note"]) + "\n"
        (destination / "notes.md").write_text(note, encoding="utf-8")
        provenance = _read_json(destination / "table_provenance.json")
        provenance["validation_route"] = revised["validation_route"]
        provenance["unresolved_issues"] = []
        provenance["source_review"] = {
            "reviewer_role": "main_agent",
            "source_pages": revised["source_physical_pages"],
            "review_note": override["review_note"],
            "operations": operations,
        }
        _write_json(destination / "table_provenance.json", provenance)
        _write_json(
            destination / "source_review_patch.json",
            {
                "schema_version": 1,
                "synthetic_id": synthetic_id,
                "source_physical_pages": revised["source_physical_pages"],
                "operations": operations,
                "review_note": override["review_note"],
            },
        )
        current_candidates.append(revised)
        current_by_identity[synthetic_id] = revised
        imported.append(
            {
                "synthetic_id": synthetic_id,
                "title_raw": revised.get("title_raw"),
                "source_physical_pages": revised["source_physical_pages"],
                "artifact_directory": revised["artifact_directory"],
            }
        )

    write_index(run_dir, current_candidates)
    current_manifest = _read_json(run_dir / "manifest.json")
    reparse_manifest = _read_json(reparse_run_dir / "manifest.json")
    final_by_key = {_identity(item): item for item in current_candidates}
    for node in current_manifest["nodes"]:
        if node.get("node_type") != "table":
            continue
        identity = node.get("node_id_match_key") or node.get("synthetic_id")
        candidate = final_by_key.get(identity)
        if candidate:
            node["source_physical_pages"] = candidate["source_physical_pages"]
            node["data_status"] = candidate["data_status"]
            node["table_artifact"] = candidate["artifact_directory"]
    existing_synthetic = {
        node.get("synthetic_id") for node in current_manifest["nodes"] if node.get("synthetic_id")
    }
    for node in reparse_manifest["nodes"]:
        synthetic_id = node.get("synthetic_id")
        if not synthetic_id or synthetic_id in existing_synthetic:
            continue
        candidate = final_by_key[synthetic_id]
        record = copy.deepcopy(node)
        record["source_physical_pages"] = candidate["source_physical_pages"]
        record["data_status"] = candidate["data_status"]
        record["table_artifact"] = candidate["artifact_directory"]
        current_manifest["nodes"].append(record)
        existing_synthetic.add(synthetic_id)
    _write_json(run_dir / "manifest.json", current_manifest)
    for candidate in current_candidates:
        provenance_path = run_dir / candidate["artifact_directory"] / "table_provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = _read_json(provenance_path)
        changed = False
        for entry in provenance.get("reconciliation", []):
            if "reparse_run_dir" in entry:
                entry.pop("reparse_run_dir", None)
                entry["reparse_snapshot"] = str(snapshot)
                changed = True
        if changed:
            _write_json(provenance_path, provenance)
    counts = Counter(item.get("data_status") for item in current_candidates)
    result = {
        "schema_version": 1,
        "reparse_snapshot": str(snapshot),
        "narrowed_existing_tables": narrowed,
        "imported_unnumbered_tables": imported,
        "logical_tables": len(current_candidates),
        "statuses": dict(counts),
    }
    if not narrowed and not imported and previous_summary:
        result["narrowed_existing_tables"] = previous_summary.get(
            "narrowed_existing_tables", []
        )
        result["imported_unnumbered_tables"] = previous_summary.get(
            "imported_unnumbered_tables", []
        )
    _write_json(run_dir / "reconciliation_summary.json", result)
    return result


def materialize_components(*, run_dir: Path, reparse_snapshot: Path) -> dict[str, Any]:
    """Apply reparse component boundaries without changing accepted cell values."""

    run_dir = run_dir.expanduser().resolve()
    reparse_snapshot = reparse_snapshot.expanduser().resolve()
    candidates = load_candidates(run_dir)
    reparsed = load_candidates(reparse_snapshot)
    reparsed_by_identity = {_identity(item): item for item in reparsed}
    updated: list[dict[str, Any]] = []
    for candidate in candidates:
        replacement = reparsed_by_identity.get(_identity(candidate))
        replacement_components = (replacement or {}).get("components") or []
        if len(replacement_components) <= 1 or len(candidate.get("components") or []) > 1:
            continue
        row_counts = [len(component["grid"]["rows"]) for component in replacement_components]
        accepted_rows = candidate["grid"]["rows"]
        if sum(row_counts) != len(accepted_rows):
            raise ValueError(
                f"Component row boundary mismatch for {_identity(candidate)}: {row_counts} vs {len(accepted_rows)} rows"
            )
        components = copy.deepcopy(replacement_components)
        offset = 0
        for component, row_count in zip(components, row_counts, strict=True):
            rows = copy.deepcopy(accepted_rows[offset : offset + row_count])
            component["grid"]["rows"] = rows
            for cell in component["grid"].get("cells", []):
                row, column = int(cell["row"]), int(cell["column"])
                if row <= len(rows) and column <= len(rows[row - 1]):
                    cell["text"] = rows[row - 1][column - 1]
            offset += row_count
        candidate["components"] = components
        artifact_dir = run_dir / candidate["artifact_directory"]
        artifact_candidate = _read_json(artifact_dir / "candidate.json")
        artifact_candidate["components"] = copy.deepcopy(components)
        _write_json(artifact_dir / "candidate.json", artifact_candidate)
        _write_workbook(artifact_dir / "table.xlsx", artifact_candidate)
        provenance_path = artifact_dir / "table_provenance.json"
        provenance = _read_json(provenance_path)
        provenance["components"] = [
            {
                "component_id": component["component_id"],
                "source_physical_pages": component.get("source_physical_pages", []),
                "fragment_ids": component.get("fragment_ids", []),
                "rows": len(component["grid"]["rows"]),
                "columns": max(
                    (len(row) for row in component["grid"]["rows"]), default=0
                ),
            }
            for component in components
        ]
        provenance.setdefault("reconciliation", []).append(
            {
                "reason": "materialize_component_boundaries",
                "reparse_snapshot": str(reparse_snapshot),
                "accepted_cell_values_changed": False,
            }
        )
        _write_json(provenance_path, provenance)
        updated.append(
            {
                "identity": _identity(candidate),
                "component_count": len(components),
                "row_counts": row_counts,
            }
        )
    write_index(run_dir, candidates)
    result = {"updated_tables": updated, "count": len(updated)}
    _write_json(run_dir / "component_materialization_summary.json", result)
    return result
