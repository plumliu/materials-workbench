from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf

from .identifiers import parse_caption
from .model import Node, ParsedManual
from .page_sources import (
    page_review_directory,
)
from .parser import parse_manual


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _node_record(node: Node) -> dict[str, Any]:
    return {
        "sequence": node.sequence,
        "node_type": node.node_type,
        "node_id_raw": node.node_id_raw,
        "synthetic_id": node.synthetic_id,
        "identifier_status": node.identifier_status,
        "node_id_core": node.identifier_core,
        "node_id_suffixes": list(node.identifier_suffixes),
        "node_id_match_key": node.identifier_match_key,
        "title_raw": node.title_raw,
        "parent_sequence": node.parent.sequence if node.parent else None,
        "catalog_pages": sorted(node.catalog_pages),
        "source_physical_pages": node.source_pages,
        "target_page": node.target_page,
        "data_status": "pending" if node.node_type == "table" else None,
    }


def _caption_lines(text: str, kind: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        candidate = line.strip()
        parsed = parse_caption(candidate, expected_kind=kind)
        if parsed is None and index + 1 < len(lines):
            candidate = candidate + " " + lines[index + 1].strip()
            parsed = parse_caption(candidate, expected_kind=kind)
        if parsed:
            results.append(
                {
                    "raw": candidate,
                    "node_id_raw": parsed.identifier.raw,
                    "match_key": parsed.identifier.match_key,
                    "title": parsed.title,
                }
            )
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in results:
        unique[(item["match_key"], item["raw"])] = item
    return list(unique.values())


def _contiguous_runs(pages: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for page in sorted(set(pages)):
        if not runs or page != runs[-1][-1] + 1:
            runs.append([page])
        else:
            runs[-1].append(page)
    return runs


def _write_segment(pdf_path: Path, pages: list[int], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as source, pymupdf.open() as segment:
        for physical_page in pages:
            segment.insert_pdf(source, from_page=physical_page - 1, to_page=physical_page - 1)
        segment.save(target)


def prepare_run(
    *,
    pdf_path: Path,
    figure_assets: Path,
    run_dir: Path,
    force: bool = False,
    figure_mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    pdf_path = pdf_path.expanduser().resolve()
    figure_assets = figure_assets.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if not figure_assets.is_dir():
        raise FileNotFoundError(figure_assets)
    run_dir.mkdir(parents=True, exist_ok=True)
    if list(run_dir.glob("Table_*/human_review.json")):
        raise ValueError("Existing human edits require explicit reconciliation")
    for name in ("segments", "ocr"):
        path = run_dir / name
        if path.exists():
            import shutil

            shutil.rmtree(path)

    manual: ParsedManual = parse_manual(pdf_path)
    page_review_dir = page_review_directory(figure_assets)
    mapped_figure_pages = {
        int(item["physical_page"])
        for item in figure_mappings
        if item.get("physical_page") is not None
    }
    known_table_pages = {
        node.target_page
        for node in manual.nodes
        if node.node_type == "table" and node.target_page is not None
    }

    metadata_pages = set(manual.front_pages)
    reference_pages = set(manual.reference_pages)
    page_records: list[dict[str, Any]] = []
    candidate_pages: list[int] = []
    document = pymupdf.open(pdf_path)
    try:
        for physical_page, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            tables = _caption_lines(text, "table")
            figures = _caption_lines(text, "figure")
            if physical_page in metadata_pages:
                classification = "metadata"
                include = False
            elif physical_page in reference_pages:
                classification = "reference"
                include = False
            elif physical_page in mapped_figure_pages and (
                tables or physical_page in known_table_pages
            ):
                classification = "mixed_figure_table"
                include = True
            elif physical_page in mapped_figure_pages:
                classification = "pure_figure"
                include = False
            elif tables or physical_page in known_table_pages:
                classification = "table_candidate"
                include = True
            else:
                # Retaining uncertain non-metadata pages is intentional: MinerU
                # is the table detector of record, and false negatives here are
                # more damaging than a harmless empty segment.
                classification = "unresolved"
                include = True
            if include:
                candidate_pages.append(physical_page)
            page_records.append(
                {
                    "physical_page": physical_page,
                    "classification": classification,
                    "include_in_table_candidate": include,
                    "mapped_figure_asset": physical_page in mapped_figure_pages,
                    "table_captions": tables,
                    "figure_captions": figures,
                    "text_characters": len(text.strip()),
                }
            )
    finally:
        document.close()

    segments: list[dict[str, Any]] = []
    for index, pages in enumerate(_contiguous_runs(candidate_pages), start=1):
        segment_id = f"segment_{index:03d}__pages_{pages[0]:04d}-{pages[-1]:04d}"
        relative = Path("segments") / f"{segment_id}.pdf"
        _write_segment(pdf_path, pages, run_dir / relative)
        segments.append(
            {
                "segment_id": segment_id,
                "pdf": relative.as_posix(),
                "physical_pages": pages,
                "segment_page_to_physical_page": {
                    str(local): physical for local, physical in enumerate(pages)
                },
            }
        )

    manifest = {
        "schema_version": 2,
        "manual_pdf": str(pdf_path),
        "page_review_dir": str(page_review_dir),
        "figure_assets": str(figure_assets),
        "page_count": manual.page_count,
        "front_pages": manual.front_pages,
        "reference_pages": manual.reference_pages,
        "nodes": [_node_record(node) for node in manual.nodes],
        "structure_validation": manual.structure_validation,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "page_manifest.json", page_records)
    _write_json(run_dir / "segments.json", segments)
    summary = {
        "run_dir": str(run_dir),
        "manual_pdf": str(pdf_path),
        "page_review_dir": str(page_review_dir),
        "page_count": manual.page_count,
        "mapped_figure_assets": sum(
            1 for item in figure_mappings if item.get("physical_page") is not None
        ),
        "unresolved_figure_assets": sum(
            1 for item in figure_mappings if item.get("physical_page") is None
        ),
        "mapped_figure_pages": len(mapped_figure_pages),
        "known_table_pages": len(known_table_pages),
        "candidate_pages": len(candidate_pages),
        "segments": len(segments),
    }
    _write_json(run_dir / "prepare_summary.json", summary)
    return summary
