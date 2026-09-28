"""Local browser UI for the final human Table check."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import pymupdf

from materials_workbench.storage import source_page, lock, child
from materials_workbench.workflow import ready_tables

from .human_review import read_review, editable_review, review_path, save_review, sheets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX = PROJECT_ROOT / "front_end" / "tables.html"


class ReviewHandler(BaseHTTPRequestHandler):
    runs_dir: Path
    save_lock = Lock()

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: object, status: int = 200) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _manuals(self) -> list[str]:
        return sorted(
            path.name for path in self.runs_dir.iterdir()
            if path.is_dir() and (path / "tables/index.json").is_file()
        )

    def _manual(self, query: dict[str, list[str]]) -> Path:
        name = query.get("manual", [""])[0]
        if name not in self._manuals():
            raise FileNotFoundError("Unknown manual")
        return self.runs_dir / name / "tables"

    def _review_status(self, run_dir: Path, record: dict) -> str:
        path = review_path(run_dir, run_dir / record["artifact_directory"])
        if not path.is_file():
            return "unreviewed"
        try:
            return read_review(path, record)["status"]
        except ValueError:
            return "stale"

    def _table(self, query: dict[str, list[str]]) -> tuple[Path, dict, Path, dict]:
        run_dir = self._manual(query)
        records = ready_tables(run_dir.parent)
        candidate_id = query.get("id", [""])[0]
        matches = [item for item in records if item["candidate_id"] == candidate_id]
        if len(matches) != 1:
            raise FileNotFoundError("Unknown or duplicated Table")
        record = matches[0]
        artifact = (run_dir / record["artifact_directory"]).resolve()
        artifact.relative_to(run_dir.resolve())
        candidate = json.loads((artifact / "candidate.json").read_text(encoding="utf-8"))
        if candidate["candidate_id"] != candidate_id:
            raise ValueError("Table artifact does not match the index")
        return run_dir, record, artifact, candidate

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in {"/", "/tables"}:
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif parsed.path == "/api/manuals":
                self._json(self._manuals())
            elif parsed.path == "/api/tables":
                run_dir = self._manual(query)
                records = ready_tables(run_dir.parent)
                self._json([{
                    "id": item["candidate_id"],
                    "caption": item.get("caption") or item.get("node_id_raw") or "未命名表格",
                    "pages": item["source_physical_pages"],
                    "source_status": item.get("data_status"),
                    "review_status": self._review_status(run_dir, item),
                } for item in records])
            elif parsed.path == "/api/table":
                run_dir, record, artifact, candidate = self._table(query)
                self._json({
                    "id": candidate["candidate_id"],
                    "caption": candidate.get("caption") or "未命名表格",
                    "footnotes": candidate.get("footnotes") or "",
                    "pages": candidate["source_physical_pages"],
                    "source_status": record.get("data_status"),
                    "components": [{
                        "id": part["component_id"],
                        "title": part.get("title_raw") or "",
                        "pages": part.get("source_physical_pages") or candidate["source_physical_pages"],
                        "rows": part["grid"]["rows"],
                        "cells": part["grid"].get("cells", []),
                    } for part in sheets(candidate)],
                    "review": editable_review(review_path(run_dir, artifact), candidate),
                })
            elif parsed.path == "/api/page":
                run_dir, _, artifact, candidate = self._table(query)
                page = int(query.get("page", ["0"])[0])
                if page not in candidate["source_physical_pages"]:
                    raise FileNotFoundError("Page is not a source of this Table")
                path = source_page(run_dir, page)
                self._send(200, path.read_bytes(), "application/pdf")
            elif parsed.path == "/api/page-image":
                run_dir, _, artifact, candidate = self._table(query)
                page = int(query.get("page", ["0"])[0])
                if page not in candidate["source_physical_pages"]:
                    raise FileNotFoundError("Page is not a source of this Table")
                path = source_page(run_dir, page)
                with pymupdf.open(path) as document:
                    image = document[0].get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), alpha=False)
                    self._send(200, image.tobytes("png"), "image/png")
            else:
                raise FileNotFoundError("Unknown route")
        except (FileNotFoundError, ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, 404 if isinstance(exc, FileNotFoundError) else 400)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path != "/api/review":
            self._json({"error": "Unknown route"}, 404)
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Expected JSON")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 5_000_000:
                raise ValueError("Invalid request size")
            incoming = json.loads(self.rfile.read(length))
            if not isinstance(incoming, dict):
                raise ValueError("Expected a review object")
            with self.save_lock, lock(child(self.runs_dir, parse_qs(parsed.query).get("manual", [""])[0]), "publication"):
                run_dir, _, artifact, candidate = self._table(parse_qs(parsed.query))
                saved = save_review(review_path(run_dir, artifact), candidate, incoming)
            self._json(saved)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, 404 if isinstance(exc, FileNotFoundError) else 400)
