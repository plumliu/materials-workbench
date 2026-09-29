"""Loopback-only UI with fixed routes, background local commands and TAR persistence."""

import json
import mimetypes
import subprocess
import sys
import tarfile
import tempfile
from contextlib import ExitStack
from threading import Thread
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pymupdf

from chart_annotator.wpd import read as read_tar
from pdf_tree_workflow.review_server import ReviewHandler

from .storage import ROOT, lock, write_json
from .workflow import (
    AssemblyBlocked,
    _figure as process_figure,
    reset_figure_project,
    figure_info,
    manual,
    operation_locks,
    paths,
    require_assembly,
    summary,
    tar_version,
)


class Handler(ReviewHandler):
    root = ROOT
    children = {}

    def _origin(self):
        host = self.headers.get("Host", "")
        if host not in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }:
            raise ValueError("Invalid local host")
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + host:
            raise ValueError("Cross-origin write rejected")

    def _figure(self, query):
        name = query.get("manual", [""])[0]
        data = manual(self.root, name)
        item = next(
            (i for i in data["figures"] if i["id"] == query.get("id", [""])[0]), None
        )
        if item is None:
            raise ValueError("未知 Figure")
        return name, item

    def _static(self, base, relative):
        target = (base / relative).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            raise FileNotFoundError("Unknown file")
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix in {".js", ".mjs"}:
            mime = "text/javascript"
        self._send(200, target.read_bytes(), mime)

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            self._origin()
            if parsed.path in {"/", "/figures"}:
                self._static(
                    ROOT / "front_end",
                    "index.html" if parsed.path == "/" else "figures.html",
                )
            elif parsed.path.startswith("/front_end/"):
                self._static(
                    ROOT / "front_end", parsed.path.removeprefix("/front_end/")
                )
            elif parsed.path.startswith("/wpd/"):
                self._static(
                    ROOT / "vendor/wpd",
                    parsed.path.removeprefix("/wpd/") or "index.html",
                )
            elif parsed.path == "/api/overview":
                self._json(
                    [
                        summary(self.root, path.stem)
                        for path in sorted((self.root / "pdfs").glob("*.pdf"))
                    ]
                )
            elif parsed.path == "/api/figures":
                name = query.get("manual", [""])[0]
                self._json(
                    [
                        figure_info(self.root, name, item)
                        for item in manual(self.root, name)["figures"]
                    ]
                )
            elif parsed.path == "/api/figure-pdf":
                name, item = self._figure(query)
                _, assets, _ = paths(self.root, name)
                self._send(200, (assets / item["id"] / (item["id"] + ".pdf")).read_bytes(), "application/pdf")
            elif parsed.path == "/api/figure-tar":
                name, item = self._figure(query)
                _, assets, _ = paths(self.root, name)
                self._send(
                    200,
                    (assets / item["id"] / (item["id"] + ".tar")).read_bytes(),
                    "application/x-tar",
                )
            elif parsed.path in {"/api/source", "/api/source-image"}:
                name = query.get("manual", [""])[0]
                data = manual(self.root, name)
                page = int(query.get("page", ["0"])[0])
                if not 1 <= page <= data["page_count"]:
                    raise ValueError("未知来源页")
                source = (
                    self.root
                    / "figure_assets"
                    / name
                    / "_page_review"
                    / f"page_{page:04d}.pdf"
                )
                if parsed.path.endswith("-image"):
                    with pymupdf.open(source) as pdf:
                        self._send(
                            200,
                            pdf[0]
                            .get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), alpha=False)
                            .tobytes("png"),
                            "image/png",
                        )
                else:
                    self._send(200, source.read_bytes(), "application/pdf")
            else:
                super().do_GET()
        except (ValueError, OSError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)

    def do_POST(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            self._origin()
            if parsed.path == "/api/review":
                return super().do_POST()
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 100_000_000:
                raise ValueError("无效的提交大小")
            if parsed.path == "/api/figure-save":
                if self.headers.get("Content-Type") != "application/x-tar":
                    raise ValueError("Expected TAR")
                name, item = self._figure(query)
                run, assets, _ = paths(self.root, name)
                archive = assets / item["id"] / (item["id"] + ".tar")
                version = json.loads(query.get("version", ["null"])[0])
                status = query.get("status", ["draft"])[0]
                if status not in {"draft", "verified"}:
                    raise ValueError("无效标注状态")
                payload = self.rfile.read(length)
                with self.save_lock, lock(run, "figure-" + item["id"], "publication"):
                    if (tar_version(archive) if archive.exists() else None) != version:
                        raise ValueError("该图已在其他窗口保存，请重新打开后再修改")
                    with tempfile.NamedTemporaryFile(
                        dir=archive.parent, suffix=".tar", delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                        stream.write(payload)
                    try:
                        project, pixels = read_tar(temporary)
                        original_pixels = read_tar(archive)[1] if archive.exists() else (
                            assets / item["id"] / (item["id"] + ".pdf")
                        ).read_bytes()
                        if pixels != original_pixels:
                            raise ValueError(
                                "原始图像已被替换，请保留当前 TAR 的原图继续标注"
                            )
                        axes = project.get("axesColl")
                        datasets = project.get("datasetColl")
                        if (
                            project.get("version", [None])[0] != 4
                            or not isinstance(axes, list)
                            or not isinstance(datasets, list)
                        ):
                            raise ValueError("无效的 WPD 项目")
                        names = [a["name"] for a in axes]
                        if len(names) != len(set(names)) or any(
                            d.get("axesName") not in names for d in datasets
                        ):
                            raise ValueError("Dataset 的坐标轴归属无效")
                        for dataset in datasets:
                            if not isinstance(dataset.get("data"), list):
                                raise ValueError("无效的 Dataset 数据")
                        temporary.replace(archive)
                        write_json(
                            run / "figures" / item["id"] / "review.json",
                            {"status": status, "version": tar_version(archive)},
                        )
                    finally:
                        temporary.unlink(missing_ok=True)
                    self._json(figure_info(self.root, name, item))
            elif parsed.path == "/api/figure-recognize":
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Expected JSON")
                incoming = json.loads(self.rfile.read(length))
                if incoming.get("confirm_reset") is not True:
                    raise ValueError("请先确认清空当前图的全部标注")
                name, item = self._figure(query)
                run, assets, _ = paths(self.root, name)
                archive = assets / item["id"] / (item["id"] + ".tar")
                # The short manual gate closes the race with intake / whole-book jobs.
                # The Figure lock is handed to the worker and lasts for the entire run.
                with self.save_lock, lock(run, "figures"), ExitStack() as held:
                    held.enter_context(lock(run, "figure-" + item["id"]))
                    if (tar_version(archive) if archive.exists() else None) != incoming.get("version"):
                        raise ValueError("该图已更新，请重新打开后再发起识别")
                    reset_figure_project(self.root, name, item)
                    write_json(run / "figures" / item["id"] / "status.json", {"status": "running", "id": item["id"]})
                    info = figure_info(self.root, name, item)
                    ownership = held.pop_all()
                    root = self.root

                    def recognize():
                        with ownership:
                            try:
                                process_figure(root, name, item, retry=True, replace_existing=True)
                            except Exception as exc:
                                write_json(run / "figures" / item["id"] / "status.json", {
                                    "status": "failed", "error": type(exc).__name__,
                                    "message": "智能识别未完成，可以重新识别或继续手工标注。",
                                })

                    try:
                        Thread(target=recognize, daemon=True, name="recognize-" + item["id"]).start()
                    except Exception:
                        ownership.close()
                        raise
                self._json(info, 202)
            elif parsed.path == "/api/action":
                if (
                    self.headers.get("Content-Type", "").split(";")[0]
                    != "application/json"
                ):
                    raise ValueError("Expected JSON")
                incoming = json.loads(self.rfile.read(length))
                name, action = incoming["manual"], incoming["action"]
                if action not in {"intake", "assemble"}:
                    raise ValueError("模型处理请交给 Codex 启动")
                run, _, pdf = paths(self.root, name)
                if not pdf.is_file():
                    raise ValueError("请将手册 PDF 放入 pdfs 目录")
                key = (name, action)
                with self.save_lock:
                    current = self.children.get(key)
                    if current and current.poll() is None:
                        raise ValueError("操作已启动")
                    with lock(run, *operation_locks(action, run)):
                        if action == "assemble":
                            require_assembly(
                                summary(self.root, name),
                                incoming.get("allow_incomplete") is True,
                            )
                    command = [
                        sys.executable,
                        "-m",
                        "materials_workbench.cli",
                        "--root",
                        str(self.root),
                        action,
                        name,
                    ]
                    if (
                        action == "assemble"
                        and incoming.get("allow_incomplete") is True
                    ):
                        command.append("--allow-incomplete")
                    # Errors are persisted by execute(); don't retain another full log copy.
                    self.children[key] = subprocess.Popen(
                        command,
                        cwd=self.root,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                self._json({"status": "started"}, 202)
            else:
                raise ValueError("Unknown route")
        except AssemblyBlocked as exc:
            self._json({"status": "blocked", "error": str(exc)}, 409)
        except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as exc:
            self._json({"error": str(exc)}, 409 if isinstance(exc, ValueError) else 400)


def serve(root=ROOT, port=8766):
    Handler.root = Path(root).resolve()
    Handler.runs_dir = Handler.root / "runs"
    for folder in ("pdfs", "figure_assets", "runs", "library"):
        (Handler.root / folder).mkdir(exist_ok=True)
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(
            f"Materials Workbench: http://127.0.0.1:{server.server_port}/", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
