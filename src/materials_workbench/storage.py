"""Fixed paths, atomic publication and process locks shared by CLI and web."""

import json
import os
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return Path(path).is_file() and json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def child(root, name):
    if not name or name in {".", ".."} or any(c in name for c in "/\\:\0"):
        raise ValueError("无效的名称")
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path.parent != root:
        raise ValueError("路径超出工作目录")
    return path


@contextmanager
def lock(run, *names):
    """OS-owned locks disappear on process exit; lock files carry no state."""
    directory = Path(run) / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name in sorted(names):
            stream = stack.enter_context((directory / name).open("a+b"))
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

                    def unlock(s=stream):
                        s.seek(0)
                        msvcrt.locking(s.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

                    def unlock(s=stream):
                        fcntl.flock(s, fcntl.LOCK_UN)

                stack.callback(unlock)
            except OSError as exc:
                raise ValueError("该手册的相关操作正在进行，请稍后重试") from exc
        yield


def candidates(run):
    records = read_json(Path(run) / "index.json") or []
    values = [
        read_json(child(run, item["artifact_directory"]) / "candidate.json")
        for item in records
    ]
    for value in values:
        if (
            not value
            or value.get("schema_version") != 2
            or type(value.get("revision")) is not int
        ):
            raise ValueError("Only workbench candidate schema 2 is supported")
    return values


def write_index(run, records):
    fields = (
        "candidate_id",
        "artifact_directory",
        "caption",
        "source_physical_pages",
        "data_status",
    )
    write_json(
        Path(run) / "index.json",
        [{key: item.get(key) for key in fields} for item in records],
    )


def publish_candidate(directory, candidate):
    path = Path(directory) / "candidate.json"
    current = read_json(path) or {}
    candidate["schema_version"] = 2
    candidate["revision"] = current.get("revision", 0) + 1
    write_json(path, candidate)


def source_page(run, page):
    manifest = read_json(Path(run) / "manifest.json")
    return Path(manifest["page_review_dir"]) / f"page_{page:04d}.pdf"
