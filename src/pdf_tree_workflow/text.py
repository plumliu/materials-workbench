from __future__ import annotations

import hashlib
import re
from pathlib import Path


INVALID_WINDOWS_CHARS = re.compile(r'[\\/:*?"<>|]')
WHITESPACE = re.compile(r"\s+")


def join_wrapped_lines(lines: list[str]) -> str:
    """Join publication line wraps while preserving printed wording."""
    result = ""
    for raw in lines:
        line = WHITESPACE.sub(" ", raw).strip()
        if not line:
            continue
        if not result:
            result = line
            continue
        if result.endswith("-"):
            token = result.rsplit(" ", 1)[-1][:-1]
            keep_hyphen = (
                "-" in token
                or any(character.isdigit() for character in token)
                or (line and (line[0].isdigit() or line[0].isupper()))
            )
            result = result + line if keep_hyphen else result[:-1] + line
        else:
            result += " " + line
    return result.strip()


def normalize_paragraph(text: str) -> str:
    return join_wrapped_lines(text.splitlines())


def safe_segment(raw: str, *, max_length: int = 100) -> str:
    value = INVALID_WINDOWS_CHARS.sub(" - ", raw)
    value = WHITESPACE.sub(" ", value).strip(" .")
    if not value:
        value = "untitled"
    if len(value) <= max_length:
        return value
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    keep = max_length - len(digest) - 2
    return f"{value[:keep].rstrip()}__{digest}"


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def relative_posix(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()
