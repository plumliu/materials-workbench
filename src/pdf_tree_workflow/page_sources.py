from __future__ import annotations

from pathlib import Path
import re

import pymupdf


PAGE_FILE_RE = re.compile(r"(?i)^page_(?P<number>\d{4,})\.pdf$")


def page_review_directory(figure_assets: Path) -> Path:
    path = figure_assets.expanduser().resolve() / "_page_review"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required single-page source directory is missing: {path}"
        )
    return path


def indexed_page_sources(
    page_review_dir: Path, *, expected_page_count: int | None = None
) -> dict[int, Path]:
    page_review_dir = page_review_dir.expanduser().resolve()
    if not page_review_dir.is_dir():
        raise FileNotFoundError(page_review_dir)
    indexed: dict[int, Path] = {}
    for path in page_review_dir.iterdir():
        if not path.is_file():
            continue
        match = PAGE_FILE_RE.match(path.name)
        if not match:
            continue
        page_number = int(match.group("number"))
        if page_number in indexed:
            raise ValueError(f"Duplicate page number {page_number} in {page_review_dir}")
        indexed[page_number] = path
    if expected_page_count is not None:
        expected = set(range(1, expected_page_count + 1))
        actual = set(indexed)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"_page_review does not match the complete PDF: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
    return dict(sorted(indexed.items()))


def validate_single_page_sources(sources: dict[int, Path]) -> None:
    for physical_page, path in sources.items():
        document = pymupdf.open(path)
        try:
            if document.page_count != 1:
                raise ValueError(
                    f"Expected one page for physical page {physical_page}: {path}"
                )
        finally:
            document.close()
