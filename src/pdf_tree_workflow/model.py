from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .identifiers import parse_identifier_prefix


@dataclass(slots=True)
class TextRegion:
    physical_page: int
    rect: tuple[float, float, float, float]
    text: str


@dataclass(slots=True)
class DirectLink:
    relation: str
    target_type: str
    target_id: str
    target_physical_page: int | None
    occurrences: int = 1
    external_file: str | None = None

    @property
    def key(self) -> tuple[str, str, str, int | None, str | None]:
        return (
            self.relation,
            self.target_type,
            self.target_id,
            self.target_physical_page,
            self.external_file,
        )


@dataclass(slots=True)
class Node:
    sequence: int
    node_id_raw: str | None
    title_raw: str
    node_type: str
    synthetic_id: str | None = None
    identifier_status: str = "present"
    catalog_pages: set[int] = field(default_factory=set)
    text_regions: list[TextRegion] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)
    target_page: int | None = None
    source_page_numbers: set[int] = field(default_factory=set)
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    references: Counter[int] = field(default_factory=Counter)
    direct_links: list[DirectLink] = field(default_factory=list)
    output_path: Path | None = None
    figure_tars: list[Path] = field(default_factory=list)
    table_artifact: Path | None = None
    table_data_status: str | None = None

    @property
    def depth(self) -> int:
        parsed = parse_identifier_prefix(self.node_id_raw or "")
        return parsed.depth if parsed else len((self.node_id_raw or self.synthetic_id or "").split("."))

    @property
    def display_id(self) -> str:
        if self.node_id_raw:
            return self.node_id_raw
        if self.synthetic_id:
            return self.synthetic_id.replace("unnumbered-", "Unnumbered ")
        return "Unresolved"

    @property
    def identifier_core(self) -> str:
        parsed = parse_identifier_prefix(self.node_id_raw or "")
        return parsed.core if parsed else (self.node_id_raw or self.synthetic_id or "")

    @property
    def identifier_suffixes(self) -> tuple[str, ...]:
        parsed = parse_identifier_prefix(self.node_id_raw or "")
        return parsed.suffixes if parsed else ()

    @property
    def identifier_match_key(self) -> str:
        parsed = parse_identifier_prefix(self.node_id_raw or "")
        return parsed.match_key if parsed else (self.node_id_raw or self.synthetic_id or "").casefold()

    @property
    def source_pages(self) -> list[int]:
        if self.source_page_numbers:
            return sorted(self.source_page_numbers)
        if self.node_type in {"table", "figure"}:
            if self.target_page is not None:
                return [self.target_page]
            return sorted(self.catalog_pages)
        pages = set(self.catalog_pages)
        pages.update(region.physical_page for region in self.text_regions)
        return sorted(pages)

    @property
    def body(self) -> str:
        return "\n\n".join(part.strip() for part in self.body_parts if part.strip()).strip()

    @property
    def stable_id(self) -> str:
        parts: list[str] = []
        current: Node | None = self
        while current is not None:
            parts.append(f"{current.display_id}:{current.node_type}:{current.sequence}")
            current = current.parent
        return "/".join(reversed(parts))


@dataclass(slots=True)
class ParsedManual:
    pdf_path: Path
    page_count: int
    front_pages: list[int]
    reference_pages: list[int]
    nodes: list[Node]
    references: dict[int, list[str]]
    structure_validation: dict[str, object]
