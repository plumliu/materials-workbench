from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from pathlib import Path
import unicodedata

import pymupdf

from .model import DirectLink, Node, ParsedManual, TextRegion
from .identifiers import IDENTIFIER_TOKEN_PATTERN, parse_identifier_prefix
from .text import join_wrapped_lines, normalize_paragraph


HEADING_RE = re.compile(
    rf"^\s*(?P<id>{IDENTIFIER_TOKEN_PATTERN})(?:\.(?=\s|\[|$))?\s*(?P<rest>.*)$"
)
LEAF_RE = re.compile(
    r"^(?P<prefix>.*?)\s*\[(?P<kind>Table|Figure)\]\s*(?P<title>.*)$",
    re.I,
)
REFERENCE_BLOCK_RE = re.compile(r"^\s*(?P<number>\d{1,3})\.\s*(?P<body>.*)$", re.S)
REF_WORD_RE = re.compile(r"\bRef(?:erence)?\.?\s*\(?\s*([0-9][0-9,\s]*)\)?", re.I)
PAREN_REF_RE = re.compile(r"\(([0-9]{1,2}(?:\s*,\s*[0-9]{1,2})*)\)")


@dataclass(slots=True)
class PdfLine:
    text: str
    rect: tuple[float, float, float, float]
    max_size: float
    bold: bool


def _rect_union(lines: list[PdfLine]) -> tuple[float, float, float, float]:
    return (
        min(line.rect[0] for line in lines),
        min(line.rect[1] for line in lines),
        max(line.rect[2] for line in lines),
        max(line.rect[3] for line in lines),
    )


def _block_lines(block: dict) -> list[PdfLine]:
    result: list[PdfLine] = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        text = "".join(span.get("text", "") for span in spans).strip()
        if not text:
            continue
        result.append(
            PdfLine(
                text=text,
                rect=tuple(float(value) for value in line["bbox"]),
                max_size=max(float(span.get("size", 0)) for span in spans),
                bold=any("bold" in span.get("font", "").lower() for span in spans),
            )
        )
    return result


def _heading_match(text: str) -> re.Match[str] | None:
    match = HEADING_RE.match(text)
    if not match:
        return None
    raw_id = match.group("id").strip()
    parsed = parse_identifier_prefix(raw_id)
    if parsed is None:
        return None
    if parsed.depth == 1 and parsed.suffixes:
        return None
    first = int(parsed.core.split(".", 1)[0])
    if first > 9:
        return None
    rest = match.group("rest").strip()
    # Empty/wrapped titles and typed catalog leaves are valid. Sentence
    # punctuation or a lowercase continuation instead indicates prose such as
    # a cross-reference or measurement, not a semantic heading.
    if re.match(r"^[,;)\]%]", rest):
        return None
    if rest and not rest.startswith("[") and re.match(r"^[a-z]", rest):
        return None
    if re.match(r"(?i)^and\s+[1-9]\d*(?:\.\d+)+\b", rest):
        return None
    if "." not in parsed.core and rest and not re.search(r"[A-Za-z]", rest):
        return None
    return match


def _heading_top_number(text: str) -> int | None:
    match = _heading_match(text)
    if match is None:
        return None
    parsed = parse_identifier_prefix(match.group("id"))
    return int(parsed.core.split(".", 1)[0]) if parsed else None


def _split_heading_segment(lines: list[PdfLine]) -> tuple[str, str, str, list[str]]:
    match = _heading_match(lines[0].text)
    if match is None:
        raise ValueError("segment does not start with a heading")

    raw_id = match.group("id").strip()
    first_rest = match.group("rest").strip()
    combined = join_wrapped_lines([first_rest, *[line.text for line in lines[1:]]])
    leaf = LEAF_RE.match(combined)
    if leaf:
        title = join_wrapped_lines(
            [leaf.group("prefix").strip(), leaf.group("title").strip()]
        )
        return (
            raw_id,
            leaf.group("kind").lower(),
            title,
            [],
        )

    node_type = "section"
    title_lines: list[str] = []
    body_start = 1
    if first_rest:
        title_lines.append(first_rest)
        body_start = 1
    elif len(lines) > 1:
        title_lines.append(lines[1].text)
        body_start = 2

    heading_style_size = lines[0].max_size
    while body_start < len(lines):
        previous = title_lines[-1] if title_lines else ""
        candidate = lines[body_start]
        same_display_style = (
            heading_style_size >= 10
            and candidate.max_size >= heading_style_size - 0.2
            and candidate.bold == lines[0].bold
        )
        if previous.rstrip().endswith("-") or same_display_style:
            title_lines.append(candidate.text)
            body_start += 1
            continue
        break

    title = join_wrapped_lines(title_lines)
    body = [line.text for line in lines[body_start:]]
    return raw_id, node_type, title, body


def _is_page_furniture(lines: list[PdfLine]) -> bool:
    if not lines:
        return True
    rect = _rect_union(lines)
    text = " ".join(line.text for line in lines)
    if rect[1] < 70 or rect[3] > 721:
        return True
    if text.startswith("© 2011 by CINDAS"):
        return True
    return False


def _find_reference_pages(document: pymupdf.Document) -> tuple[list[int], list[int]]:
    reference_start: int | None = None
    for index, page in enumerate(document):
        if re.search(r"(?im)^\s*REFERENCES\.?\s*$", page.get_text()):
            reference_start = index + 1
            break
    if reference_start is None:
        raise ValueError("Could not locate the REFERENCES section")

    reference_pages: list[int] = []
    for physical_page in range(reference_start, document.page_count + 1):
        page = document[physical_page - 1]
        page_text = page.get_text()
        if (
            physical_page > reference_start
            and re.search(r"(?m)^\s*(?:Figure|Table)\s+\d", page_text)
        ):
            break
        blocks = page.get_text("blocks", sort=False)
        numbered = sum(
            1
            for block in blocks
            if REFERENCE_BLOCK_RE.match(normalize_paragraph(block[4]))
        )
        if physical_page == reference_start or numbered:
            reference_pages.append(physical_page)
            continue
        break

    front_pages = list(range(1, reference_start))
    return front_pages, reference_pages


def _parse_nodes(document: pymupdf.Document, front_pages: list[int]) -> list[Node]:
    nodes: list[Node] = []
    current: Node | None = None
    started = False

    for physical_page in front_pages:
        page = document[physical_page - 1]
        for block in page.get_text("dict", sort=False).get("blocks", []):
            lines = _block_lines(block)
            if _is_page_furniture(lines):
                continue

            heading_indices = [
                index
                for index, line in enumerate(lines)
                if _heading_match(line.text)
                and (
                    index == 0
                    or line.bold
                    or line.max_size >= 9.5
                    or bool(re.search(r"\[(?:Table|Figure)\]", line.text, re.I))
                    or (
                        bool(_heading_match(line.text).group("rest").strip())
                        and len(line.text) <= 50
                    )
                )
            ]
            if started and current is not None and heading_indices:
                current_top = int(current.identifier_core.split(".", 1)[0])
                heading_indices = [
                    index
                    for index in heading_indices
                    if _heading_top_number(lines[index].text)
                    in {current_top, current_top + 1}
                ]
            if not started and heading_indices:
                root_starts = [
                    index
                    for index in heading_indices
                    if lines[index].max_size >= 11
                    and lines[index].bold
                    and parse_identifier_prefix(
                        _heading_match(lines[index].text).group("id")
                    ).core == "1"
                ]
                if not root_starts:
                    continue
                first_root = root_starts[0]
                heading_indices = [index for index in heading_indices if index >= first_root]
            if not heading_indices:
                if started and current is not None and current.node_type == "section":
                    text = join_wrapped_lines([line.text for line in lines])
                    if text:
                        current.body_parts.append(text)
                        current.text_regions.append(
                            TextRegion(physical_page, _rect_union(lines), text)
                        )
                continue

            started = True
            if heading_indices[0] > 0 and current is not None and current.node_type == "section":
                prefix = lines[: heading_indices[0]]
                text = join_wrapped_lines([line.text for line in prefix])
                if text:
                    current.body_parts.append(text)
                    current.text_regions.append(
                        TextRegion(physical_page, _rect_union(prefix), text)
                    )

            for position, start in enumerate(heading_indices):
                end = (
                    heading_indices[position + 1]
                    if position + 1 < len(heading_indices)
                    else len(lines)
                )
                segment = lines[start:end]
                raw_id, node_type, title, body_lines = _split_heading_segment(segment)
                current = Node(
                    sequence=len(nodes) + 1,
                    node_id_raw=raw_id,
                    title_raw=title,
                    node_type=node_type,
                    catalog_pages={physical_page},
                )
                nodes.append(current)
                region_text = join_wrapped_lines([line.text for line in segment])
                region = TextRegion(physical_page, _rect_union(segment), region_text)
                current.text_regions.append(region)
                if node_type == "section":
                    body_text = join_wrapped_lines(body_lines)
                    if body_text:
                        current.body_parts.append(body_text)

    if not nodes:
        raise ValueError("No numbered nodes were parsed from the front matter")
    return nodes


def _assign_hierarchy(nodes: list[Node]) -> None:
    stack: list[Node] = []
    for node in nodes:
        depth = node.depth
        while len(stack) >= depth:
            stack.pop()
        if depth > 1:
            if not stack:
                raise ValueError(f"Missing parent for node {node.node_id_raw}")
            node.parent = stack[-1]
            node.parent.children.append(node)
        stack.append(node)


def _assign_leaf_targets(document: pymupdf.Document, nodes: list[Node]) -> None:
    leaves = [node for node in nodes if node.node_type in {"table", "figure"}]
    for node in leaves:
        for physical_page in sorted(node.catalog_pages):
            page = document[physical_page - 1]
            for link in page.get_links():
                target = link.get("page", -1)
                if link.get("kind") != pymupdf.LINK_GOTO or target < 0:
                    continue
                link_rect = pymupdf.Rect(link["from"])
                if any(
                    link_rect.intersects(pymupdf.Rect(region.rect))
                    for region in node.text_regions
                    if region.physical_page == physical_page
                ):
                    node.target_page = target + 1
                    break

                if node.target_page is not None:
                    node.source_page_numbers.add(node.target_page)
                    break

        if node.target_page is not None:
            continue

        # Leaf regions are catalog-only and are not kept in text_regions. Match the
        # visible linked number directly as a fallback.
        for physical_page in sorted(node.catalog_pages):
            page = document[physical_page - 1]
            for link in page.get_links():
                if link.get("kind") != pymupdf.LINK_GOTO or link.get("page", -1) < 0:
                    continue
                visible = page.get_textbox(pymupdf.Rect(link["from"])).strip()
                if node.node_id_raw in visible:
                    node.target_page = int(link["page"]) + 1
                    break
            if node.target_page is not None:
                node.source_page_numbers.add(node.target_page)
                break

    # Some link rectangles are malformed. Search the linked catalog number by page
    # coordinates before falling back to a text-only scan of data pages.
    for node in leaves:
        if node.target_page is not None:
            continue
        label = f"{node.node_type.title()} {node.node_id_raw}"
        for physical_page in range(1, document.page_count + 1):
            if physical_page in node.catalog_pages:
                continue
            text = normalize_paragraph(document[physical_page - 1].get_text())
            if label.lower() in text.lower():
                node.target_page = physical_page
                node.source_page_numbers.add(physical_page)
                break


def _reference_block_order(
    blocks: list[tuple[float, float, float, float, str]], page_width: float
) -> list[tuple[float, float, float, float, str]]:
    """Return column-major reading order for one References page."""

    midpoint = page_width / 2
    return sorted(
        blocks,
        key=lambda block: (0 if block[0] < midpoint else 1, block[1], block[0]),
    )


def _parse_reference_text_blocks(
    pages: list[list[str]],
) -> dict[int, list[str]]:
    """Group already ordered blocks, preserving continuations and duplicates."""

    references: dict[int, list[str]] = {}
    current_entry: tuple[int, int] | None = None
    for blocks in pages:
        for text in blocks:
            match = REFERENCE_BLOCK_RE.match(text)
            if match:
                number = int(match.group("number"))
                entries = references.setdefault(number, [])
                entries.append(match.group("body").strip())
                current_entry = (number, len(entries) - 1)
                continue
            if current_entry is not None and text:
                number, index = current_entry
                references[number][index] = (
                    references[number][index].rstrip() + " " + text
                ).strip()
    return dict(sorted(references.items()))


def _parse_references(
    document: pymupdf.Document, reference_pages: list[int]
) -> dict[int, list[str]]:
    ordered_pages: list[list[str]] = []
    for physical_page in reference_pages:
        page = document[physical_page - 1]
        eligible: list[tuple[float, float, float, float, str]] = []
        for block in page.get_text("blocks", sort=False):
            rect = tuple(float(value) for value in block[:4])
            if rect[1] < 70 or rect[3] > 721:
                continue
            text = normalize_paragraph(block[4])
            if re.fullmatch(r"REFERENCES\.?", text, re.I):
                continue
            if text:
                eligible.append((*rect, text))
        ordered_pages.append(
            [
                block[4]
                for block in _reference_block_order(eligible, float(page.rect.width))
            ]
        )
    return _parse_reference_text_blocks(ordered_pages)


def _outline_title_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    value = value.replace("�", "").replace("’", "'")
    return "".join(character for character in value if character.isalnum())


def _outline_match(
    *, title: str, level: int, physical_page: int, sections: list[Node]
) -> tuple[Node | None, str]:
    bookmark_key = _outline_title_key(title)
    base_key = _outline_title_key(re.split(r"[:;]", title, maxsplit=1)[0])
    candidates = [
        node
        for node in sections
        if node.depth == level and physical_page in node.source_pages
    ]
    exact = [
        node for node in candidates if _outline_title_key(node.title_raw) == bookmark_key
    ]
    if len(exact) == 1:
        return exact[0], "exact_title_depth_page"
    prefixes = [
        node
        for node in candidates
        if len(base_key) >= 6
        and (
            _outline_title_key(node.title_raw).startswith(base_key)
            or base_key.startswith(_outline_title_key(node.title_raw))
        )
    ]
    if len(prefixes) == 1:
        return prefixes[0], "title_prefix_depth_page"
    return None, "ambiguous" if exact or prefixes else "no_match"


def _is_ancestor(candidate: Node, node: Node) -> bool:
    current = node.parent
    while current is not None:
        if current is candidate:
            return True
        current = current.parent
    return False


def _validate_outline(document: pymupdf.Document, nodes: list[Node]) -> dict[str, object]:
    """Cross-check section hierarchy without treating sparse bookmarks as authority."""

    sections = [node for node in nodes if node.node_type == "section"]
    ignored_titles = {"text", "references"}
    matched: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    outline_stack: dict[int, Node | None] = {}
    last_sequence = 0
    toc = document.get_toc(simple=False)

    for level, title, physical_page, *_details in toc:
        if title.strip().casefold().rstrip(".") in ignored_titles:
            continue
        for depth in [value for value in outline_stack if value >= level]:
            outline_stack.pop(depth, None)
        node, method = _outline_match(
            title=title,
            level=int(level),
            physical_page=int(physical_page),
            sections=sections,
        )
        if node is None:
            unmatched.append(
                {
                    "outline_level": int(level),
                    "title": title.strip(),
                    "physical_page": int(physical_page),
                    "reason": method,
                }
            )
            outline_stack[int(level)] = None
            continue

        nearest_outline_parent = next(
            (
                outline_stack[depth]
                for depth in range(int(level) - 1, 0, -1)
                if outline_stack.get(depth) is not None
            ),
            None,
        )
        hierarchy_ok = (
            nearest_outline_parent is None
            or _is_ancestor(nearest_outline_parent, node)
        )
        page_ok = int(physical_page) in node.source_pages
        order_ok = node.sequence >= last_sequence
        record = {
            "outline_level": int(level),
            "title": title.strip(),
            "physical_page": int(physical_page),
            "node_sequence": node.sequence,
            "node_id_raw": node.node_id_raw,
            "node_title_raw": node.title_raw,
            "match_method": method,
            "page_ok": page_ok,
            "hierarchy_ok": hierarchy_ok,
            "order_ok": order_ok,
        }
        matched.append(record)
        if not page_ok:
            issues.append({"code": "outline_page_mismatch", **record})
        if not hierarchy_ok:
            issues.append({"code": "outline_hierarchy_mismatch", **record})
        if not order_ok:
            issues.append({"code": "outline_order_mismatch", **record})
        outline_stack[int(level)] = node
        last_sequence = max(last_sequence, node.sequence)

    return {
        "schema_version": 1,
        "authority": "parsed_numbered_sections",
        "outline_role": "cross_validation_only",
        "outline_entries": len(toc),
        "matched_entries": matched,
        "unmatched_entries": unmatched,
        "issues": issues,
        "summary": {
            "matched": len(matched),
            "unmatched": len(unmatched),
            "issues": len(issues),
            "status": "pass" if not issues else "review_required",
        },
    }


def _citation_numbers(text: str, max_reasonable: int = 99) -> list[int]:
    values: list[int] = []
    for pattern in (REF_WORD_RE, PAREN_REF_RE):
        for match in pattern.finditer(text):
            for token in re.findall(r"\d+", match.group(1)):
                value = int(token)
                if 0 < value <= max_reasonable:
                    values.append(value)
    return values


def _region_links(page: pymupdf.Page, region: TextRegion) -> list[dict]:
    bounds = pymupdf.Rect(region.rect)
    return [
        link
        for link in page.get_links()
        if pymupdf.Rect(link["from"]).intersects(bounds)
    ]


def _scan_node_relations(
    document: pymupdf.Document,
    nodes: list[Node],
    reference_pages: list[int],
) -> None:
    reference_page_set = set(reference_pages)
    leaf_by_page: dict[int, Node] = {
        node.target_page: node
        for node in nodes
        if node.node_type in {"table", "figure"} and node.target_page is not None
    }

    for node in nodes:
        link_counter: Counter[tuple[str, str, str, int | None, str | None]] = Counter()
        node.references.clear()
        link_examples: dict[
            tuple[str, str, str, int | None, str | None], DirectLink
        ] = {}

        if node.node_type == "section":
            scan_units = [
                (document[region.physical_page - 1], region, _region_links(document[region.physical_page - 1], region))
                for region in node.text_regions
            ]
            for part in node.body_parts:
                node.references.update(_citation_numbers(part))
        elif node.target_page is not None:
            scan_units = []
            for physical_page in node.source_pages:
                page = document[physical_page - 1]
                full_region = TextRegion(physical_page, (0.0, 0.0, float(page.rect.width), float(page.rect.height)), page.get_text())
                scan_units.append((page, full_region, page.get_links()))
                node.references.update(_citation_numbers(page.get_text()))
        else:
            scan_units = []

        for page, _region, links in scan_units:
            for link in links:
                kind = link.get("kind")
                target_index = int(link.get("page", -1))
                target_page = target_index + 1 if target_index >= 0 else None
                link_rect = pymupdf.Rect(link["from"])
                nearby = page.get_textbox(link_rect + (-28, -3, 70, 3))

                if kind == pymupdf.LINK_GOTO and target_page in reference_page_set:
                    node.references.update(_citation_numbers(nearby))
                    continue

                if kind == pymupdf.LINK_GOTO and target_page in leaf_by_page:
                    target = leaf_by_page[target_page]
                    if target is node:
                        continue
                    direct = DirectLink(
                        relation="related",
                        target_type=target.node_type.title(),
                        target_id=target.node_id_raw,
                        target_physical_page=target_page,
                    )
                elif kind == pymupdf.LINK_GOTOR:
                    external_file = str(link.get("file") or link.get("uri") or "").strip()
                    direct = DirectLink(
                        relation="external",
                        target_type="PDF",
                        target_id=external_file or nearby.strip() or "unresolved",
                        target_physical_page=None,
                        external_file=external_file or None,
                    )
                else:
                    continue

                link_counter[direct.key] += 1
                link_examples[direct.key] = direct

        node.direct_links = []
        for key, count in link_counter.items():
            direct = link_examples[key]
            direct.occurrences = count
            node.direct_links.append(direct)


def parse_manual(pdf_path: Path) -> ParsedManual:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    document = pymupdf.open(pdf_path)
    try:
        front_pages, reference_pages = _find_reference_pages(document)
        nodes = _parse_nodes(document, front_pages)
        _assign_hierarchy(nodes)
        _assign_leaf_targets(document, nodes)
        references = _parse_references(document, reference_pages)
        _scan_node_relations(document, nodes, reference_pages)
        structure_validation = _validate_outline(document, nodes)
        return ParsedManual(
            pdf_path=pdf_path,
            page_count=document.page_count,
            front_pages=front_pages,
            reference_pages=reference_pages,
            nodes=nodes,
            references=references,
            structure_validation=structure_validation,
        )
    finally:
        document.close()
