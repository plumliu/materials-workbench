from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import uuid

import pymupdf

from .human_review import apply_verified_review
from .model import DirectLink, Node, ParsedManual
from .identifiers import ParsedIdentifier, parse_identifier_prefix, parse_caption
from .page_sources import (
    indexed_page_sources,
    page_review_directory,
    validate_single_page_sources,
)
from .parser import parse_manual, _scan_node_relations
from .text import safe_segment, yaml_quote


def _node_display_name(node: Node) -> str:
    if node.node_type == "table" and node.node_id_raw is None:
        return f"[Table] {node.display_id} {node.title_raw}".strip()
    marker = ""
    if node.node_type == "table":
        marker = " [Table]"
    elif node.node_type == "figure":
        marker = " [Figure]"
    return f"{node.display_id}{marker} {node.title_raw}".strip()


def _node_prefix(node: Node) -> str:
    if node.node_type == "table":
        if node.node_id_raw is None:
            return f"[Table] {node.display_id}"
        return f"{node.display_id} [Table]"
    if node.node_type == "figure":
        return f"{node.display_id} [Figure]"
    return node.display_id


def _minimum_segment_length(node: Node) -> int:
    return len(_node_prefix(node)) + 10


def _shortened_node_segment(node: Node, limit: int) -> str:
    full = safe_segment(_node_display_name(node), max_length=1000)
    if len(full) <= limit:
        return full
    prefix = safe_segment(_node_prefix(node), max_length=1000)
    digest = hashlib.sha1(_node_display_name(node).encode("utf-8")).hexdigest()[:8]
    room = limit - len(prefix) - len(digest) - 3
    if room > 0:
        title = safe_segment(node.title_raw, max_length=1000)[:room].rstrip()
        return f"{prefix} {title}__{digest}"
    return f"{prefix}__{digest}"


def _assign_output_paths(nodes: list[Node], root: Path) -> None:
    descendant_budget: dict[int, int] = {}

    def reserve(node: Node) -> int:
        value = max(
            (
                1 + _minimum_segment_length(child) + reserve(child)
                for child in node.children
            ),
            default=0,
        )
        descendant_budget[node.sequence] = value
        return value

    for top in (node for node in nodes if node.parent is None):
        reserve(top)

    used_by_parent: dict[Path, Counter[str]] = {}
    for node in nodes:
        parent_path = node.parent.output_path if node.parent is not None else root
        if parent_path is None:
            raise RuntimeError("parent output path has not been assigned")
        # Leave enough room for node-local files such as
        # ``related_pages/Figure_<id>__physical_page_####.pdf``. The final root
        # can also be a few characters longer than the short staging root.
        available = (
            180
            - len(str(parent_path))
            - 1
            - descendant_budget[node.sequence]
        )
        segment_limit = max(_minimum_segment_length(node), min(100, available))
        base = _shortened_node_segment(node, segment_limit)
        counter = used_by_parent.setdefault(parent_path, Counter())
        counter[base.casefold()] += 1
        occurrence = counter[base.casefold()]
        segment = base if occurrence == 1 else f"{base}__{occurrence:02d}"
        node.output_path = parent_path / segment


def _locate_figure_tars(nodes: list[Node], figure_assets: Path) -> None:
    figure_assets = figure_assets.expanduser().resolve()
    if not figure_assets.is_dir():
        raise FileNotFoundError(figure_assets)
    figure_nodes = [node for node in nodes if node.node_type == "figure"]
    explicit_keys = {node.identifier_match_key for node in figure_nodes}
    asset_dirs: list[tuple[Path, ParsedIdentifier]] = []
    for candidate_dir in figure_assets.iterdir():
        if not candidate_dir.is_dir() or candidate_dir.name.startswith("_"):
            continue
        match = re.match(r"(?i)^Figure_(.+)$", candidate_dir.name)
        if not match:
            continue
        parsed = parse_identifier_prefix(match.group(1))
        if parsed:
            asset_dirs.append((candidate_dir, parsed))

    for node in figure_nodes:
        matched_dirs: list[Path] = []
        for candidate_dir, parsed in asset_dirs:
            exact = parsed.match_key == node.identifier_match_key
            implicit_panel = (
                not node.identifier_suffixes
                and parsed.core == node.identifier_core
                and bool(parsed.suffixes)
                and parsed.match_key not in explicit_keys
            )
            if exact or implicit_panel:
                matched_dirs.append(candidate_dir)
        for asset_dir in sorted(matched_dirs, key=lambda path: path.name.casefold()):
            exact = asset_dir / f"{asset_dir.name}.tar"
            if exact.is_file():
                node.figure_tars.append(exact)


def _validate_tar(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
            return bool(members)
    except (OSError, tarfile.TarError):
        return False


def _content_status(node: Node) -> str:
    if node.node_type in {"table", "figure"}:
        return "text_extracted" if node.target_page is not None else "missing_source_page"
    return "text_extracted" if node.body else "empty_in_source"


def _data_status(node: Node) -> str:
    if node.node_type == "section":
        return "not_applicable"
    if node.node_type == "table":
        return node.table_data_status or "pending"
    if not node.figure_tars:
        return "missing"
    return (
        "tar_present"
        if all(_validate_tar(path) for path in node.figure_tars)
        else "partial"
    )


def _standard_figure_tar_name(node: Node, source: Path) -> str:
    asset_unit = source.parent.name
    expected_prefix = f"Figure_{node.node_id_raw}"
    if asset_unit.startswith(expected_prefix):
        return f"{asset_unit}.tar"
    return f"{expected_prefix}.tar"


def _printed_page(document: pymupdf.Document, physical_page: int) -> int | str | None:
    text = document[physical_page - 1].get_text("text")
    match = re.search(r"(?m)^Page\s+([^\s]+)\s*$", text)
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else value


def _yaml_list(values: list[int | str]) -> str:
    rendered = [str(value) if isinstance(value, int) else yaml_quote(value) for value in values]
    return "[" + ", ".join(rendered) + "]"


def _write_content(node: Node, document: pymupdf.Document) -> None:
    if node.output_path is None:
        raise RuntimeError("node output path is missing")
    physical_pages = node.source_pages
    printed_pages = [
        printed
        for physical in physical_pages
        if (printed := _printed_page(document, physical)) is not None
    ]
    lines = [
        "---",
        f"node_type: {node.node_type}",
        f"stable_id: {yaml_quote(node.stable_id)}",
        (
            f"node_id_raw: {yaml_quote(node.node_id_raw)}"
            if node.node_id_raw is not None
            else "node_id_raw: null"
        ),
        (
            f"synthetic_id: {yaml_quote(node.synthetic_id)}"
            if node.synthetic_id is not None
            else "synthetic_id: null"
        ),
        f"identifier_status: {node.identifier_status}",
        f"title_raw: {yaml_quote(node.title_raw)}",
        f"source_physical_pages: {_yaml_list(physical_pages)}",
        f"source_printed_pages: {_yaml_list(printed_pages)}",
        f"content_status: {_content_status(node)}",
        f"data_status: {_data_status(node)}",
    ]
    if node.node_type == "figure":
        if not node.figure_tars:
            lines.append("figure_tars: []")
        else:
            names = [
                yaml_quote(_standard_figure_tar_name(node, path))
                for path in node.figure_tars
            ]
            lines.append(f"figure_tars: [{', '.join(names)}]")
    lines.extend(["---", ""])
    if node.node_type == "section":
        if node.body:
            lines.extend([node.body, ""])
    else:
        lines.extend([node.title_raw, ""])
    (node.output_path / "content.md").write_text("\n".join(lines), encoding="utf-8")


def _write_root_references(root: Path, references: dict[int, list[str]]) -> None:
    lines = ["# References", ""]
    for number, entries in sorted(references.items()):
        for index, entry in enumerate(entries, start=1):
            suffix = f" (entry {index})" if len(entries) > 1 else ""
            lines.extend([f"## Reference {number}{suffix}", "", entry, ""])
    (root / "REFERENCES.md").write_text("\n".join(lines), encoding="utf-8")


def _write_node_references(node: Node, references: dict[int, list[str]]) -> None:
    if node.output_path is None or not node.references:
        return
    lines = ["# References used by this node", ""]
    for number in sorted(node.references):
        lines.extend([f"## Reference {number}", ""])
        if number in references:
            entries = references[number]
            if len(entries) == 1:
                lines.append(entries[0])
            else:
                for index, entry in enumerate(entries, start=1):
                    lines.extend([f"### Source entry {index}", "", entry, ""])
        else:
            lines.append(
                "Unresolved: the source cites this reference number, but no matching "
                "entry was present in the extracted References section."
            )
        lines.append("")
    (node.output_path / "references.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _related_filename(link: DirectLink, related_dir: Path) -> str:
    page = link.target_physical_page or 0
    del related_dir
    target_type = safe_segment(link.target_type, max_length=24)
    target_id = safe_segment(link.target_id, max_length=40)
    return f"{target_type}_{target_id}__physical_page_{page:04d}.pdf"


def _write_links(node: Node, page_cache: Path) -> None:
    if node.output_path is None or not node.direct_links:
        return
    related_dir = node.output_path / "related_pages"
    rows: list[list[str]] = []
    for link in sorted(
        node.direct_links,
        key=lambda item: (
            item.relation,
            item.target_type,
            item.target_id,
            item.target_physical_page or 0,
        ),
    ):
        local_copy = ""
        policy = "record_only" if link.relation == "external" else "direct_only"
        if link.relation == "related" and link.target_physical_page is not None:
            related_dir.mkdir(exist_ok=True)
            filename = _related_filename(link, related_dir)
            shutil.copy2(
                page_cache / f"physical_page_{link.target_physical_page:04d}.pdf",
                related_dir / filename,
            )
            local_copy = f"related_pages/{filename}"
        rows.append(
            [
                link.relation,
                link.target_type,
                link.target_id,
                str(link.target_physical_page or ""),
                str(link.occurrences),
                local_copy,
                policy,
            ]
        )
    lines = [
        "# Direct links",
        "",
        "| relation | target_type | target_id | target_physical_page | occurrences | local_copy | policy |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        escaped = [value.replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.append("")
    (node.output_path / "links.md").write_text("\n".join(lines), encoding="utf-8")


def _populate_page_cache(
    page_review_dir: Path, page_cache: Path, expected_page_count: int
) -> int:
    sources = indexed_page_sources(
        page_review_dir, expected_page_count=expected_page_count
    )
    validate_single_page_sources(sources)
    page_cache.mkdir(parents=True, exist_ok=True)
    for index, source in sources.items():
        shutil.copy2(source, page_cache / f"physical_page_{index:04d}.pdf")
    return len(sources)


def _copy_source_pages(node: Node, page_cache: Path) -> None:
    if node.output_path is None:
        raise RuntimeError("node output path is missing")
    source_dir = node.output_path / "source_pages"
    source_dir.mkdir(exist_ok=True)
    for physical_page in node.source_pages:
        filename = f"physical_page_{physical_page:04d}.pdf"
        shutil.copy2(page_cache / filename, source_dir / filename)


def _copy_figure_tar(node: Node) -> None:
    if node.node_type != "figure" or not node.figure_tars:
        return
    if node.output_path is None:
        raise RuntimeError("node output path is missing")
    for source in node.figure_tars:
        target = node.output_path / _standard_figure_tar_name(node, source)
        shutil.copyfile(source, target)
        target.chmod(stat.S_IREAD | stat.S_IWRITE)


def _copy_table_artifacts(node: Node, table_run_dir: Path | None) -> None:
    if node.node_type != "table" or node.table_artifact is None:
        return
    if node.output_path is None:
        raise RuntimeError("node output path is missing")
    if not node.table_artifact.is_dir():
        raise FileNotFoundError(node.table_artifact)
    required = ("table.xlsx", "notes.md", "table_provenance.json")
    for filename in required:
        source = node.table_artifact / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, node.output_path / filename)
    patch = node.table_artifact / "luna_patch.json"
    if patch.is_file():
        shutil.copy2(patch, node.output_path / patch.name)
    if table_run_dir is not None:
        apply_verified_review(node.table_artifact, node.output_path, table_run_dir)


def _load_table_manifest(manual: ParsedManual, table_run_dir: Path) -> None:
    manifest_path = table_run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No Table manifest in {table_run_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported Table manifest; use the workbench schema")
    by_sequence = {node.sequence: node for node in manual.nodes}
    records_by_sequence = {
        int(record["sequence"]): record
        for record in manifest.get("nodes", [])
        if record.get("sequence") is not None
    }
    by_key: dict[str, list[Node]] = {}
    for node in manual.nodes:
        if node.node_type == "table":
            by_key.setdefault(node.identifier_match_key, []).append(node)
    additions: list[tuple[Node, int | None]] = []
    for record in manifest.get("nodes", []):
        if record.get("node_type") != "table":
            continue
        sequence = int(record["sequence"])
        node = by_sequence.get(sequence)
        key = record.get("node_id_match_key") or ""
        # Parser changes can insert or remove nodes and therefore shift every
        # later sequence. Sequence is only a fast path: the semantic identifier
        # must agree before an accepted Table artifact can be attached.
        if (
            node is None
            or node.node_type != "table"
            or (key and node.identifier_match_key != key)
        ):
            options = by_key.get(key, [])
            exact = [
                item
                for item in options
                if (item.node_id_raw or "").casefold()
                == str(record.get("node_id_raw", "")).casefold()
            ]
            node = exact[0] if len(exact) == 1 else options[0] if len(options) == 1 else None
        if node is None:
            node = Node(
                sequence=sequence,
                node_id_raw=(
                    str(record["node_id_raw"])
                    if record.get("node_id_raw") is not None
                    else None
                ),
                title_raw=str(record.get("title_raw") or ""),
                node_type="table",
                synthetic_id=record.get("synthetic_id"),
                identifier_status=str(record.get("identifier_status") or "present"),
                catalog_pages=set(int(value) for value in record.get("catalog_pages", [])),
                target_page=record.get("target_page"),
                source_page_numbers=set(
                    int(value) for value in record.get("source_physical_pages", [])
                ),
            )
            additions.append((node, record.get("parent_sequence")))
            by_sequence[sequence] = node
            by_key.setdefault(node.identifier_match_key, []).append(node)
        else:
            node.source_page_numbers = set(
                int(value) for value in record.get("source_physical_pages", [])
            ) or node.source_page_numbers
            if record.get("title_raw") and not node.title_raw:
                node.title_raw = str(record["title_raw"])
        artifact = record.get("table_artifact")
        if artifact:
            node.table_artifact = (table_run_dir / artifact).resolve()
        node.table_data_status = record.get("data_status") or node.table_data_status
    for node, parent_sequence in additions:
        parent = by_sequence.get(int(parent_sequence)) if parent_sequence is not None else None
        parent_record = (
            records_by_sequence.get(int(parent_sequence))
            if parent_sequence is not None
            else None
        )
        if parent_record is not None:
            parent_key = parent_record.get("node_id_match_key") or ""
            if (
                parent is None
                or parent.node_type != "section"
                or (parent_key and parent.identifier_match_key != parent_key)
            ):
                options = [
                    item
                    for item in manual.nodes
                    if item.node_type == "section"
                    and item.identifier_match_key == parent_key
                ]
                exact = [
                    item
                    for item in options
                    if item.title_raw.casefold()
                    == str(parent_record.get("title_raw") or "").casefold()
                ]
                parent = exact[0] if len(exact) == 1 else options[0] if len(options) == 1 else None
        node.parent = parent
        if parent is not None:
            parent.children.append(node)
        manual.nodes.append(node)


def _merge_intake_figures(manual: ParsedManual, records: list[dict]) -> None:
    """Intake source pages also cover Figures omitted by the printed catalog."""
    for record in records:
        raw_id = record["id"].removeprefix("Figure_")
        identifier = parse_identifier_prefix(raw_id)
        if identifier is None:
            raise ValueError(f"Invalid Figure identifier: {raw_id}")
        matches = [node for node in manual.nodes if node.node_type == "figure" and node.identifier_match_key == identifier.match_key]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous catalog Figure: {raw_id}")
        if matches:
            node = matches[0]
        else:
            caption = parse_caption(record["caption"], expected_kind="figure")
            node = Node(sequence=max(n.sequence for n in manual.nodes) + 1, node_id_raw=raw_id,
                        title_raw=caption.title if caption else record["caption"], node_type="figure")
            parents = [n for n in manual.nodes if n.node_type == "section" and identifier.core.startswith(n.identifier_core + ".")]
            node.parent = max(parents, key=lambda n: n.depth) if parents else None
            if node.parent:
                node.parent.children.append(node)
            manual.nodes.append(node)
        node.target_page = record["page"]
        node.source_page_numbers = {record["page"]}


def _remove_generated_tree(path: Path, output_parent: Path) -> None:
    resolved = path.resolve()
    parent = output_parent.resolve()
    if resolved == parent or not resolved.is_relative_to(parent):
        raise ValueError(f"Refusing to remove path outside output parent: {resolved}")

    def clear_readonly(function, name, _excinfo):
        os.chmod(name, stat.S_IREAD | stat.S_IWRITE)
        function(name)

    shutil.rmtree(resolved, onerror=clear_readonly)


def _build_summary(manual: ParsedManual, root: Path) -> dict[str, object]:
    counts = Counter(node.node_type for node in manual.nodes)
    figure_status = Counter(_data_status(node) for node in manual.nodes if node.node_type == "figure")
    table_status = Counter(_data_status(node) for node in manual.nodes if node.node_type == "table")
    missing_source = [
        f"{node.node_type}:{node.display_id}"
        for node in manual.nodes
        if node.node_type in {"table", "figure"} and node.target_page is None
    ]
    return {
        "output_root": str(root),
        "page_count": manual.page_count,
        "front_pages": manual.front_pages,
        "reference_pages": manual.reference_pages,
        "node_counts": dict(counts),
        "figure_status": dict(figure_status),
        "table_status": dict(table_status),
        "missing_source_leaves": missing_source,
        "references_extracted": len(manual.references),
        "reference_entries_extracted": sum(
            len(entries) for entries in manual.references.values()
        ),
        "structure_validation": manual.structure_validation["summary"],
    }


def build_tree(
    *,
    pdf_path: Path,
    figure_assets: Path,
    figure_records: list[dict],
    output_parent: Path,
    table_run_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    pdf_path = pdf_path.expanduser().resolve()
    figure_assets = figure_assets.expanduser().resolve()
    output_parent = output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output_root = output_parent / pdf_path.stem
    if output_root.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_root}. Use --force to replace it."
        )

    manual = parse_manual(pdf_path)
    table_run_dir = table_run_dir.expanduser().resolve()
    _load_table_manifest(manual, table_run_dir)
    _merge_intake_figures(manual, figure_records)
    _locate_figure_tars(manual.nodes, figure_assets)
    page_review_dir = page_review_directory(figure_assets)

    # Keep the temporary root deliberately short. On Windows, repeating the PDF
    # stem in a staging parent and again in its child can push otherwise valid
    # deep chapter paths beyond MAX_PATH before the final rename.
    staging = output_parent / f".build-{uuid.uuid4().hex[:8]}"
    page_cache = staging / ".page_cache"
    staging.mkdir(parents=True)
    try:
        root = staging
        shutil.copy2(pdf_path, root / pdf_path.name)
        _write_root_references(root, manual.references)
        (root / "STRUCTURE_VALIDATION.json").write_text(
            json.dumps(manual.structure_validation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _assign_output_paths(manual.nodes, root)
        split_count = _populate_page_cache(
            page_review_dir, page_cache, manual.page_count
        )
        if split_count != manual.page_count:
            raise ValueError(
                f"Page split count {split_count} does not match parsed count {manual.page_count}"
            )

        document = pymupdf.open(pdf_path)
        try:
            _scan_node_relations(document, manual.nodes, manual.reference_pages)
            for node in manual.nodes:
                if node.output_path is None:
                    raise RuntimeError("node output path is missing")
                node.output_path.mkdir(parents=True)
                _write_content(node, document)
                _copy_source_pages(node, page_cache)
                _copy_figure_tar(node)
                _copy_table_artifacts(node, table_run_dir)
                _write_node_references(node, manual.references)
                _write_links(node, page_cache)
        finally:
            document.close()

        shutil.rmtree(page_cache)
        backup = output_parent / f".previous-{uuid.uuid4().hex[:8]}"
        if output_root.exists():
            output_root.replace(backup)
        try:
            staging.replace(output_root)
        except OSError:
            if backup.exists():
                backup.replace(output_root)
            raise
        if backup.exists():
            _remove_generated_tree(backup, output_parent)
        summary = _build_summary(manual, output_root)
        return summary
    finally:
        if staging.exists():
            _remove_generated_tree(staging, output_parent)


def summary_json(summary: dict[str, object]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2)
