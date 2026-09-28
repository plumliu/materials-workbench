from pdf_tree_workflow.builder import _write_node_references, _write_root_references
from pdf_tree_workflow.model import Node
from pdf_tree_workflow.parser import (
    _parse_reference_text_blocks,
    _reference_block_order,
    _validate_outline,
)


def test_reference_blocks_are_read_left_column_before_right_column():
    blocks = [
        (336.0, 82.0, 570.0, 120.0, "13. Right column"),
        (72.0, 580.0, 300.0, 690.0, "12. Last left-column entry"),
        (72.0, 100.0, 300.0, 130.0, "1. First left-column entry"),
    ]
    ordered = _reference_block_order(blocks, 612.0)
    assert [block[4] for block in ordered] == [
        "1. First left-column entry",
        "12. Last left-column entry",
        "13. Right column",
    ]


def test_reference_grouping_preserves_cross_page_continuation_and_duplicates():
    references = _parse_reference_text_blocks(
        [
            ["82. Previous", "83. First source entry"],
            ["continued on the next page", "83. Second source entry", "84. Next"],
        ]
    )
    assert references[83] == [
        "First source entry continued on the next page",
        "Second source entry",
    ]
    assert references[84] == ["Next"]


class _OutlineDocument:
    def get_toc(self, simple=False):
        del simple
        return [
            [1, "Text", 1, {}],
            [1, "General", 1, {}],
            [2, "Commercial Designations", 1, {}],
            [1, "References", 6, {}],
        ]


def test_outline_cross_validation_uses_bookmarks_without_replacing_hierarchy():
    root = Node(
        sequence=1,
        node_id_raw="1",
        title_raw="General",
        node_type="section",
        catalog_pages={1},
    )
    child = Node(
        sequence=2,
        node_id_raw="1.1",
        title_raw="Commercial Designations",
        node_type="section",
        catalog_pages={1},
        parent=root,
    )
    root.children.append(child)
    result = _validate_outline(_OutlineDocument(), [root, child])
    assert result["summary"] == {
        "matched": 2,
        "unmatched": 0,
        "issues": 0,
        "status": "pass",
    }
    assert result["authority"] == "parsed_numbered_sections"


def test_duplicate_reference_entries_reach_root_and_node_files(tmp_path):
    references = {83: ["First source entry", "Second source entry"]}
    _write_root_references(tmp_path, references)
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    node = Node(
        sequence=1,
        node_id_raw="1",
        title_raw="General",
        node_type="section",
        output_path=node_dir,
    )
    node.references[83] = 1
    _write_node_references(node, references)
    root_text = (tmp_path / "REFERENCES.md").read_text(encoding="utf-8")
    node_text = (node_dir / "references.md").read_text(encoding="utf-8")
    assert "Reference 83 (entry 1)" in root_text
    assert "Reference 83 (entry 2)" in root_text
    assert "### Source entry 1" in node_text
    assert "### Source entry 2" in node_text
