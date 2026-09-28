from pdf_tree_workflow.table_processing import (
    _caption_in_text,
    _looks_like_cross_page_continuation,
    _looks_like_standalone_table_title,
    normalize_table_html,
)


def test_body_cross_reference_is_not_used_as_a_pending_table_caption():
    caption, parsed = _caption_in_text(
        {"text": "See Table 3.2.1.2 for details.\nThe values are summarized below."}
    )
    assert caption == ""
    assert parsed is None


def test_line_leading_table_caption_is_detected():
    caption, parsed = _caption_in_text(
        {"text": "Table No. 3.2.1.2 Tensile properties"}
    )
    assert caption == "Table No. 3.2.1.2 Tensile properties"
    assert parsed is not None
    assert parsed.identifier.match_key == "3.2.1.2"


def test_abbreviated_line_leading_table_caption_is_detected():
    _caption, parsed = _caption_in_text({"text": "Tbl. 3.2.1.2 Tensile properties"})
    assert parsed is not None
    assert parsed.identifier.match_key == "3.2.1.2"


def test_standalone_unnumbered_table_title_is_a_hard_boundary():
    assert _looks_like_standalone_table_title("AMS 4993")
    assert _looks_like_standalone_table_title("ASTM B348")
    assert not _looks_like_standalone_table_title("Notes:")
    assert not _looks_like_standalone_table_title("See Table 3.2.1.2 for details.")


def test_cross_page_continuation_requires_compatible_repeated_header_and_layout():
    first = normalize_table_html("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")
    same_header = "<table><tr><th>A</th><th>B</th></tr><tr><td>3</td><td>4</td></tr></table>"
    different_header = "<table><tr><th>Element</th><th>Minimum</th><th>Maximum</th></tr></table>"
    previous = {"grid": first, "bbox": [10, 700, 980, 970]}
    assert _looks_like_cross_page_continuation(previous, same_header, [10, 20, 980, 500])
    assert not _looks_like_cross_page_continuation(previous, different_header, [10, 20, 980, 500])
    assert not _looks_like_cross_page_continuation(previous, same_header, [10, 400, 980, 900])
