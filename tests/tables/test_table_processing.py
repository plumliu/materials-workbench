from pdf_tree_workflow.table_processing import normalize_table_html, quality_issues


def test_html_grid_expands_rowspan_and_colspan():
    grid = normalize_table_html(
        """
        <table>
          <tr><th rowspan="2">Heat</th><th colspan="2">Property</th></tr>
          <tr><th>Fty</th><th>NTS</th></tr>
          <tr><td>A</td><td>110</td><td>166</td></tr>
        </table>
        """
    )
    assert grid["rows"] == [
        ["Heat", "Property", ""],
        ["", "Fty", "NTS"],
        ["A", "110", "166"],
    ]
    assert any(cell["rowspan"] == 2 for cell in grid["cells"])
    assert any(cell["colspan"] == 2 for cell in grid["cells"])


def test_cross_page_is_provenance_info_not_automatic_review():
    candidate = {
        "node_id_raw": "3.2.5.1",
        "source_physical_pages": [54, 55],
        "grid": normalize_table_html("<table><tr><td>A</td></tr></table>"),
    }
    issues = quality_issues(candidate)
    assert any(issue["code"] == "multi_page_table" for issue in issues)
    assert not any(issue["severity"] in {"review", "error"} for issue in issues)
