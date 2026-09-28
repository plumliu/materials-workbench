import pymupdf

from pdf_tree_workflow.preparation import _contiguous_runs, _write_segment


def test_contiguous_runs_never_bridge_removed_pages():
    assert _contiguous_runs([11, 12, 14, 31, 32, 33, 63]) == [
        [11, 12],
        [14],
        [31, 32, 33],
        [63],
    ]


def test_segment_contains_only_selected_pages(tmp_path):
    source = tmp_path / "manual.pdf"
    with pymupdf.open() as document:
        for label in ("first", "second", "third"):
            page = document.new_page()
            page.insert_text((72, 72), label)
        document.save(source)
    target = tmp_path / "segments" / "selected.pdf"
    _write_segment(source, [1, 3], target)
    with pymupdf.open(target) as segment:
        assert [page.get_text().strip() for page in segment] == ["first", "third"]
