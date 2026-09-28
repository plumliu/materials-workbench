import pytest

from pdf_tree_workflow.identifiers import parse_caption, parse_identifier_prefix
from pdf_tree_workflow.parser import _heading_match


@pytest.mark.parametrize(
    ("value", "core", "suffixes", "key"),
    [
        ("3.1.1", "3.1.1", (), "3.1.1"),
        ("3.1.1(a)", "3.1.1", ("a",), "3.1.1|a"),
        ("3.1.1(A)", "3.1.1", ("a",), "3.1.1|a"),
        ("3.1.1a", "3.1.1", ("a",), "3.1.1|a"),
        ("3.1.1A", "3.1.1", ("a",), "3.1.1|a"),
        ("3.1.1.a", "3.1.1", ("a",), "3.1.1|a"),
        ("3.1.1.A", "3.1.1", ("a",), "3.1.1|a"),
        ("3.1.1(1)", "3.1.1", ("1",), "3.1.1|1"),
        ("3.1.1A(1)", "3.1.1", ("a", "1"), "3.1.1|a|1"),
        ("3.1.1-A", "3.1.1", ("a",), "3.1.1|a"),
        ("3.1.1–A", "3.1.1", ("a",), "3.1.1|a"),
        ("3.1.1[A]", "3.1.1", ("a",), "3.1.1|a"),
        ("3·1·1(a)", "3.1.1", ("a",), "3.1.1|a"),
        ("３．１．１（Ａ）", "3.1.1", ("a",), "3.1.1|a"),
    ],
)
def test_identifier_variants(value, core, suffixes, key):
    parsed = parse_identifier_prefix(value)
    assert parsed is not None
    assert parsed.core == core
    assert parsed.suffixes == suffixes
    assert parsed.match_key == key
    assert parsed.depth == 3


def test_caption_preserves_raw_identifier():
    parsed = parse_caption("Table 3.1.1(A) Some property")
    assert parsed is not None
    assert parsed.identifier.raw == "3.1.1(A)"
    assert parsed.title == "Some property"


@pytest.mark.parametrize(
    "value",
    [
        "Table No. 3.1.1 Some property",
        "Tbl. 3.1.1 Some property",
        "Figure No. 3.1.1A Some plot",
        "Fig. 3.1.1A Some plot",
    ],
)
def test_caption_kind_aliases(value):
    parsed = parse_caption(value)
    assert parsed is not None
    assert parsed.kind in {"table", "figure"}


def test_body_cross_reference_is_not_a_heading():
    assert _heading_match("3.2.7.2.4 and 3.2.7.2.5 show the results") is None


def test_root_material_formula_is_not_a_heading():
    assert _heading_match("4V alloy castings at room temperature") is None


@pytest.mark.parametrize(
    "value",
    [
        "3.3.7.1.10, the fracture toughness would be",
        "3.3.1.16). It is not clear how this behavior",
        "1.25 inch specimen",
        "2 to 4 hrs in vacuum",
        "3.5% salt water",
    ],
)
def test_sentence_and_measurement_prefixes_are_not_headings(value):
    assert _heading_match(value) is None
