import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from chart_annotator.domain.models import TextEvidence, TextObservation
from chart_annotator.figure import ingest_figure, render_figure
from chart_annotator.graph import build_workflow
from chart_annotator.text_evidence import (
    extract_pdf_text,
    normalize,
    numeric_value,
    reconcile_text,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "text,value",
    [
        ("−12.5", -12.5),
        ("10⁸", 100000000),
        ("2 × 10⁻²", 0.02),
        ("1.5e3", 1500),
        ("80%", 80),
        ("1,200", 1200),
        (".5", 0.5),
        ("1,20", None),
        ("NaN", None),
        ("1e9999", None),
        ("10^9999", None),
        ("3.2.1.7", None),
        ("Ftu", None),
    ],
)
def test_numeric_normalization(text, value):
    assert numeric_value(text) == value


def observation(identifier, text, box, source="pdf_text_layer"):
    return TextObservation(
        id=identifier,
        source=source,
        raw_text=text,
        normalized_text=normalize(text),
        bbox_px=box,
        image_size=(500, 500),
        method="synthetic evidence",
        numeric_value=numeric_value(text),
    )


def test_all_correspondence_states_and_repeated_values():
    pdf = [
        observation("p0", "120", (0, 0, 30, 20)),
        observation("p1", "120", (0, 100, 30, 120)),
        observation("p2", "10^8", (100, 0, 140, 20)),
        observation("p3", "70", (200, 0, 230, 20)),
        observation("p4", "pdf", (300, 0, 330, 20)),
        observation("p5", "110", (300, 200, 330, 220)),
    ]
    ocr = [
        observation("o0", "120", (0, 0, 30, 20), "rapidocr_general"),
        observation("o1", "120", (0, 100, 30, 120), "rapidocr_general"),
        observation("o2", "10^7", (100, 0, 140, 20), "rapidocr_general"),
        observation("o3", "70", (200, 300, 230, 320), "rapidocr_general"),
        observation("o4", "ocr", (400, 300, 450, 320), "rapidocr_general"),
        observation("o5", "110", (300, 200, 330, 220), "rapidocr_general"),
        observation("o6", "30", (300, 200, 330, 220), "rapidocr_general"),
    ]
    original = [x.model_dump() for x in pdf + ocr]
    matches = reconcile_text(pdf, ocr)
    assert {m.status for m in matches} == {
        "agreed",
        "text_conflict",
        "position_conflict",
        "multiple_candidates",
        "pdf_only",
        "ocr_only",
    }
    assert [
        (m.pdf_observation_ids, m.ocr_observation_ids)
        for m in matches
        if m.status == "agreed"
    ] == [(["p0"], ["o0"]), (["p1"], ["o1"])]
    assert next(
        m for m in matches if m.status == "multiple_candidates"
    ).ocr_observation_ids == ["o5", "o6"]
    assert [x.model_dump() for x in pdf + ocr] == original
    assert sorted(
        i for m in matches for i in m.pdf_observation_ids + m.ocr_observation_ids
    ) == sorted(o.id for o in pdf + ocr)


def test_real_superscripts_and_vertical_words(tmp_path):
    source = ROOT / "tests/charts/fixtures/charts/Figure_3.5.1.1/source.pdf"
    directory, asset = ingest_figure(source, tmp_path)
    asset, _ = render_figure(directory, asset)
    result = extract_pdf_text(directory, asset)
    superscripts = {
        o.normalized_text: o for o in result.observations if "^" in o.normalized_text
    }
    assert superscripts["10^8"].raw_text == "108"
    assert superscripts["10^8"].numeric_value == 100000000
    assert len(superscripts["10^8"].pdf_span_ids) == 2
    assert any(flags & 1 for flags in superscripts["10^8"].font_flags)
    assert any(o.direction_pdf == (0, -1) for o in result.observations)
    assert all(
        o.font_names and o.pdf_span_ids and o.image_size for o in result.observations
    )
    assert (directory / "evidence/pdf_text_raw.json").exists()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_pdf_word_mapping_after_crop_and_rotation(tmp_path, rotation):
    source = tmp_path / "text.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=300, height=400)
        page.insert_text((100, 160), "123.5", fontsize=20)
        page.set_cropbox(pymupdf.Rect(40, 70, 280, 350))
        page.set_rotation(rotation)
        pdf.save(source)
    directory, asset = ingest_figure(source, tmp_path)
    asset, path = render_figure(directory, asset)
    result = extract_pdf_text(directory, asset)
    word = result.observations[0]
    assert word.numeric_value == 123.5
    with Image.open(path) as image:
        assert image.crop(word.bbox_px).convert("L").getextrema()[0] == 0
    restored = pymupdf.Rect(word.bbox_px) * ~pymupdf.Matrix(
        *asset.pdf_to_pixel_transform
    )
    assert tuple(restored) == pytest.approx(word.bbox_pdf, abs=0.0001)


@pytest.mark.parametrize(
    "label,number,expected_pdf",
    [("Cast", 16, False), ("Wrought", 15, True), ("Wrought", 141, False)],
)
def test_real_raster_and_hybrid_pages(tmp_path, label, number, expected_pdf):
    source = tmp_path / "page.pdf"
    with (
        pymupdf.open(ROOT / "pdfs" / f"Ti-6V-4V({label}).pdf") as pdf,
        pymupdf.open() as single,
    ):
        single.insert_pdf(pdf, from_page=number - 1, to_page=number - 1)
        single.save(source)
    result = build_workflow().invoke(
        {
            "input_path": str(source),
            "output_dir": str(tmp_path),
            "mode": "text-evidence",
        }
    )
    assert result["status"] in ("text_evidence_complete", "needs_resolution"), result
    pdf = TextEvidence.model_validate_json(
        Path(result["pdf_text_observations"]).read_text(encoding="utf-8")
    )
    ocr = TextEvidence.model_validate_json(
        Path(result["ocr_text_observations"]).read_text(encoding="utf-8")
    )
    assert bool(pdf.observations) == expected_pdf
    assert pdf.status == ("available" if expected_pdf else "unavailable")
    assert ocr.status == "available" and ocr.observations
    assert all(o.quad_px and o.engine_score is not None for o in ocr.observations)
    canonical = TextEvidence.model_validate_json(
        Path(result["reconciled_text_candidates"]).read_text(encoding="utf-8")
    )
    assert canonical.observations
    reconciliation = json.loads(
        (Path(result["run_dir"]) / "evidence/text_reconciliation.json").read_text(
            encoding="utf-8"
        )
    )
    assert reconciliation["matches"]


def test_ocr_failure_preserves_pdf_artifact(tmp_path, monkeypatch):
    from chart_annotator import text_evidence

    source = tmp_path / "page.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 100), "120")
        pdf.save(source)

    def fail(*args):
        raise RuntimeError("provider-sensitive-text")

    monkeypatch.setattr(text_evidence, "run_general_ocr", fail)
    result = build_workflow().invoke(
        {
            "input_path": str(source),
            "output_dir": str(tmp_path),
            "mode": "text-evidence",
        }
    )
    assert result["status"] == "failed"
    assert (
        TextEvidence.model_validate_json(
            Path(result["pdf_text_observations"]).read_text()
        ).status
        == "available"
    )
    assert (
        TextEvidence.model_validate_json(
            Path(result["ocr_text_observations"]).read_text()
        ).status
        == "failed"
    )
    assert "provider-sensitive-text" not in str(result)


def test_blank_image_is_unavailable_not_ocr_failure(tmp_path):
    source = tmp_path / "blank.png"
    Image.new("RGB", (100, 100), "white").save(source)
    result = build_workflow().invoke(
        {
            "input_path": str(source),
            "output_dir": str(tmp_path),
            "mode": "text-evidence",
        }
    )
    assert result["status"] == "needs_resolution"
    assert result["validation_issues"][0].code == "no_readable_text"
    assert (
        TextEvidence.model_validate_json(
            Path(result["pdf_text_observations"]).read_text()
        ).reason
        == "image_input"
    )
    assert (
        TextEvidence.model_validate_json(
            Path(result["ocr_text_observations"]).read_text()
        ).reason
        == "no_ocr_text"
    )


def test_reconciliation_rejects_duplicate_ids_and_mixed_coordinates():
    item = observation("same", "120", (0, 0, 30, 20))
    with pytest.raises(ValueError, match="unique"):
        reconcile_text([item], [item])
    other = observation("other", "120", (0, 0, 30, 20), "rapidocr_general")
    other.image_size = (600, 600)
    with pytest.raises(ValueError, match="coordinate"):
        reconcile_text([item], [other])
