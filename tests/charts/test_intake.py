import json
from pathlib import Path

import pymupdf
import pytest

from chart_annotator.domain.models import TextObservation
from chart_annotator.intake import (
    discover_figure_pages,
    find_captions,
    inventory_manual,
    materialize_figures,
    normalize_text,
    run_intake,
    split_manual_pages,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def manuals(tmp_path_factory):
    output = tmp_path_factory.mktemp("manuals")
    results = {}
    for label in ("Cast", "Wrought"):
        source = ROOT / "pdfs" / f"Ti-6V-4V({label}).pdf"
        if not source.exists():
            pytest.skip("Local handbook assets are not checked into Git")
        directory, manifest = inventory_manual(source, output)
        split_manual_pages(directory, manifest)
        discover_figure_pages(
            directory, manifest, evidence_dir=output / "runs" / label / "evidence"
        )
        materialize_figures(directory, manifest)
        results[label] = directory, manifest
    return results


@pytest.mark.parametrize("label", ["Cast", "Wrought"])
def test_full_manual_baselines(manuals, label):
    directory, manifest = manuals[label]
    expected = json.loads(
        (ROOT / "tests/charts/fixtures" / f"{label.lower()}_intake.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest.page_count == expected["page_count"]
    assert manifest.figure_job_count == expected["figure_job_count"]
    assert manifest.encryption_present and manifest.empty_password_authenticated
    for actual, baseline in zip(manifest.pages, expected["pages"], strict=True):
        assert actual.figure_status == baseline["figure_status"]
        assert actual.text_layer_available == baseline["text_layer_available"]
        assert [f.figure_id for f in actual.figures] == [
            f["figure_id"] for f in baseline["figures"]
        ]
        assert [f.caption for f in actual.figures] == [
            f["caption"] for f in baseline["figures"]
        ]
        assert Path(actual.pdf_text_observations).exists()
        with pymupdf.open(directory / actual.single_page_pdf) as pdf:
            assert len(pdf) == 1
            assert pdf[0].rotation == actual.rotation
            assert (pdf[0].rect.width, pdf[0].rect.height) == (
                actual.page_width_pdf,
                actual.page_height_pdf,
            )
    assert not list(directory.rglob("*.json"))
    page = directory / manifest.pages[0].single_page_pdf
    with pymupdf.open(page) as pdf:
        original = pdf[0].get_pixmap().samples
    split_manual_pages(directory, manifest)
    with pymupdf.open(page) as pdf:
        assert pdf[0].get_pixmap().samples == original


def test_scanned_caption_and_reference_negatives(manuals):
    page = manuals["Cast"][1].pages[15]
    assert page.ocr_status == "completed"
    assert page.figures[0].figure_id == "3.3.1.1"
    assert len(page.figures[0].observation_ids) == 3
    for label, numbers in [("Cast", [47, 63, 72, 73]), ("Wrought", [63, 98, 141, 142])]:
        for number in numbers:
            page = manuals[label][1].pages[number - 1]
            assert page.figure_status == "non_figure" and not page.figures


def make_pdf(path, blocks, *, rotation=0, password=None):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for y, text in blocks:
            page.insert_text((30, y), text)
        page.set_rotation(rotation)
        kwargs = (
            {
                "encryption": pymupdf.PDF_ENCRYPT_AES_256,
                "owner_pw": "owner",
                "user_pw": password,
            }
            if password is not None
            else {}
        )
        pdf.save(path, **kwargs)
    return path


def test_canonical_intake_multiple_figures_and_repeat(tmp_path):
    source = make_pdf(
        tmp_path / "Manual.pdf",
        [
            (100, "Figure 1.2 First independent caption"),
            (300, "Figure 1.3 Second independent caption"),
        ],
        rotation=90,
        password="",
    )
    data = run_intake(source, tmp_path / "assets", tmp_path / "run", ocr=lambda _: [])
    assert data["schema_version"] == 2
    assert [i["id"] for i in data["figures"]] == ["Figure_1.2", "Figure_1.3"]
    assert not data["issues"]
    page = tmp_path / "assets/Manual/_page_review/page_0001.pdf"
    with pymupdf.open(page) as actual, pymupdf.open(source) as expected:
        assert actual[0].rotation == 90
        assert actual[0].get_pixmap().samples == expected[0].get_pixmap().samples
    assert not list((tmp_path / "assets").rglob("*.json"))
    repeated = run_intake(source, tmp_path / "assets", tmp_path / "run", ocr=lambda _: [])
    assert repeated == data


@pytest.mark.parametrize(
    "text",
    [
        "See Figure 1.2",
        "*See Figure 1.2",
        "Table note\nFigure 1.2 reference",
        "Results are shown in Figure 1.2",
        "Fig. 1.2",
        "Figure 1.2bad_suffix",
    ],
)
def test_references_not_captions(text):
    observation = TextObservation(
        id="block",
        source="pdf_text_layer",
        raw_text=text,
        normalized_text=normalize_text(text),
        method="test",
    )
    assert not find_captions([observation])


@pytest.mark.parametrize("scanned", [False, True])
def test_duplicate_ids_block_automatic_naming(tmp_path, scanned):
    source = make_pdf(
        tmp_path / "Duplicate.pdf",
        []
        if scanned
        else [
            (100, "Figure 1.2 First independent caption"),
            (300, "Figure 1.2 Second independent caption"),
        ],
    )

    def ocr(_):
        return [
            TextObservation(
                id=f"ocr_{y}",
                source="rapidocr_full_page",
                raw_text="Figure 1.2 Caption",
                normalized_text="Figure 1.2 Caption",
                bbox_px=(10, y, 200, y + 20),
                method="test",
            )
            for y in (100, 500)
        ]

    data = run_intake(source, tmp_path / "assets", tmp_path / "run", ocr=ocr)
    assert not data["figures"]
    assert any(issue["code"] == "duplicate_figure_id" for issue in data["issues"])


def test_ocr_failure_is_reported_without_provider_details(tmp_path):
    source = make_pdf(tmp_path / "Empty.pdf", [])

    def fail(_):
        raise RuntimeError("sensitive-provider-details")

    data = run_intake(source, tmp_path / "assets", tmp_path / "run", ocr=fail)
    assert data["issues"] and not data["figures"]
    assert "sensitive-provider-details" not in str(data)


def test_invalid_and_encrypted_inputs_rejected(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"invalid")
    with pytest.raises(pymupdf.FileDataError):
        run_intake(bad, tmp_path / "assets", tmp_path / "run")
    locked = make_pdf(tmp_path / "locked.pdf", [], password="reader")
    with pytest.raises(ValueError, match="password"):
        run_intake(locked, tmp_path / "assets", tmp_path / "run")
    with pytest.raises(ValueError, match="PDF manual"):
        run_intake(
            tmp_path / "old_manifest.json", tmp_path / "assets", tmp_path / "run"
        )
