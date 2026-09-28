"""PDF intake: shared source pages, caption discovery and canonical Figure records."""

import json
import re
import shutil
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pymupdf
from PIL import Image

from chart_annotator.domain.models import (
    FigureCaption,
    IntakeInventory,
    PageRecord,
    TextObservation,
    ValidationIssue,
)

CAPTION = re.compile(r"^Figure\s+([0-9]+(?:\.[0-9A-Za-z]+)+)(?=\.?\s|\.?$)")
# ponytail: intake-only OCR trigger; this is not a figure acceptance threshold.
# M1 runs OCR on every rendered figure, irrespective of PDF text availability.
SPARSE_TEXT_CHARS = 40
OCR_SCALE = 2.0
OCRReader = Callable[[Path], list[TextObservation]]


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def write_json(path: Path, value: object) -> None:
    """Artifacts are immutable. Never silently replace an existing stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def open_pdf(path: Path) -> pymupdf.Document:
    doc = pymupdf.open(path)
    if not doc.is_pdf or not len(doc):
        doc.close()
        raise ValueError("Input must be a nonempty PDF")
    if doc.needs_pass and not doc.authenticate(""):
        doc.close()
        raise ValueError("PDF requires a nonempty password")
    return doc


def image_dimensions(source: Path) -> tuple[int, int]:
    with Image.open(source) as image:
        if (
            image.format not in {"PNG", "JPEG", "TIFF", "BMP", "WEBP"}
            or getattr(image, "n_frames", 1) != 1
        ):
            raise ValueError("Input must be a supported single-frame image")
        image.load()
        return image.size


def inventory_manual(source: Path, output: Path) -> tuple[Path, IntakeInventory]:
    source = source.resolve(strict=True)
    manual_id = source.stem
    pages = []
    encrypted = authenticated = False
    with open_pdf(source) as doc:
        encrypted = bool(doc.metadata.get("encryption"))
        authenticated = bool(doc.authenticate("")) if encrypted else False
        for number, page in enumerate(doc, 1):
            blocks = [b for b in page.get_text("blocks") if b[6] == 0]
            count = len(normalize_text(" ".join(b[4] for b in blocks)))
            pages.append(
                PageRecord(
                    source_page=number,
                    page_width_pdf=page.rect.width,
                    page_height_pdf=page.rect.height,
                    rotation=page.rotation,
                    text_layer_available=bool(count),
                    text_char_count=count,
                    embedded_image_count=len(page.get_images()),
                    object_summary={
                        "text_blocks": len(blocks),
                        "drawings": len(page.get_drawings()),
                        "image_placements": len(page.get_image_info()),
                    },
                )
            )
    directory = output.resolve() / source.stem
    directory.mkdir(parents=True, exist_ok=False)
    return directory, IntakeInventory(
        manual_id=manual_id,
        source_pdf=str(source),
        encryption_present=encrypted,
        empty_password_authenticated=authenticated,
        page_count=len(pages),
        pages=pages,
    )


def split_manual_pages(directory: Path, manifest: IntakeInventory) -> IntakeInventory:
    review = directory / "_page_review"
    review.mkdir(exist_ok=False)
    with open_pdf(directory / manifest.source_pdf) as source:
        for record in manifest.pages:
            path = review / f"page_{record.source_page:04d}.pdf"
            with pymupdf.open() as single:
                single.insert_pdf(
                    source,
                    from_page=record.source_page - 1,
                    to_page=record.source_page - 1,
                )
                single.save(path)
            with open_pdf(path) as checked:
                if len(checked) != 1:
                    raise ValueError("Split PDF must have exactly one page")
                if checked[0].rotation != record.rotation or tuple(checked[0].rect)[
                    2:
                ] != (record.page_width_pdf, record.page_height_pdf):
                    raise ValueError("Split changed page geometry")
            record.single_page_pdf = path.relative_to(directory).as_posix()
    return manifest


class LocalOCR:
    """Lazily create one existing RapidOCR engine per intake run."""

    def __init__(self) -> None:
        self.engine = None

    def __call__(self, image_path: Path) -> list[TextObservation]:
        if self.engine is None:
            from rapidocr import RapidOCR

            self.engine = RapidOCR()
        result = self.engine(image_path)
        with Image.open(image_path) as image:
            size = image.size
        observations = []
        if result.txts is None:
            return observations
        for i, (text, box, score) in enumerate(
            zip(result.txts, result.boxes, result.scores, strict=True)
        ):
            observations.append(
                TextObservation(
                    id=f"text_ocr_{i:04d}",
                    source="rapidocr_full_page",
                    raw_text=text,
                    normalized_text=normalize_text(text),
                    image_size=size,
                    bbox_px=(
                        float(box[:, 0].min()),
                        float(box[:, 1].min()),
                        float(box[:, 0].max()),
                        float(box[:, 1].max()),
                    ),
                    engine_score=float(score),
                    method="RapidOCR full-page, 2x PDF render",
                )
            )
        return observations


def find_captions(observations: list[TextObservation]) -> list[FigureCaption]:
    captions = []
    for index, observation in enumerate(observations):
        match = CAPTION.match(observation.normalized_text)
        if not match:
            continue
        members = [observation]
        bbox = observation.bbox_px
        # OCR emits lines, PDF emits independent layout blocks. Join adjacent
        # OCR continuation lines using measured line height and horizontal overlap.
        if observation.source == "rapidocr_full_page" and bbox:
            for following in observations[index + 1 :]:
                next_box = following.bbox_px
                if not next_box or CAPTION.match(following.normalized_text):
                    break
                previous = members[-1].bbox_px
                height = previous[3] - previous[1]
                if not (
                    previous[1] < next_box[1]
                    and next_box[1] - previous[3] <= height
                    and next_box[0] < bbox[2]
                    and next_box[2] > bbox[0]
                ):
                    break
                members.append(following)
                bbox = (
                    min(bbox[0], next_box[0]),
                    min(bbox[1], next_box[1]),
                    max(bbox[2], next_box[2]),
                    max(bbox[3], next_box[3]),
                )
        captions.append(
            FigureCaption(
                figure_id=match[1],
                caption=" ".join(m.normalized_text for m in members),
                caption_source=observation.source,
                caption_bbox_pdf=observation.bbox_pdf,
                caption_bbox_px=bbox,
                observation_ids=[m.id for m in members],
            )
        )
    return captions


def discover_figure_pages(
    directory: Path, manifest: IntakeInventory, ocr: OCRReader | None = None, *, evidence_dir: Path
) -> IntakeInventory:
    ocr = ocr if ocr is not None else LocalOCR()
    with open_pdf(directory / manifest.source_pdf) as doc:
        for record, page in zip(manifest.pages, doc, strict=True):
            evidence_dir.mkdir(parents=True, exist_ok=True)
            prefix = str(evidence_dir / f"page_{record.source_page:04d}")
            pdf_observations = [
                TextObservation(
                    id=f"text_pdf_{i:04d}",
                    source="pdf_text_layer",
                    raw_text=b[4],
                    normalized_text=normalize_text(b[4]),
                    bbox_pdf=tuple(b[:4]),
                    method="PyMuPDF independent text block; unrotated PDF points",
                )
                for i, b in enumerate(page.get_text("blocks"))
                if b[6] == 0
            ]
            record.pdf_text_observations = f"{prefix}_pdf.json"
            write_json(
                directory / record.pdf_text_observations,
                [o.model_dump(mode="json") for o in pdf_observations],
            )
            record.figures = find_captions(pdf_observations)
            if record.text_char_count < SPARSE_TEXT_CHARS:
                image_path = directory / f"{prefix}.png"
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(OCR_SCALE, OCR_SCALE), alpha=False
                )
                pix.save(image_path)
                transform = page.rotation_matrix * pymupdf.Matrix(OCR_SCALE, OCR_SCALE)
                write_json(
                    directory / f"{prefix}_render.json",
                    {
                        "schema_version": "intake-render/v1",
                        "image": str(image_path),
                        "image_size": [pix.width, pix.height],
                        "pdf_to_pixel_transform": list(transform),
                    },
                )
                try:
                    ocr_observations = ocr(image_path)
                except Exception:  # noqa: BLE001 - third-party OCR error boundary
                    # Never persist provider/third-party exception strings.
                    record.ocr_status = "failed"
                    record.issues.append(
                        ValidationIssue(
                            code="ocr_failed",
                            message="Local page OCR failed",
                            node="discover_figure_pages",
                        )
                    )
                else:
                    record.ocr_status = "completed"
                    record.ocr_text_observations = f"{prefix}_ocr.json"
                    write_json(
                        directory / record.ocr_text_observations,
                        [o.model_dump(mode="json") for o in ocr_observations],
                    )
                    for caption in find_captions(ocr_observations):
                        existing = next(
                            (
                                f
                                for f in record.figures
                                if f.figure_id == caption.figure_id
                                and f.caption_source == "pdf_text_layer"
                                and f.caption_bbox_pdf is not None
                                and caption.caption_bbox_px is not None
                                and (
                                    pymupdf.Rect(f.caption_bbox_pdf) * transform
                                ).intersects(pymupdf.Rect(caption.caption_bbox_px))
                            ),
                            None,
                        )
                        if existing:
                            existing.observation_ids.extend(caption.observation_ids)
                            if existing.caption != caption.caption:
                                record.issues.append(
                                    ValidationIssue(
                                        code="caption_text_conflict",
                                        message="PDF and OCR captions differ; both observations retained",
                                        evidence_ids=existing.observation_ids,
                                    )
                                )
                        else:
                            record.figures.append(caption)
                    if not ocr_observations:
                        record.issues.append(
                            ValidationIssue(
                                code="no_readable_text",
                                message="No caption evidence available after OCR",
                            )
                        )
            record.figure_status = (
                "needs_review"
                if record.issues
                else ("detected" if record.figures else "non_figure")
            )
    return manifest


def materialize_figures(
    directory: Path, manifest: IntakeInventory
) -> list[dict]:
    counts = Counter(f.figure_id for p in manifest.pages for f in p.figures)
    jobs = []
    for page in manifest.pages:
        duplicates = [f.figure_id for f in page.figures if counts[f.figure_id] > 1]
        if duplicates:
            page.figure_status = "needs_review"
            page.issues.append(
                ValidationIssue(
                    code="duplicate_figure_id",
                    message="Duplicate Figure ID; automatic naming stopped",
                    evidence_ids=duplicates,
                )
            )
        if page.figure_status != "detected":
            continue
        for caption in page.figures:
            figure_id = f"Figure_{caption.figure_id}"
            figure_dir = directory / figure_id
            figure_dir.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(directory / page.single_page_pdf, figure_dir / f"{figure_id}.pdf")
            jobs.append({"id": figure_id, "page": page.source_page, "caption": caption.caption})
    manifest.figure_job_count = len(jobs)
    return jobs


def run_intake(source: Path, output: Path, run_dir: Path, ocr: OCRReader | None = None) -> dict:
    """Publish the workbench's sole intake contract; no legacy run descriptors."""
    from materials_workbench.storage import write_json as atomic_json

    if source.suffix.lower() != ".pdf":
        raise ValueError("Intake requires a PDF manual")
    directory, manifest = inventory_manual(source, output)
    split_manual_pages(directory, manifest)
    discover_figure_pages(directory, manifest, ocr, evidence_dir=run_dir / "intake/evidence")
    jobs = materialize_figures(directory, manifest)
    stat = source.stat()
    data = {
        "schema_version": 2, "name": source.stem, "revision": 1,
        "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "page_count": manifest.page_count,
        "figures": jobs,
        "issues": [dict(page=page.source_page, **issue.model_dump(mode="json")) for page in manifest.pages for issue in page.issues],
    }
    atomic_json(run_dir / "manual.json", data)
    return data
