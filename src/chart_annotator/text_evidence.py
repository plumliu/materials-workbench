"""Local text observations and conservative correspondence, never calibration."""

import math
import re
import unicodedata
from pathlib import Path

import pymupdf
from PIL import Image

from chart_annotator.domain.models import (
    SourceAsset,
    TextEvidence,
    TextMatch,
    TextObservation,
)
from chart_annotator.intake import open_pdf, write_json

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def normalize(text: str) -> str:
    text = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda match: "^" + match[0].translate(SUPERSCRIPTS), text
    )
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    return " ".join(text.split())


def numeric_value(text: str) -> float | None:
    value = normalize(text).replace(" ", "")
    value = value.removesuffix("%")  # Values stay in displayed percent units.
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
    try:
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            number = float(value)
        elif match := re.fullmatch(
            r"(?:(?P<coefficient>[+-]?(?:\d+(?:\.\d*)?|\.\d+))[×x*])?10\^(?P<exponent>[+-]?\d+)",
            value,
        ):
            number = float(match["coefficient"] or 1) * 10.0 ** int(match["exponent"])
        else:
            return None
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def extract_pdf_text(directory: Path, asset: SourceAsset) -> TextEvidence:
    if asset.kind != "pdf":
        result = TextEvidence(
            source_id=asset.source_id, status="unavailable", reason="image_input"
        )
    else:
        if (
            asset.pdf_to_pixel_transform is None
            or not asset.render_width
            or not asset.render_height
        ):
            raise ValueError("PDF text needs the recorded render transform")
        transform = pymupdf.Matrix(*asset.pdf_to_pixel_transform)
        observations = []
        with open_pdf(directory / asset.path) as pdf:
            raw = pdf[0].get_text(
                "rawdict",
                flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES,
            )
        write_json(directory / "evidence/pdf_text_raw.json", raw)
        for b, block in enumerate(raw["blocks"]):
            for line_index, line in enumerate(block.get("lines", [])):
                line_id = f"pdf_b{b}_l{line_index}"
                # Keep span boundaries to interpret explicit PDF superscript flags.
                tokens, current = [], []
                for s, span in enumerate(line["spans"]):
                    for char in span["chars"]:
                        if char["c"].isspace():
                            if current:
                                tokens.append(current)
                                current = []
                        else:
                            current.append((char, span, f"{line_id}_s{s}"))
                if current:
                    tokens.append(current)
                for token in tokens:
                    rect = pymupdf.Rect(token[0][0]["bbox"])
                    layout, previous_super = "", False
                    for char, span, _ in token:
                        rect |= pymupdf.Rect(char["bbox"])
                        is_super = bool(span["flags"] & 1)
                        if is_super and not previous_super and layout:
                            layout += "^"
                        layout += char["c"]
                        previous_super = is_super
                    pixel_rect = rect * transform
                    direction = pymupdf.Point(*line["dir"])
                    vector = direction * transform - pymupdf.Point(0, 0) * transform
                    length = math.hypot(vector.x, vector.y)
                    normalized = normalize(layout)
                    spans = {span_id: span for _, span, span_id in token}
                    observations.append(
                        TextObservation(
                            id=f"text_pdf_{len(observations):04d}",
                            source="pdf_text_layer",
                            raw_text="".join(char["c"] for char, _, _ in token),
                            normalized_text=normalized,
                            bbox_pdf=tuple(rect),
                            bbox_px=tuple(pixel_rect),
                            image_size=(asset.render_width, asset.render_height),
                            method="PyMuPDF characters grouped at whitespace; explicit superscript span flags",
                            numeric_value=numeric_value(normalized),
                            direction_pdf=tuple(direction),
                            direction_px=(vector.x / length, vector.y / length),
                            line_id=line_id,
                            pdf_span_ids=list(spans),
                            font_names=[s["font"] for s in spans.values()],
                            font_sizes_pdf=[s["size"] for s in spans.values()],
                            font_flags=[s["flags"] for s in spans.values()],
                        )
                    )
        result = TextEvidence(
            source_id=asset.source_id,
            status="available" if observations else "unavailable",
            reason=None if observations else "no_pdf_text",
            observations=observations,
        )
    write_json(directory / "evidence/pdf_text.json", result.model_dump(mode="json"))
    return result


def run_general_ocr(
    directory: Path, asset: SourceAsset, image_path: Path
) -> TextEvidence:
    from rapidocr import RapidOCR

    with Image.open(image_path) as image:
        size = image.size
    if size != (asset.render_width, asset.render_height):
        raise ValueError("OCR image dimensions differ from recorded render")
    output = RapidOCR()(image_path, return_word_box=True, text_score=0.0)
    raw, observations = [], []
    lines = (
        zip(
            output.txts,
            output.boxes,
            output.scores,
            output.word_results,
            strict=True,
        )
        if output.txts
        else ()
    )
    for i, (text, box, score, words) in enumerate(lines):
        if not words:
            raise ValueError("OCR word locations unavailable for recognized line")
        raw.append(
            {
                "line_id": f"ocr_line_{i}",
                "text": text,
                "box": box.tolist(),
                "score": float(score),
                "words": words,
            }
        )
        for text, score, box in words:
            if box is None:
                raise ValueError("OCR word location unavailable")
            xs, ys = zip(*box, strict=True)
            normalized = normalize(text)
            observations.append(
                TextObservation(
                    id=f"text_ocr_{len(observations):04d}",
                    source="rapidocr_general",
                    raw_text=text,
                    normalized_text=normalized,
                    bbox_px=(min(xs), min(ys), max(xs), max(ys)),
                    quad_px=tuple(tuple(p) for p in box),
                    image_size=size,
                    line_id=f"ocr_line_{i}",
                    method="RapidOCR full rendered image, return_word_box=True",
                    numeric_value=numeric_value(normalized),
                    engine_score=float(score),
                )
            )
    write_json(directory / "evidence/ocr_general_raw.json", raw)
    result = TextEvidence(
        source_id=asset.source_id,
        status="available" if observations else "unavailable",
        reason=None if observations else "no_ocr_text",
        observations=observations,
    )
    write_json(directory / "evidence/ocr_general.json", result.model_dump(mode="json"))
    return result


def reconcile_text(
    pdf: list[TextObservation], ocr: list[TextObservation]
) -> list[TextMatch]:
    """Mutual box-center containment; ambiguous components never select a winner.

    ponytail: quadratic scan is bounded by one figure's text; spatial indexing
    only if profiling on denser figures shows a bottleneck.
    """
    if len({o.id for o in pdf + ocr}) != len(pdf) + len(ocr):
        raise ValueError("Text observation IDs must be unique across sources")
    sizes = {o.image_size for o in pdf + ocr if o.image_size is not None}
    if len(sizes) > 1:
        raise ValueError("Cannot reconcile different image coordinate systems")
    neighbors = {}
    all_obs = {o.id: o for o in pdf + ocr}
    for p in pdf:
        for o in ocr:
            if p.bbox_px is None or o.bbox_px is None:
                continue
            a, b = pymupdf.Rect(p.bbox_px), pymupdf.Rect(o.bbox_px)
            if a.contains((b.tl + b.br) / 2) and b.contains((a.tl + a.br) / 2):
                neighbors.setdefault(p.id, set()).add(o.id)
                neighbors.setdefault(o.id, set()).add(p.id)
    matches, used = [], set()
    pdf_ids = {o.id for o in pdf}

    def add(ids, status, method):
        used.update(ids)
        matches.append(
            TextMatch(
                id=f"text_match_{len(matches):04d}",
                status=status,
                pdf_observation_ids=sorted(set(ids) & pdf_ids),
                ocr_observation_ids=sorted(set(ids) - pdf_ids),
                method=method,
            )
        )

    for identifier in neighbors:
        if identifier in used:
            continue
        component, pending = set(), [identifier]
        while pending:
            current = pending.pop()
            if current not in component:
                component.add(current)
                pending.extend(neighbors.get(current, set()) - component)
        status = (
            "multiple_candidates"
            if len(component) > 2
            else "agreed"
            if len({all_obs[i].normalized_text for i in component}) == 1
            else "text_conflict"
        )
        add(component, status, "mutual_bbox_center_containment")
    for p in pdf:
        if p.id in used:
            continue
        same_pdf = [
            o.id
            for o in pdf
            if o.id not in used and o.normalized_text == p.normalized_text
        ]
        same_ocr = [
            o.id
            for o in ocr
            if o.id not in used and o.normalized_text == p.normalized_text
        ]
        if same_ocr:
            add(
                same_pdf + same_ocr,
                "position_conflict"
                if len(same_pdf) == len(same_ocr) == 1
                else "multiple_candidates",
                "same_text_without_unique_spatial_match",
            )
    for observation in pdf + ocr:
        if observation.id not in used:
            add(
                [observation.id],
                "pdf_only" if observation.id in pdf_ids else "ocr_only",
                "no_corresponding_observation",
            )
    return matches
