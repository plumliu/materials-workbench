"""Local OCR beside candidate spines, retaining full-page and PDF evidence."""

from pathlib import Path

import numpy as np
from PIL import Image
from rapidocr import RapidOCR

from chart_annotator.domain.models import TextEvidence, TextObservation
from chart_annotator.domain.workflow import Geometry
from chart_annotator.intake import write_json
from chart_annotator.text_evidence import normalize, numeric_value


def read_ticks(
    image: Path,
    geometry: Geometry,
    texts: list[TextObservation],
    source_id: str,
    directory: Path,
    *,
    search_unlabelled=False,
    grounded_range=None,
) -> TextEvidence:
    reader = RapidOCR()
    observations, raw = [], []
    with Image.open(image) as source:
        for spine in geometry.spines:
            related = [
                t
                for t in geometry.ticks
                if t.spine_id == spine.id and t.text_observation_ids
            ]
            ids = {i for t in related for i in t.text_observation_ids}
            boxes = [o.bbox_px for o in texts if o.id in ids and o.bbox_px]
            glyph_height = (
                max(b[3] - b[1] for b in boxes) if boxes else max(12, spine.width * 4)
            )
            if grounded_range is not None:
                direction, low, high = grounded_range
                pad = max(16, max(geometry.image_size) * 0.04)
                coordinate = spine.coordinate
                boxes = [
                    (low, coordinate - pad, high, coordinate + pad)
                    if direction == "x"
                    else (coordinate - pad, low, coordinate + pad, high)
                ]
            if not boxes:
                if not search_unlabelled:
                    continue
                # Grounding can recover a spine without any global text/tick IDs.
                # Search an image-scaled strip; its words still need actual T ink.
                pad = max(16, max(geometry.image_size) * 0.04)
                a, b, c, d = spine.bbox
                boxes = [(a - pad, b - pad, c + pad, d + pad)]
            # One strip per spine, not one model/engine initialization per tick.
            x0 = max(0, int(min(b[0] for b in boxes) - glyph_height / 2))
            y0 = max(0, int(min(b[1] for b in boxes) - glyph_height / 2))
            x1 = min(source.width, int(max(b[2] for b in boxes) + glyph_height / 2 + 1))
            y1 = min(
                source.height, int(max(b[3] for b in boxes) + glyph_height / 2 + 1)
            )
            crop = source.crop((x0, y0, x1, y1)).convert("RGB")
            directory.mkdir(parents=True, exist_ok=True)
            crop.save(directory / f"{spine.id}.png")
            result = reader(np.asarray(crop), return_word_box=True, text_score=0.0)
            words = [
                word for line in (result.word_results or []) for word in line if word
            ]
            raw.append(
                {"spine_id": spine.id, "crop_bbox": [x0, y0, x1, y1], "words": words}
            )
            for text, score, quad in words:
                normalized = normalize(text)
                value = numeric_value(normalized)
                if value is None:
                    continue
                translated = tuple((float(x) + x0, float(y) + y0) for x, y in quad)
                xs, ys = zip(*translated, strict=True)
                observations.append(
                    TextObservation(
                        id=f"text_tick_{len(observations):04d}",
                        source="rapidocr_tick_crop",
                        raw_text=text,
                        normalized_text=normalized,
                        numeric_value=value,
                        bbox_px=(min(xs), min(ys), max(xs), max(ys)),
                        quad_px=translated,
                        image_size=geometry.image_size,
                        engine_score=float(score),
                        method=f"RapidOCR spine strip {spine.id}; crop translated to render",
                    )
                )
    # RapidOCR's word box output is already lists of Python scalars.
    write_json(directory / "raw.json", raw)
    return TextEvidence(
        source_id=source_id,
        status="available" if observations else "unavailable",
        reason=None if observations else "no_numeric_tick_text",
        observations=observations,
    )
