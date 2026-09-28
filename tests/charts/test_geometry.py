from pathlib import Path

import pytest

from chart_annotator.figure import ingest_figure, render_figure
from chart_annotator.geometry import detect
from chart_annotator.text_evidence import extract_pdf_text


def test_real_shared_t_preserves_110_and_30_despite_pdf_ink_gap(tmp_path):
    source = Path(__file__).parent / "fixtures/charts/Figure_3.2.1.7/source.pdf"
    directory, asset = ingest_figure(source, tmp_path)
    asset, image = render_figure(directory, asset)
    words = extract_pdf_text(directory, asset).observations
    geometry = detect(image, words)
    values = {w.id: w.numeric_value for w in words}
    shared = [
        t
        for t in geometry.ticks
        if {110, 30} <= {values[i] for i in t.text_observation_ids}
    ]
    assert shared
    assert all(t.status == "multiple_candidates" for t in shared)
    # No model or human calibration reference participates in candidate production.


def test_glyph_strokes_are_not_ticks(tmp_path):
    from PIL import Image, ImageDraw

    image = tmp_path / "glyph.png"
    canvas = Image.new("RGB", (300, 250), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((50, 30, 250, 200), outline="black", width=3)
    draw.line((50, 90, 64, 90), fill="black", width=2)
    # Nearby disconnected numeral-like strokes; they must not intersect the spine.
    draw.rectangle((37, 115, 42, 128), outline="black", width=1)
    canvas.save(image)
    geometry = detect(image, [])
    left = min(
        (s for s in geometry.spines if s.orientation == "vertical"),
        key=lambda s: s.coordinate,
    )
    positions = [t.intersection_px[1] for t in geometry.ticks if t.spine_id == left.id]
    assert any(abs(y - 90) <= 2 for y in positions)
    assert not any(113 <= y <= 131 for y in positions)


def test_wrought_thick_axis_does_not_bridge_nearby_numerals():
    # Wrought p15, Figure 1.5.3, original pixels [115,1079,807,1186].
    # The old thickness-sized closing produced 33 candidates from seven ticks.
    image = Path(__file__).parent / "fixtures/wrought_tick_strip.png"
    geometry = detect(image, [])
    spine = max(
        (s for s in geometry.spines if s.orientation == "horizontal"),
        key=lambda s: s.bbox[2] - s.bbox[0],
    )
    positions = [t.intersection_px[0] for t in geometry.ticks if t.spine_id == spine.id]
    assert positions == pytest.approx(
        [42, 142.5, 242.5, 340.26, 437.5, 538.64, 638.09], abs=1
    )
