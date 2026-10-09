# Reviewed teaching examples

The directory contains eight reviewed JSON examples. They are evidence for model
prompts, not unquestionable ground truth, and contain no sampled chart data or
credentials.

Axis planning has a six-example curriculum:

| Example | Axis lesson |
|---|---|
| 3.2.1.1 | Broken numeric X and side-specific Y mappings |
| 3.2.1.7 | Categorical X and a shared 110/30 boundary |
| 3.2.1.11 | Left/right scales with the same units but different origins; four independent Ys |
| 3.2.1.6 | Four spatially separate coordinate frames |
| 3.5.1.1 | Logarithmic X with vertically separate Y mappings |
| 3.2.7.2.3 | Two complete unit systems in one physical frame |

The first Axis request receives all six images; subsequent turns retain them
in the unchanged message history rather than appending them again.
Structural validation cannot detect every plausible but incorrect mapping.
Figure 3.2.1.10 is retained for offline regression tests and is not sent in Axis requests.

Figure 3.2.1.11 joins the fixed Axis curriculum alongside 3.2.1.7, making six
images. Its clean image and four-Y answer are reused from the existing example.
RA=0 and e=10 align horizontally in this figure; Ftu=170 and Fty=180 also align.
Figure 3.2.1.1 remains as a contrasting example where RA/e really share a scale.
Figure 3.2.1.7 teaches categorical X and shared mappings, and also remains available
for Dataset teaching. Axis selection does not exclude the current figure, so a subsequent run
of 3.2.1.11 is a teaching-example replay, not an independent quality evaluation.

Axis and Dataset repairs retain previous proposals in their message history and
receive tool feedback with current issues and zero-based field paths. Structural
acceptance does not prove image semantics. Preserve unaffected content unless a
dependent correction or current image evidence requires a change.

Figures 2.3.2.1 and 3.2.1.1 are coordinate-bearing Grounding examples. They teach
rough 0–1000 positions at axis intersections; Python performs local snapping,
fitting and validation.

Dataset planning uses clean renders of 3.2.1.7, 3.2.1.11 and 3.2.1.6. Each input
lists already approved Axes and each answer is projected to the minimal
`axis/name/kind` protocol. The current Figure is excluded by source ID or explicit
Figure caption. No runtime image hashing is used.
