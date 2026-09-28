# Reviewed teaching examples

The directory contains eight reviewed JSON examples. They are evidence for model
prompts, not unquestionable ground truth, and contain no sampled chart data or
credentials.

Axis planning has a six-example curriculum:

| Example | Axis lesson |
|---|---|
| 3.2.1.10 | Ordinary shared numeric X with several Y mappings |
| 3.2.1.1 | Broken numeric X and side-specific Y mappings |
| 3.2.1.7 | Categorical X and a shared 110/30 boundary |
| 3.2.1.6 | Four spatially separate coordinate frames |
| 3.5.1.1 | Logarithmic X with vertically separate Y mappings |
| 3.2.7.2.3 | Two complete unit systems in one physical frame |

A normal Axis request receives only 3.2.1.10, 3.2.1.7 and 3.2.7.2.3. The full
six-image curriculum is used once only after Axis structure validation fails.

Figures 2.3.2.1 and 3.2.1.1 are coordinate-bearing Grounding examples. They teach
rough 0–1000 positions at axis intersections; Python performs local snapping,
fitting and validation.

Dataset planning uses clean renders of 3.2.1.7, 3.2.1.11 and 3.2.1.6. Each input
lists already approved Axes and each answer is projected to the minimal
`axis/name/kind` protocol. The current Figure is excluded by source ID or explicit
Figure caption. No runtime image hashing is used.
