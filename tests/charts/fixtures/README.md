# M0 regression fixtures

`cast_intake.json` and `wrought_intake.json` freeze the independently reviewed
2026-09-14 manual intake audit: SHA-256, every physical page, Figure IDs, complete
captions and evidence sources. Runtime code does not read these expectations.

`charts/Figure_*/` contains eight actual human reference TARs, their PDF image
members, fixed 2x PDF renders, versioned empty structural ChartPlans and reference
metadata. The source was the local human TAR collection recorded by the Cast
structure audit. `reference.json` records each original TAR hash and known
difficulty. TAR data points remain **evaluation-only**, never model inputs or
candidate-generation inputs. No human data points are copied into ChartPlan.

The historical fixtures record axis names/scales, Dataset names/membership/kinds and
Point Group order. `reference_region_*` IDs distinguish source axes; they are
**not** a gold panel count. Panel grouping, text/geometry candidate IDs, shared-axis
relationships, and deterministic calibration gates await M1–M3 evidence. Original
human calibration is retained in `reference.json` and TAR only for evaluation.
These are structural regression fixtures, not exported production projects.

The WPD empty-project contract remains PROJECT_PLAN §10: image and project JSON,
calibrated axes, unique empty Datasets and ordered Point Groups, no fabricated
category points, no sampled points, and equivalent structure after TAR roundtrip.
The exporter/reader and actual WPD 4.7 compatibility validation are now implemented;
see docs/E2E_VALIDATION.md.

2026-09-14 reference correction: these human artifacts are fallible historical
records, not unquestionable semantic truth. Figure 3.5.1.6 has six historical
Datasets and isLogX=false even though the printed ticks are logarithmic. The skill's
ten-Dataset Smooth/Notch explanation also lacks the stated callouts in that image.
Keep the original files for provenance. After image and PDF marker inspection the
user froze a separate corrected 1-Axis/6-Dataset reference: log X, three ordinary
observation series, the filled-triangle slash subset, and neutral upper/lower
curves. No confirmed open-triangle slash occurrence, no Smooth/Notch attribution.
Reviewed runtime examples are separately authored in prompts/v2/examples with
empty data and no human calibration. Eight historical images (including the frozen
Figure 3.5.1.6) and two explicitly illustrative special-marker charts supply the teaching set. Tests of old
fixture serialization do not validate their scientific interpretation.
