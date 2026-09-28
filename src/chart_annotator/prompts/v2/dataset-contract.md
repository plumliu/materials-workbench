# Minimal Dataset contract

The supplied Axes are immutable ground truth. In the CURRENT image, every colored
line and ring was produced from Python-validated calibration; it is not rough model
grounding. Use the listed Axis IDs exactly. Do not create panels, Axes, coordinates,
styles, evidence IDs, conditions or sampled data as separate fields.

Return only:

- `datasets`: one object per visible collection object, with exactly `axis`, `name`
  and `kind`;
- `unresolved`: specific visual ambiguities, otherwise `[]`.

Allowed `kind` values are `scatter`, `curve`, `bar`, `point_group`,
`range_boundary` and `distribution_boundary`.

## Non-negotiable final-name rules

The `name` field is the exact final string that a human will see in
WebPlotDigitizer. It is not a summary. Apply all of the following rules before
returning JSON:

- Copy every visible legend or callout condition in full. Never abbreviate,
  shorten, paraphrase, simplify, normalize or translate it, even when it is
  long. Preserve its capitalization, numbers, punctuation, parentheses, units
  and material-processing abbreviations. Join a visually wrapped label in
  reading order, but do not discard any words.
- The literal separator ` | ` (space, vertical bar, space) is mandatory between
  the semantic parts of ordinary Dataset names. Do not replace it with spaces,
  hyphens, commas or slashes.
- A marker with a printed condition MUST be named
  `<property> | <complete visible condition text> | <fill> <shape>`.
  Without a condition, use `<property> | <fill> <shape>`.
- A curve with a printed condition MUST be named
  `<property> | <complete visible condition text> | <line style> trend curve`.
  Without a condition, use `<property> | <line style> trend curve`.
- Special forms explicitly defined below, such as
  `<property> Average Value + Spread of Value`, keep their exact special form.
  Do not apply the ordinary marker format to them.

For example, if the image prints `Annealed: 1300F, 2hrs in Argon, AC`, then:

- VALID: `Ftu | Annealed: 1300F, 2hrs in Argon, AC | filled circle`
- INVALID: `Ftu Annealed filled circle`
- INVALID: `Ftu | Annealed | filled circle`

Both invalid names lose source information. Never emit them. If the complete
condition cannot be read, describe that exact problem in `unresolved`; do not
silently substitute a shorter condition.

Name each Dataset from image truth, following these rules:

- Prefer exact visible legend or callout text. Preserve language, capitalization,
  punctuation, parentheses and units.
- Names must be unique across the whole Figure project. On collision, append only
  the smallest visible Axis discriminator: first the differing numeric property and
  unit, then the category/X title, and the full Axis name only as a last resort.
  Never append an internal ID or enumerate all categories.
- A marker name contains its property/condition when needed and its visible style:
  `open`, `filled` or `half-filled` plus the exact shape. Use `inverted triangle`
  for a downward triangle. `cross`, `plus` and `star` are exact shape names.
- A visible slash/cross/star modifier is a separate Dataset only when the chart
  defines its meaning; include the base style, modifier and meaning in the name.
- A curve name contains its property/condition when supported, visible line style
  (`solid`, `dashed`, `dotted` or `dash-dot`) and role such as `trend curve`.
  Markers and a visible curve are separate Datasets.
- A range band is two `range_boundary` Datasets named `upper boundary` and
  `lower boundary`, each with its visible line style. Do not collect the fill.
- A filled frequency silhouette is one `distribution_boundary` named
  `<property> | <line style> frequency distribution boundary`. Do not collect the
  fill or straight zero-frequency edge.
- A vertical average-and-spread object is one `point_group` named exactly
  `<property> Average Value + Spread of Value`; its average curve is separate.
  Python supplies the fixed `upper`, `average`, `lower` group configuration.
- A horizontal spread has no validated automatic rule; put it in `unresolved`.
- For categorical bars, create one Dataset per visible legend condition per numeric
  Axis. If a condition repeats across numeric Axes, name it
  `<condition> | <numeric property and unit>`; omit shared category titles/lists.
- Arrowed markers stay in their marker Dataset. Never create a `runout` Dataset.
- Exclude axes, ticks, borders, gridlines, text, arrows, hatching, dimensions,
  specimen drawings, fills and legend-only samples.

Return one final name, never alternatives. Preserve a visible source inconsistency.
If ownership, occurrence or wording cannot be determined from the image, report the
specific ambiguity in `unresolved` instead of guessing.

Before returning JSON, silently audit every proposed Dataset: its `axis` is one
of the supplied IDs; its `kind` matches the visible object; every applicable
condition is copied in full; every ordinary multi-part name uses the literal
` | ` separator; and the object contains exactly `axis`, `name` and `kind`.
