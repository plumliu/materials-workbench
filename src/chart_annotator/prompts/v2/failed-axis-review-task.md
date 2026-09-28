You are correcting axis grounding after deterministic local image processing failed.

The review sheet contains the original full chart and enlarged views for ONLY the failed Axes. Orange X marks are your previous normalized grounding positions. Blue rings are locally detected T intersections. Purple diamonds are label-to-spine projections. A direction may have orange marks only when Python could not find the axis near them.

Return one grouped `Grounding` object that preserves exactly the coordinate groups and ordered Y mappings in `structure`. Each group contains one X anchor list and one anchor list per Y. Never repeat X under the Ys and never invent IDs.

Rules:

1. Correct only numeric directions that Python failed to fit. `review_context.successful_fit_directions` lists directions that already fitted successfully; copy their anchors exactly from `grounding` even if you would choose different points.
2. Use the full-chart overview to locate the true axis spine or shared coordinate boundary. Do not infer a spine from the center of a cropped label or from the location of an axis title.
3. Each corrected direction needs 3 or 4 reliable visible tick anchors when available, and never more than 4. Every anchor must include the exact visible label, its numeric value, and a point at the tick/spine intersection.
4. Coordinates are normalized 0–1000 in the ORIGINAL chart shown in `review_context.original_image_size`, never in the contact sheet or an enlarged crop.
5. A shared boundary may carry different printed values for adjacent Y mappings. Keep the value that belongs to the current Axis while using the same geometric intersection when appropriate. Do not use a label center as the intersection.
6. When a labeled tick has no perpendicular tick stroke, place the point by projecting the label's value position onto the confirmed axis spine. Preserve linear or logarithmic spacing across all anchors.
7. Do not change group order, Y order, scales, labels, or Dataset information. Do not add coordinate mappings.
8. Return your best concrete correction. `unresolved` must be empty; Python will decide whether the corrected anchors are geometrically sufficient.

Output strict JSON only, matching the supplied `Grounding` schema.
