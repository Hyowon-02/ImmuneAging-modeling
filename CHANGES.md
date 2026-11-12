# Curation changes

- Extracted modeling stages into Python scripts, retaining source cell numbers.
- Removed all notebook outputs and display-only/duplicate training cells.
- Kept donor-level-CV cell 1 from 3.Modeling; omitted obsolete cell 0.
- Enabled NK, CD4T and CD8T training; original saved cell enabled CD8T only.
- Set resampling to 100; original saved source specified 10 despite a 100-run output name.
- Added split donor-list/metadata exports and missing output-directory creation.
- Converted recognized local server roots to repository-relative paths.
- Included latest session Li gene mapping and one-time raw-count normalization.
- Included MAE-only Holm-3 comparison with original boxplot colors and larger stars.
- No fitted coefficients were regenerated. These edits do not certify that current
  scripts produce the manuscript numbers; release validation remains necessary.
