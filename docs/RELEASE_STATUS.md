# Release status

This is a curated code package, not a final validated release.

Outstanding evidence needed before using it as the final paper repository:

1. Training-only AAG selection and updated internal validation, or explicit labeling
   of the historical information leakage. Current scripts preserve the historical workflow.
2. Frozen final estimators, scalers, ordered feature lists, split manifest and coefficients.
3. Exact original dependency lockfiles (Python pip freeze / R sessionInfo).
4. Full execution against the actual input datasets and comparison with manuscript metrics.
5. Confirm gene mapping coverage, Li coefficient provenance and shared healthy-donor IDs.
6. Decide and document the external scaling policy; the current default uses the validation cohort.
7. Author-approved LICENSE, citation metadata and immutable release/DOI.

The user reports fixing COVID first-timepoint selection in later disease analyses.
Those updated disease scripts were not part of this compact core extraction; older
Integration_plot code is deliberately not included as if it were the corrected final version.
Raw sequencing data, patient-level predictions and private server paths are not release assets.
