# ImmuneAging modeling

Core modeling code for the immune-aging manuscript: AAG input construction,
donor splitting, pseudocells, donor-level cross-validation, resampling,
external prediction and comparison with Li et al.

**Release preparation snapshot, not a validated final publication release.**
Code was curated from the supplied analysis notebooks. Real cohort data and fitted
models are not bundled, and end-to-end numerical reproduction has not been run.
Read [release status](docs/RELEASE_STATUS.md) before using performance results.

## Included workflows

| Stage | Purpose |
|---|---|
| prepare-aida | AIDA donor/cell-type annotation, count layer and gene filters |
| pseudobulk / select-aags | Donor pseudobulk; DESeq2 spline LRT and amplitude filter |
| split | Stratified donor split and saved donor lists |
| size-sensitivity | Training-donor pseudocell-size comparison |
| pseudocells | 20-cell count sums, CP10K/log1p, older-donor oversampling |
| train | Donor-level CV and model/scaler/feature exports |
| resampling | 100 repeats of splitting and ElasticNet fitting |
| composition-baseline | Cell-proportion comparison model |
| external-pseudocells / predict | External pseudocell generation and our-model prediction |
| predict-li | Mapping-aware Li coefficient inference on healthy controls |
| compare | Paired healthy-donor absolute-error plot and MAE tests |

Routine exploratory plots, obsolete training cells, sccomp, enrichment plots and
disease-specific duplicate notebooks are omitted. Module interpretation/WGCNA is
outside this compact modeling release; it is needed separately to reproduce Fig. 6.

## Setup and execution

Use Python 3.10 or a compatible environment verified against the original analysis.
`requirements.txt` contains manuscript-reported pins plus unpinned dependencies;
it is not an exact environment lock. Install the R dependencies only for AAG selection.

```bash
python -m pip install -r requirements.txt
python run.py --help
```

Stage inputs and their locations are listed in [INPUTS.md](docs/INPUTS.md).
Each stage runs in its own workflow directory. Configure constants at the top of
the corresponding script where needed; `run.py` handles working directories.

Historical analysis order (retained for provenance, with the leakage caveat below):

```bash
python run.py prepare-aida
python run.py pseudobulk
python run.py select-aags
python run.py split
python run.py pseudocells
python run.py train
```

`size-sensitivity`, `resampling`, and `composition-baseline` are separate analyses.
With externally prepared raw-count data and trained models:

```bash
python run.py external-pseudocells
python run.py predict
python run.py predict-li
python run.py compare
```

## Essential interpretation notes

The historical AAG list was selected on the full AIDA cohort before splitting.
Thus its internal test metrics are not unbiased held-out estimates. Packaging does
not repair this: final validation requires training-only feature selection and
updated models/results, with feature selection inside CV if reporting independent
CV performance. Resampling the fixed full-cohort gene list does not resolve leakage.

The supplied external predictor defaults to cohort-specific scaling
(`SCALER_MODE="validation"`), not the fixed training scaler. This is preserved
and must be reported. The optional train-scaler branch exists, but changing it
changes the analysis and requires regenerated results.

The Li workflow uses gene mapping, CP10K/log1p, 15-cell means with dynamic sampling,
and donor medians (paper Methods). The package round(mean) result is also exported.
Its coefficients must have been exported at `s="lambda.min"`; they are not included.
The paired comparison restricts to Normal controls and the shared unique donors.
Only MAE is tested across NK/CD4T/CD8T, with three-test Holm correction. Bootstrap
intervals are pointwise, and the permutation test assumes within-donor exchangeability.

## Provenance and artifacts

[SOURCE_MANIFEST.json](docs/SOURCE_MANIFEST.json) records original notebook cells
and source hashes. [CHANGES.md](docs/CHANGES.md) records curation changes.
Upload trained models, scalers and ordered features with an immutable release/DOI;
see [MODEL_ARTIFACTS.md](docs/MODEL_ARTIFACTS.md).

Before public release, choose an author-approved code license, confirm author/citation
metadata and create a versioned archive. No license grant or DOI is invented here.
Li reference: https://doi.org/10.1038/s43587-025-00819-z
Reference implementation: https://github.com/CiiM-Bioinformatics-group/scImmuAging
