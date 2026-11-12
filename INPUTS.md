# Inputs and generated files

All paths are relative to the repository root. No patient-level data are bundled.

| Input | Required content | Used by |
|---|---|---|
| workflows/raw_data/AIDA_Freeze_v2.h5ad | AIDA v2 object with original annotations and raw UMI counts in `.raw` | prepare-aida |
| workflows/raw_data/Batch_info.xlsx | DCP_ID and scRNA-seq Experimental Batch | prepare-aida |
| workflows/raw_data/AIDA_pp.h5ad | Prepared AIDA; obs donor_id, age, sex, Country, Modeling_annotation; raw_counts layer | split, pseudobulk, resampling |
| workflows/raw_data/gene_id_mapping.csv | Mapping columns expected by our-model predictor; inspect its header handling | predict |
| workflows/raw_data/gene_id_mapping_full.csv | ensembl_id, gene_symbol | predict-li |
| workflows/raw_data/li_reference/model_coefficients.csv | gene or gene_clean, coef, celltype; one (Intercept) per cell type; lambda.min coefficients | predict-li |
| workflows/6.healthy_validation/Korean_COVID_validation/data/covid_pp_annotation.h5ad | Raw counts in X; obs sampleID, state, age, sex, Modeling_annotation; all genes before normalization | external-pseudocells, predict-li |
| workflows/4.prediction_model/Modeling_results/saved_models/ | Trained estimators, scalers, ordered feature JSON as produced by train | predict |

Outputs are written beside each workflow under data/, Modeling_results/,
validation_results/, or figures/. The scripts preserve these relative relationships.
Use original authorized data sources (AIDA/CODA) to obtain the input objects.
External QC/annotation is an input prerequisite, not reproduced by this compact repository.
Do not replace raw external X with normalized data before `external-pseudocells`:
our model aggregates counts first, whereas Li averages transformed individual cells.

The manuscript healthy comparison expects 57 shared donors. Li may produce an
extra donor because its small-cell-count rule differs; the comparison uses only
the intersection and saves donor_membership.csv. Verify inclusion IDs before reporting N.
