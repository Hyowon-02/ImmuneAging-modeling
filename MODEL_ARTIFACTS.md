# Model artifacts to release

For each NK/CD4T/CD8T clock provide the exact final estimator, its StandardScaler,
ordered feature list including Ensembl IDs and symbols, intercept and coefficients,
normalization/pseudocell parameters, software versions and random seeds.
Record SHA-256 hashes and the code commit used to generate them. Joblib/pickle
requires compatible versions; load only trusted model files. Coefficient CSV plus
scaler means/scales offers a portable linear-model specification.

Include donor split identifiers only where data permissions allow; otherwise supply
a reproducible controlled-access manifest procedure. Never include individual ages
or clinical predictions in public assets by default.

Model binaries are ignored by .gitignore. Store approved artifacts as release assets
or in a DOI archive and add their real URLs and checksums here after validation.
No pretrained model is included in this package.
