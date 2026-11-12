# Curated publication source. See docs/SOURCE_MANIFEST.json.

# Source notebook cell 1
# ==============================================================================
# 1. Imports / configuration
# ==============================================================================

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


# ==============================================================================
# Paths
# ==============================================================================
# Current working directory:
# ~/workspace/Immune_aging/4.prediction_model

AIDA_PATH = Path("../raw_data/AIDA_pp.h5ad")
TRAIN_TEST_DIR = Path("data/train_test")

# 모든 cell type은 donor를 한 번 나눈 뒤 cell type별로 subset한 것이므로
# donor 명단은 NK train/test 하나만 reference로 사용
REFERENCE_CELLTYPE = "NK"

OUT_DIR = Path("Modeling_results/cell_proportion_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# AnnData columns
# ==============================================================================

DONOR_COL = "donor_id"
AGE_COL = "age"

MAJOR_ANN_COL = "Annotation_Level2"
SUBTYPE_ANN_COL = "Annotation_Level3"


# ==============================================================================
# Analysis cell types / subtypes
# 2_cellsubtype_proportion(1).ipynb와 동일한 annotation
# ==============================================================================

CELL_GROUPS = OrderedDict(
    {
        "NK": [
            "CD16+_NK",
            "CD56+_NK",
        ],

        "Mono": [
            "CD14+_Monocyte",
            "CD16+_Monocyte",
        ],

        "CD4T": [
            "CD4+_T_naive",
            "CD4+_T_cm",
            "CD4+_T_em",
            "CD4+_T_cyt",
            "Treg",
        ],

        "CD8T": [
            "CD8+_T_naive",
            "CD8+_T_GZMKhi",
            "CD8+_T_GZMBhi",
            "MAIT",
        ],
    }
)

CELL_TYPES = list(CELL_GROUPS.keys())

# 기존 subtype proportion 분석과 동일
MIN_CELLS_PER_MAJOR = 10


# ==============================================================================
# ElasticNet
# ==============================================================================

RANDOM_STATE = 251219
CV_FOLDS = 5

# 현재 pseudo-cell training에서
# <50세: 100 pseudocells / >=50세: 200 pseudocells였던 상대적 비중을 맞춤.
# 필요 없으면 False로 바꾸면 됨.
USE_AGE_WEIGHTING = True
OLDER_AGE_CUTOFF = 50
YOUNG_WEIGHT = 1.0
OLDER_WEIGHT = 2.0


print("Working directory :", Path.cwd())
print("AIDA              :", AIDA_PATH.resolve())
print("Train/Test        :", TRAIN_TEST_DIR.resolve())
print("Output            :", OUT_DIR.resolve())

# Source notebook cell 3
# ==============================================================================
# 2. Read reference donor split
# ==============================================================================

train_path = TRAIN_TEST_DIR / f"{REFERENCE_CELLTYPE}_train.h5ad"
test_path = TRAIN_TEST_DIR / f"{REFERENCE_CELLTYPE}_test.h5ad"

if not train_path.exists():
    raise FileNotFoundError(f"Missing: {train_path}")

if not test_path.exists():
    raise FileNotFoundError(f"Missing: {test_path}")


# X는 필요 없으므로 backed mode로 donor_id만 읽음
ad_train_ref = sc.read_h5ad(train_path, backed="r")
ad_test_ref = sc.read_h5ad(test_path, backed="r")

if DONOR_COL not in ad_train_ref.obs.columns:
    raise KeyError(f"'{DONOR_COL}' not found in {train_path}")

if DONOR_COL not in ad_test_ref.obs.columns:
    raise KeyError(f"'{DONOR_COL}' not found in {test_path}")


train_donors = pd.Index(
    ad_train_ref.obs[DONOR_COL]
    .dropna()
    .astype(str)
    .unique()
)

test_donors = pd.Index(
    ad_test_ref.obs[DONOR_COL]
    .dropna()
    .astype(str)
    .unique()
)

ad_train_ref.file.close()
ad_test_ref.file.close()


overlap = train_donors.intersection(test_donors)

if len(overlap) > 0:
    raise ValueError(
        "Train/Test donor overlap detected:\n"
        + "\n".join(overlap[:20].astype(str))
    )


print(f"Reference cell type : {REFERENCE_CELLTYPE}")
print(f"Train donors        : {len(train_donors):,}")
print(f"Test donors         : {len(test_donors):,}")
print(f"Overlap             : {len(overlap)}")

# Source notebook cell 5
# ==============================================================================
# 3. Load AIDA_pp metadata / annotations
# ==============================================================================

if not AIDA_PATH.exists():
    raise FileNotFoundError(f"Missing: {AIDA_PATH}")


# Expression matrix는 필요 없으므로 backed mode
adata = sc.read_h5ad(AIDA_PATH, backed="r")

required_cols = [
    DONOR_COL,
    AGE_COL,
    MAJOR_ANN_COL,
    SUBTYPE_ANN_COL,
]

missing_cols = [
    col
    for col in required_cols
    if col not in adata.obs.columns
]

if missing_cols:
    adata.file.close()
    raise KeyError(
        "AIDA_pp.h5ad is missing required obs columns: "
        + ", ".join(missing_cols)
    )


obs = adata.obs[required_cols].copy()

adata.file.close()
del adata


# ------------------------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------------------------

obs = obs.dropna(
    subset=[
        DONOR_COL,
        AGE_COL,
        MAJOR_ANN_COL,
    ]
).copy()

obs[DONOR_COL] = obs[DONOR_COL].astype(str)

obs[AGE_COL] = pd.to_numeric(
    obs[AGE_COL],
    errors="coerce",
)

obs = obs.dropna(
    subset=[AGE_COL]
).copy()


# ------------------------------------------------------------------------------
# donor마다 age가 하나인지 확인
# ------------------------------------------------------------------------------

age_nunique = (
    obs.groupby(
        DONOR_COL,
        observed=True,
    )[AGE_COL]
    .nunique(dropna=False)
)

inconsistent_age = age_nunique[
    age_nunique > 1
]

if not inconsistent_age.empty:
    raise ValueError(
        "Some donors have inconsistent age values:\n"
        + inconsistent_age.head(20).to_string()
    )


donor_metadata = (
    obs[
        [
            DONOR_COL,
            AGE_COL,
        ]
    ]
    .drop_duplicates(
        subset=DONOR_COL
    )
    .set_index(DONOR_COL)
)


# ==============================================================================
# Annotation_Level2 -> Major_Group
# 2_cellsubtype_proportion(1).ipynb와 동일
# ==============================================================================

level2 = (
    obs[MAJOR_ANN_COL]
    .astype("string")
)

obs["Major_Group"] = pd.Series(
    pd.NA,
    index=obs.index,
    dtype="object",
)


# CD4T
obs.loc[
    level2.str.contains(
        "CD4",
        case=False,
        na=False,
    ),
    "Major_Group",
] = "CD4T"


# CD8T
obs.loc[
    level2.str.contains(
        "CD8",
        case=False,
        na=False,
    ),
    "Major_Group",
] = "CD8T"


# NK
obs.loc[
    level2.str.contains(
        "NK",
        case=False,
        na=False,
    ),
    "Major_Group",
] = "NK"


# Mono
obs.loc[
    level2.str.contains(
        "Mono",
        case=False,
        na=False,
    ),
    "Major_Group",
] = "Mono"


print("\nMajor group mapping")
print("=" * 60)

print(
    pd.crosstab(
        obs[MAJOR_ANN_COL],
        obs["Major_Group"],
        dropna=False,
    ).to_string()
)


# 현재 model에서 쓰는 4개 major group만 유지
obs = obs[
    obs["Major_Group"].isin(
        CELL_GROUPS.keys()
    )
].copy()


# ==============================================================================
# Annotation_Level3 -> Analysis_Subtype
# ==============================================================================

obs[SUBTYPE_ANN_COL] = (
    obs[SUBTYPE_ANN_COL]
    .astype("object")
    .where(
        obs[SUBTYPE_ANN_COL].notna(),
        "Missing_Level3",
    )
    .astype(str)
)

obs["Analysis_Subtype"] = (
    obs[SUBTYPE_ANN_COL]
    .copy()
)


print(f"\nCells retained : {len(obs):,}")
print(f"Donors retained: {obs[DONOR_COL].nunique():,}")

# Source notebook cell 7
# ==============================================================================
# 4. Calculate donor-level within-major subtype proportions
# ==============================================================================

composition_dict = {}


for major_group, target_subtypes in CELL_GROUPS.items():

    print(
        f"\n{'=' * 70}\n"
        f"Processing major group: {major_group}\n"
        f"{'=' * 70}"
    )


    major_obs = obs[
        obs["Major_Group"] == major_group
    ].copy()


    if major_obs.empty:
        print(f"No cells were found for {major_group}; skip.")
        continue


    # ==========================================================================
    # Denominator
    # donor별 해당 major lineage의 모든 cell
    # ==========================================================================

    denominator = (
        major_obs[DONOR_COL]
        .value_counts()
        .sort_index()
    )

    denominator.name = "n_major_cells"


    valid_donors = (
        denominator[
            denominator >= MIN_CELLS_PER_MAJOR
        ]
        .index
    )

    excluded_donors = (
        denominator[
            denominator < MIN_CELLS_PER_MAJOR
        ]
        .index
    )


    print(
        f"Donors with {major_group}: "
        f"{len(denominator):,}"
    )

    print(
        f"Donors retained with >= {MIN_CELLS_PER_MAJOR} "
        f"{major_group} cells: {len(valid_donors):,}"
    )

    print(
        f"Excluded donors: {len(excluded_donors):,}"
    )


    major_obs = major_obs[
        major_obs[DONOR_COL].isin(
            valid_donors
        )
    ].copy()

    denominator = denominator.loc[
        valid_donors
    ]


    # ==========================================================================
    # Numerator = donor x target subtype
    # ==========================================================================

    subtype_counts = pd.crosstab(
        major_obs[DONOR_COL],
        major_obs["Analysis_Subtype"],
    )

    subtype_counts = subtype_counts.reindex(
        index=valid_donors,
        columns=target_subtypes,
        fill_value=0,
    )


    # ==========================================================================
    # Proportion (%) = target subtype cells / all major cells * 100
    # ==========================================================================

    subtype_percentage = (
        subtype_counts
        .div(
            denominator,
            axis=0,
        )
        * 100
    )


    # ==========================================================================
    # QC: target에 포함되지 않은 같은-lineage cells
    # ==========================================================================

    subtype_percentage["Other_within_major"] = (
        100
        - subtype_percentage[
            target_subtypes
        ].sum(axis=1)
    )

    subtype_percentage["Other_within_major"] = (
        subtype_percentage["Other_within_major"]
        .mask(
            subtype_percentage["Other_within_major"].abs() < 1e-10,
            0,
        )
    )


    # ==========================================================================
    # Add age / cell count
    # ==========================================================================

    comp_df = (
        subtype_percentage
        .join(
            denominator,
            how="left",
        )
        .join(
            donor_metadata,
            how="left",
        )
    )

    comp_df.index.name = DONOR_COL


    # ==========================================================================
    # QC
    # ==========================================================================

    row_sum = (
        comp_df[
            target_subtypes
            + ["Other_within_major"]
        ]
        .sum(axis=1)
    )


    print(
        "Within-major row-sum check: "
        f"min={row_sum.min():.6f}, "
        f"median={row_sum.median():.6f}, "
        f"max={row_sum.max():.6f}"
    )

    print(
        "Other_within_major: "
        f"median={comp_df['Other_within_major'].median():.2f}%"
    )


    composition_dict[
        major_group
    ] = comp_df

# Source notebook cell 9
# ==============================================================================
# 5. ElasticNet modeling
# ==============================================================================

def evaluate_regression(
    y_true,
    y_pred,
):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    pearson_r, pearson_p = pearsonr(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Pearson_R": pearson_r,
        "Pearson_P": pearson_p,
    }


results = []
prediction_list = []


for cell_type in CELL_TYPES:

    print(
        f"\n{'=' * 70}\n"
        f"ElasticNet: {cell_type}\n"
        f"{'=' * 70}"
    )


    comp_df = composition_dict[
        cell_type
    ].copy()

    feature_cols = CELL_GROUPS[
        cell_type
    ]


    # ==========================================================================
    # Apply the already-defined donor split
    # ==========================================================================

    available_train = train_donors[
        train_donors.isin(
            comp_df.index
        )
    ]

    available_test = test_donors[
        test_donors.isin(
            comp_df.index
        )
    ]


    missing_train = train_donors[
        ~train_donors.isin(
            comp_df.index
        )
    ]

    missing_test = test_donors[
        ~test_donors.isin(
            comp_df.index
        )
    ]


    print(
        f"Train donors: {len(available_train):,} "
        f"(missing after >= {MIN_CELLS_PER_MAJOR} cell filter: "
        f"{len(missing_train):,})"
    )

    print(
        f"Test donors : {len(available_test):,} "
        f"(missing after >= {MIN_CELLS_PER_MAJOR} cell filter: "
        f"{len(missing_test):,})"
    )


    train_df = comp_df.loc[
        available_train
    ].copy()

    test_df = comp_df.loc[
        available_test
    ].copy()


    # ==========================================================================
    # X / y
    # ==========================================================================

    X_train = (
        train_df[
            feature_cols
        ]
        .astype(float)
    )

    X_test = (
        test_df[
            feature_cols
        ]
        .astype(float)
    )

    y_train = (
        train_df[AGE_COL]
        .astype(float)
        .to_numpy()
    )

    y_test = (
        test_df[AGE_COL]
        .astype(float)
        .to_numpy()
    )


    print(f"Features: {feature_cols}")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")


    # ==========================================================================
    # StandardScaler: fit on train only
    # ==========================================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ==========================================================================
    # Optional age weighting
    # ==========================================================================

    if USE_AGE_WEIGHTING:

        sample_weight = np.where(
            y_train >= OLDER_AGE_CUTOFF,
            OLDER_WEIGHT,
            YOUNG_WEIGHT,
        )

        print(
            f"Age weighting: "
            f"<{OLDER_AGE_CUTOFF}={YOUNG_WEIGHT}, "
            f">={OLDER_AGE_CUTOFF}={OLDER_WEIGHT}"
        )

    else:

        sample_weight = None

        print("Age weighting: OFF")


    # ==========================================================================
    # ElasticNet only
    # ==========================================================================

    model = ElasticNetCV(
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=1,
        max_iter=10000,
        precompute=False,
    )

    model.fit(
        X_train_scaled,
        y_train,
        sample_weight=sample_weight,
    )


    # ==========================================================================
    # Fixed test set: evaluate once
    # ==========================================================================

    y_pred = model.predict(
        X_test_scaled
    )

    metrics = evaluate_regression(
        y_test,
        y_pred,
    )


    print(
        f"MAE={metrics['MAE']:.3f} | "
        f"RMSE={metrics['RMSE']:.3f} | "
        f"R2={metrics['R2']:.3f} | "
        f"Pearson R={metrics['Pearson_R']:.3f}"
    )

    print(
        f"alpha={model.alpha_:.6g} | "
        f"l1_ratio={model.l1_ratio_}"
    )


    results.append(
        {
            "Cell_Type": cell_type,
            "N_Train_Donors": len(train_df),
            "N_Test_Donors": len(test_df),
            "N_Features": len(feature_cols),
            "Features": ";".join(feature_cols),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "Pearson_R": metrics["Pearson_R"],
            "Pearson_P": metrics["Pearson_P"],
            "Alpha": model.alpha_,
            "L1_Ratio": model.l1_ratio_,
        }
    )


    pred_df = pd.DataFrame(
        {
            DONOR_COL: test_df.index.astype(str),
            "Cell_Type": cell_type,
            "true_age": y_test,
            "pred_age": y_pred,
        }
    )

    pred_df["residual"] = (
        pred_df["pred_age"]
        - pred_df["true_age"]
    )

    pred_df["abs_error"] = (
        pred_df["residual"].abs()
    )

    prediction_list.append(
        pred_df
    )

# Source notebook cell 11
# ==============================================================================
# 6. Summary / save
# ==============================================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.concat(
    prediction_list,
    ignore_index=True,
)


display_cols = [
    "Cell_Type",
    "N_Train_Donors",
    "N_Test_Donors",
    "N_Features",
    "MAE",
    "RMSE",
    "R2",
    "Pearson_R",
]

display(
    results_df[
        display_cols
    ]
)


metrics_path = (
    OUT_DIR
    / "cell_proportion_ElasticNet_test_metrics.csv"
)

predictions_path = (
    OUT_DIR
    / "cell_proportion_ElasticNet_test_predictions.csv"
)


results_df.to_csv(
    metrics_path,
    index=False,
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


print(f"\nSaved: {metrics_path.resolve()}")
print(f"Saved: {predictions_path.resolve()}")
