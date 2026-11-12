# Curated publication source. See docs/SOURCE_MANIFEST.json.

# Source notebook cell 0
import os
import re
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import lightgbm as lgb

from tqdm.auto import tqdm
from scipy import sparse
from scipy.stats import pearsonr

from sklearn.base import clone
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ==============================================================
# 0. Config
# ==============================================================

RANDOM_SEED = 260601
np.random.seed(RANDOM_SEED)

CELLTYPES = ["NK", "CD8T", "CD4T"]

INPUT_RDS_DIR = "data/train_test"
DEG_DIR = "./../2.DEG_analysis/data/DEG_analysis"

SAVE_DIR = "pseudocell_size_test"
DONOR_PRED_DIR = os.path.join(SAVE_DIR, "donor_level_predictions")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DONOR_PRED_DIR, exist_ok=True)

LAYER_NAME = "raw_counts"

COL_DONOR = "donor_id"
COL_AGE = "age"
COL_SEX = "sex"
COL_CELLTYPE = "Modeling_annotation"

MIN_CELLS_PER_DONOR = 30

PSEUDOCELL_SIZES = [
    1, 5, 10, 20, 35, 50,
    100, 150, 250, 500, 750, "ALL"
]

# 기존 oversampling 방식 유지
N_PSEUDO_YOUNG_TRAIN = 100
N_PSEUDO_OLD_TRAIN = 200
AGE_OVERSAMPLE_CUTOFF = 50

N_CV = 5
MAX_ITER = 10000


# --------------------------------------------------------------
# Pseudocell-size test에서는 hyperparameter tuning을 하지 않음
# --------------------------------------------------------------

FIXED_LASSO_ALPHA = 0.01

FIXED_ELASTICNET_ALPHA = 0.01
FIXED_ELASTICNET_L1_RATIO = 0.5

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None

LGBM_N_ESTIMATORS = 100
LGBM_LEARNING_RATE = 0.1
LGBM_NUM_LEAVES = 31


# ==============================================================
# 1. Basic helper functions
# ==============================================================

def get_layer_matrix(adata, layer_name=LAYER_NAME):
    if layer_name == "X" or layer_name is None:
        return adata.X

    if layer_name in adata.layers:
        return adata.layers[layer_name]

    raise ValueError(
        f"Layer '{layer_name}' not found in adata."
    )


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if (
        len(y_pred) < 2
        or np.std(y_pred) < 1e-9
        or np.std(y_true) < 1e-9
    ):
        pr = np.nan
    else:
        pr, _ = pearsonr(y_true, y_pred)

    return mae, rmse, r2, pr


# ==============================================================
# 2. DEG feature loading
# ==============================================================

def load_deg_features(ct, ad_tr, ad_te, deg_dir):
    deg_path = os.path.join(
        deg_dir,
        f"significant_DEGs_{ct}.csv"
    )

    if not os.path.exists(deg_path):
        raise FileNotFoundError(
            f"DEG file not found: {deg_path}"
        )

    deg_df = pd.read_csv(deg_path)

    if "gene" not in deg_df.columns:
        raise ValueError(
            f"'gene' column not found in {deg_path}"
        )

    if "padj" in deg_df.columns:
        deg_df = deg_df.sort_values(
            "padj",
            ascending=True
        )

    deg_genes = (
        deg_df["gene"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    train_genes = set(ad_tr.var_names)
    test_genes = set(ad_te.var_names)

    genes = [
        g for g in deg_genes
        if g in train_genes
        and g in test_genes
    ]

    return genes, deg_df, deg_path


# ==============================================================
# 3. Pseudocell generation
# ==============================================================

def get_feature_indices(adata, genes):
    var_index = pd.Index(adata.var_names)
    return var_index.get_indexer(genes)


def sum_counts_for_indices(X, row_idx, feature_idx):
    X_sub = X[row_idx, :]

    if sparse.issparse(X_sub):
        total_sum = float(X_sub.sum())
        feature_sum = np.asarray(
            X_sub[:, feature_idx].sum(axis=0)
        ).ravel()
    else:
        total_sum = float(X_sub.sum())
        feature_sum = np.asarray(
            X_sub[:, feature_idx].sum(axis=0)
        ).ravel()

    return feature_sum, total_sum


def make_pseudocell_matrix_for_size(
    adata,
    cell_type_value,
    split_name,
    genes,
    pseudocell_size,
    seed=RANDOM_SEED,
    cell_type_col=COL_CELLTYPE,
    donor_col=COL_DONOR,
    age_col=COL_AGE,
    sex_col=COL_SEX,
    layer_name=LAYER_NAME,
):
    rng = np.random.default_rng(seed)

    if cell_type_col not in adata.obs.columns:
        raise ValueError(
            f"{cell_type_col} not found in adata.obs"
        )

    mask = (
        adata.obs[cell_type_col].values
        == cell_type_value
    )

    if mask.sum() == 0:
        return None, None

    subset_obs = adata.obs.loc[mask].copy()
    target_indices = np.where(mask)[0]

    X = get_layer_matrix(
        adata,
        layer_name=layer_name
    )

    feature_idx = get_feature_indices(
        adata,
        genes
    )

    if np.any(feature_idx < 0):
        missing = [
            g for g, idx
            in zip(genes, feature_idx)
            if idx < 0
        ]

        raise ValueError(
            f"Missing genes in adata.var_names: "
            f"{missing[:10]}"
        )

    rows = []
    metas = []

    donors = (
        subset_obs[donor_col]
        .dropna()
        .unique()
    )

    for donor in donors:
        donor_mask = (
            subset_obs[donor_col].values
            == donor
        )

        n_real = int(
            donor_mask.sum()
        )

        if n_real < MIN_CELLS_PER_DONOR:
            continue

        donor_global_indices = (
            target_indices[donor_mask]
        )

        donor_age = float(
            subset_obs.loc[
                subset_obs[donor_col] == donor,
                age_col
            ].iloc[0]
        )

        if sex_col in subset_obs.columns:
            donor_sex = (
                subset_obs.loc[
                    subset_obs[donor_col] == donor,
                    sex_col
                ].iloc[0]
            )
        else:
            donor_sex = "Unknown"

        # ------------------------------------------------------
        # 생성할 pseudocell 수
        # ------------------------------------------------------

        if pseudocell_size == "ALL":
            current_n_pseudo = 1

        elif (
            split_name == "train"
            and donor_age >= AGE_OVERSAMPLE_CUTOFF
        ):
            current_n_pseudo = N_PSEUDO_OLD_TRAIN

        else:
            current_n_pseudo = N_PSEUDO_YOUNG_TRAIN

        # ------------------------------------------------------
        # Pseudocell 생성
        # ------------------------------------------------------

        for i in range(current_n_pseudo):
            if pseudocell_size == "ALL":
                sampled_idx = donor_global_indices
                k_used = n_real

            else:
                sampled_idx = rng.choice(
                    donor_global_indices,
                    size=int(pseudocell_size),
                    replace=True
                )
                k_used = int(pseudocell_size)

            feature_sum, total_sum = sum_counts_for_indices(
                X=X,
                row_idx=sampled_idx,
                feature_idx=feature_idx
            )

            # CP10K + log1p
            if total_sum <= 0:
                norm_feature = np.zeros_like(
                    feature_sum,
                    dtype=np.float64
                )
            else:
                norm_feature = np.log1p(
                    (feature_sum / total_sum) * 1e4
                )

            rows.append(norm_feature)

            metas.append({
                "pseudocell_id":
                    f"{cell_type_value}_"
                    f"{split_name}_"
                    f"{donor}_"
                    f"K{pseudocell_size}_"
                    f"pc{i+1}",

                "donor_id": donor,
                "cell_type": cell_type_value,
                "age": donor_age,
                "sex": donor_sex,
                "split": split_name,
                "n_real_cells": n_real,
                "pseudocell_size": pseudocell_size,
                "k_used": k_used
            })

    if len(rows) == 0:
        return None, None

    X_df = pd.DataFrame(
        np.vstack(rows),
        columns=genes,
        index=[
            m["pseudocell_id"]
            for m in metas
        ]
    )

    meta_df = (
        pd.DataFrame(metas)
        .set_index("pseudocell_id")
    )

    return X_df, meta_df


# ==============================================================
# 4. Donor-level 5-fold CV
# ==============================================================

def make_donor_cv_splits(
    meta_train,
    n_splits=N_CV,
    seed=RANDOM_SEED
):
    donor_ids = (
        meta_train["donor_id"]
        .astype(str)
        .values
    )

    unique_donors = np.array(
        sorted(
            pd.unique(donor_ids)
        )
    )

    if len(unique_donors) < n_splits:
        raise ValueError(
            f"Only {len(unique_donors)} donors "
            f"available for {n_splits}-fold CV."
        )

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    cv_splits = []

    for fold_idx, (
        donor_train_idx,
        donor_valid_idx
    ) in enumerate(
        kf.split(unique_donors),
        start=1
    ):
        train_donors = set(
            unique_donors[
                donor_train_idx
            ]
        )

        valid_donors = set(
            unique_donors[
                donor_valid_idx
            ]
        )

        overlap = (
            train_donors
            & valid_donors
        )

        if overlap:
            raise RuntimeError(
                f"Donor leakage detected "
                f"in fold {fold_idx}"
            )

        train_rows = np.where(
            np.isin(
                donor_ids,
                list(train_donors)
            )
        )[0]

        valid_rows = np.where(
            np.isin(
                donor_ids,
                list(valid_donors)
            )
        )[0]

        cv_splits.append(
            (
                train_rows,
                valid_rows
            )
        )

    return cv_splits


# ==============================================================
# 5. Fixed models
#
# No hyperparameter tuning in pseudocell-size test
# ==============================================================

def get_fixed_models():
    return {
        "Lasso": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Lasso(
                    alpha=FIXED_LASSO_ALPHA,
                    max_iter=MAX_ITER
                )
            )
        ]),

        "ElasticNet": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                ElasticNet(
                    alpha=FIXED_ELASTICNET_ALPHA,
                    l1_ratio=FIXED_ELASTICNET_L1_RATIO,
                    max_iter=MAX_ITER,
                    random_state=RANDOM_SEED
                )
            )
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            n_jobs=-1,
            random_state=RANDOM_SEED
        ),

        "LightGBM": lgb.LGBMRegressor(
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=-1
        )
    }


# ==============================================================
# 6. Cross-validation
# ==============================================================

def cross_validate_model(
    model_name,
    model,
    X,
    meta,
    cv_splits
):
    y = (
        meta["age"]
        .astype(float)
        .values
    )

    # LightGBM-safe gene names
    if model_name == "LightGBM":
        X_model = X.rename(
            columns=lambda x:
                re.sub(
                    r"[^\w]",
                    "_",
                    str(x)
                )
        )
    else:
        X_model = X

    fold_predictions = []

    for fold_idx, (
        train_idx,
        valid_idx
    ) in enumerate(
        cv_splits,
        start=1
    ):
        X_train = X_model.iloc[
            train_idx
        ]

        X_valid = X_model.iloc[
            valid_idx
        ]

        y_train = y[
            train_idx
        ]

        y_valid = y[
            valid_idx
        ]

        fold_model = clone(
            model
        )

        fold_model.fit(
            X_train,
            y_train
        )

        y_pred = fold_model.predict(
            X_valid
        )

        meta_valid = (
            meta.iloc[
                valid_idx
            ]
            .copy()
        )

        fold_df = pd.DataFrame({
            "donor_id":
                meta_valid[
                    "donor_id"
                ].astype(str).values,

            "cell_type":
                meta_valid[
                    "cell_type"
                ].values,

            "true_age":
                y_valid,

            "pred_age":
                y_pred,

            "sex":
                meta_valid[
                    "sex"
                ].values,

            "n_real_cells":
                meta_valid[
                    "n_real_cells"
                ].values,

            "pseudocell_size":
                meta_valid[
                    "pseudocell_size"
                ].values,

            "k_used":
                meta_valid[
                    "k_used"
                ].values,

            "fold":
                fold_idx
        })

        fold_predictions.append(
            fold_df
        )

    # ----------------------------------------------------------
    # 모든 out-of-fold prediction 결합
    # ----------------------------------------------------------

    oof_df = pd.concat(
        fold_predictions,
        ignore_index=True
    )

    # ----------------------------------------------------------
    # Pseudocell-level metrics
    # ----------------------------------------------------------

    (
        pc_mae,
        pc_rmse,
        pc_r2,
        pc_pr
    ) = evaluate(
        oof_df["true_age"],
        oof_df["pred_age"]
    )

    # ----------------------------------------------------------
    # Donor-level prediction
    #
    # 최종 modeling 코드와 동일하게 prediction은 median
    # ----------------------------------------------------------

    donor_df = (
        oof_df
        .groupby("donor_id")
        .agg(
            Cell_Type=(
                "cell_type",
                "first"
            ),

            True_Age=(
                "true_age",
                "first"
            ),

            Predicted_Age=(
                "pred_age",
                "median"
            ),

            Predicted_Age_SD=(
                "pred_age",
                "std"
            ),

            N_Pseudocells=(
                "pred_age",
                "size"
            ),

            Sex=(
                "sex",
                "first"
            ),

            N_Real_Cells=(
                "n_real_cells",
                "first"
            ),

            Pseudocell_Size=(
                "pseudocell_size",
                "first"
            ),

            K_Used=(
                "k_used",
                "first"
            )
        )
        .reset_index()
    )

    donor_df[
        "Predicted_Age_SD"
    ] = (
        donor_df[
            "Predicted_Age_SD"
        ]
        .fillna(0)
    )

    (
        donor_mae,
        donor_rmse,
        donor_r2,
        donor_pr
    ) = evaluate(
        donor_df[
            "True_Age"
        ].values,

        donor_df[
            "Predicted_Age"
        ].values
    )

    metrics = {
        "Pseudocell_MAE": pc_mae,
        "Pseudocell_RMSE": pc_rmse,
        "Pseudocell_R2": pc_r2,
        "Pseudocell_Pearson_R": pc_pr,

        "Donor_MAE": donor_mae,
        "Donor_RMSE": donor_rmse,
        "Donor_R2": donor_r2,
        "Donor_Pearson_R": donor_pr,

        "N_Eval_Donors": donor_df.shape[0]
    }

    return metrics, donor_df, oof_df


# ==============================================================
# 7. Main pipeline
# ==============================================================

def main():
    all_results = []

    print(
        "=== Pseudocell size robustness test ==="
    )

    print(
        "Models: Lasso / ElasticNet / "
        "RandomForest / LightGBM"
    )

    print(
        "Evaluation: donor-level "
        "5-fold OOF cross-validation"
    )

    # ==========================================================
    # Cell type
    # ==========================================================

    for ct in tqdm(
        CELLTYPES,
        desc="Cell types"
    ):
        print(
            f"\n\n📌 Cell type: {ct}"
        )

        tr_path = os.path.join(
            INPUT_RDS_DIR,
            f"{ct}_train.h5ad"
        )

        te_path = os.path.join(
            INPUT_RDS_DIR,
            f"{ct}_test.h5ad"
        )

        if (
            not os.path.exists(tr_path)
            or not os.path.exists(te_path)
        ):
            print(
                f"   [Skip] Missing "
                f"train/test h5ad for {ct}"
            )
            continue

        ad_tr = sc.read_h5ad(
            tr_path
        )

        # Test data는 gene availability 확인에만 사용
        # size 선택 성능 평가에는 사용하지 않음
        ad_te = sc.read_h5ad(
            te_path
        )

        # ======================================================
        # DEG features
        # ======================================================

        try:
            genes, _, deg_path = (
                load_deg_features(
                    ct,
                    ad_tr,
                    ad_te,
                    DEG_DIR
                )
            )

        except Exception as e:
            print(
                f"   [Skip] DEG loading failed: "
                f"{e}"
            )
            continue

        if len(genes) == 0:
            print(
                f"   [Skip] No DEG features for {ct}"
            )
            continue

        print(
            f"   DEG features: {len(genes)}"
        )

        print(
            f"   DEG file: {deg_path}"
        )

        models = get_fixed_models()

        # ======================================================
        # Pseudocell sizes
        # ======================================================

        for k in tqdm(
            PSEUDOCELL_SIZES,
            desc=f"{ct} pseudocell sizes",
            leave=False
        ):
            print(
                f"\n   >> Testing pseudocell size: {k}"
            )

            try:
                # size별 sampling seed 고정
                if k == "ALL":
                    k_seed = (
                        RANDOM_SEED
                        + 999999
                    )
                    k_label = "ALL"

                else:
                    k_seed = (
                        RANDOM_SEED
                        + int(k)
                    )
                    k_label = str(k)

                # ==================================================
                # Training donor에서만 pseudocell 생성
                # ==================================================

                X_train, meta_train = (
                    make_pseudocell_matrix_for_size(
                        adata=ad_tr,
                        cell_type_value=ct,
                        split_name="train",
                        genes=genes,
                        pseudocell_size=k,
                        seed=k_seed,
                        layer_name=LAYER_NAME
                    )
                )

                if X_train is None:
                    print(
                        "      [Skip] "
                        "No pseudocells generated"
                    )
                    continue

                # ==================================================
                # 동일한 donor-level fold를 모든 모델이 공유
                # ==================================================

                cv_splits = (
                    make_donor_cv_splits(
                        meta_train,
                        n_splits=N_CV,
                        seed=RANDOM_SEED
                    )
                )

                mean_real_cells = (
                    meta_train
                    .drop_duplicates(
                        "donor_id"
                    )[
                        "n_real_cells"
                    ]
                    .mean()
                )

                # ==================================================
                # Four models
                # ==================================================

                for (
                    model_name,
                    model
                ) in models.items():

                    print(
                        f"      {model_name} ..."
                    )

                    (
                        metrics,
                        donor_pred_df,
                        oof_pred_df
                    ) = cross_validate_model(
                        model_name=model_name,
                        model=model,
                        X=X_train,
                        meta=meta_train,
                        cv_splits=cv_splits
                    )

                    # ==================================================
                    # Donor-level prediction 저장
                    #
                    # 기존 형식 유지 + Model만 filename에 추가
                    # ==================================================

                    donor_pred_path = os.path.join(
                        DONOR_PRED_DIR,

                        f"{ct}_"
                        f"{model_name}_"
                        f"K{k_label}_"
                        f"donor_level_predictions.csv"
                    )

                    donor_pred_df.to_csv(
                        donor_pred_path,
                        index=False
                    )

                    # ==================================================
                    # Main result
                    #
                    # 기존 output column을 최대한 유지
                    # N_Test_*는 held-out test가 아니라
                    # OOF validation 전체를 의미함
                    # ==================================================

                    result = {
                        "Cell_Type":
                            ct,

                        "Pseudocell_Size":
                            k,

                        "Model":
                            model_name,

                        "N_Features":
                            X_train.shape[1],

                        "N_Train_Pseudocells":
                            X_train.shape[0],

                        # 기존 output 호환용
                        # OOF validation prediction 수
                        "N_Test_Pseudocells":
                            oof_pred_df.shape[0],

                        "N_Train_Donors":
                            meta_train[
                                "donor_id"
                            ].nunique(),

                        # 기존 output 호환용
                        # OOF validation donor 수
                        "N_Test_Donors":
                            donor_pred_df.shape[0],

                        "Mean_Train_Real_Cells_Per_Donor":
                            mean_real_cells,

                        # OOF는 모든 training donor를 한 번씩 평가하므로
                        # train과 동일
                        "Mean_Test_Real_Cells_Per_Donor":
                            mean_real_cells,

                        # hyperparameter tuning 없음
                        "alpha":
                            (
                                FIXED_LASSO_ALPHA
                                if model_name == "Lasso"
                                else FIXED_ELASTICNET_ALPHA
                                if model_name == "ElasticNet"
                                else np.nan
                            ),

                        "l1_ratio":
                            (
                                FIXED_ELASTICNET_L1_RATIO
                                if model_name == "ElasticNet"
                                else np.nan
                            ),

                        "CV_Folds":
                            N_CV,

                        "CV_Unit":
                            "donor",

                        "Evaluation_Set":
                            "5-fold OOF validation",

                        **metrics
                    }

                    all_results.append(
                        result
                    )

                    print(
                        f"         ✅ "
                        f"donor r="
                        f"{metrics['Donor_Pearson_R']:.3f}, "
                        f"donor MAE="
                        f"{metrics['Donor_MAE']:.3f}, "
                        f"donor RMSE="
                        f"{metrics['Donor_RMSE']:.3f}, "
                        f"pc r="
                        f"{metrics['Pseudocell_Pearson_R']:.3f}"
                    )

                    # 오래 걸릴 수 있으므로 매 모델마다 저장
                    pd.DataFrame(
                        all_results
                    ).to_csv(
                        os.path.join(
                            SAVE_DIR,
                            "pseudocell_size_all_models_results_partial.csv"
                        ),
                        index=False
                    )

            except Exception as e:
                print(
                    f"      [Fail] "
                    f"{ct}, "
                    f"pseudocell size={k}: "
                    f"{e}"
                )

                all_results.append({
                    "Cell_Type": ct,
                    "Pseudocell_Size": k,
                    "Model": np.nan,
                    "Error": str(e)
                })

                pd.DataFrame(
                    all_results
                ).to_csv(
                    os.path.join(
                        SAVE_DIR,
                        "pseudocell_size_all_models_results_partial.csv"
                    ),
                    index=False
                )

        del ad_tr, ad_te

    # ==============================================================
    # 8. Save final result
    # ==============================================================

    if len(all_results) == 0:
        print(
            "\nNo results generated."
        )
        return

    result_df = pd.DataFrame(
        all_results
    )

    final_path = os.path.join(
        SAVE_DIR,
        "pseudocell_size_all_models_results.csv"
    )

    result_df.to_csv(
        final_path,
        index=False
    )

    # ==============================================================
    # 9. 95% maximum Pearson r summary
    # ==============================================================

    valid_result_df = result_df[
        result_df[
            "Donor_Pearson_R"
        ].notna()
    ].copy()

    selection_results = []

    for (
        ct,
        model_name
    ), group in valid_result_df.groupby(
        [
            "Cell_Type",
            "Model"
        ]
    ):
        numeric_group = (
            group[
                group[
                    "Pseudocell_Size"
                ].astype(str)
                != "ALL"
            ]
            .copy()
        )

        numeric_group[
            "Pseudocell_Size_Num"
        ] = (
            numeric_group[
                "Pseudocell_Size"
            ]
            .astype(int)
        )

        numeric_group = (
            numeric_group
            .sort_values(
                "Pseudocell_Size_Num"
            )
        )

        max_r = (
            numeric_group[
                "Donor_Pearson_R"
            ]
            .max()
        )

        threshold_95 = (
            max_r * 0.95
        )

        reached = (
            numeric_group[
                numeric_group[
                    "Donor_Pearson_R"
                ]
                >= threshold_95
            ]
        )

        if len(reached) > 0:
            selected_size = int(
                reached.iloc[0][
                    "Pseudocell_Size_Num"
                ]
            )
        else:
            selected_size = np.nan

        selection_results.append({
            "Cell_Type":
                ct,

            "Model":
                model_name,

            "Max_Pearson_R":
                max_r,

            "Threshold_95":
                threshold_95,

            "First_Size_Reaching_95pct":
                selected_size
        })

    selection_df = pd.DataFrame(
        selection_results
    )

    selection_path = os.path.join(
        SAVE_DIR,
        "pseudocell_size_95pct_summary.csv"
    )

    selection_df.to_csv(
        selection_path,
        index=False
    )

    # ==============================================================
    # 10. Print summary
    # ==============================================================

    print(
        "\n=== COMPLETE ==="
    )

    print(
        f"Saved result: {final_path}"
    )

    print(
        f"Saved donor-level predictions in: "
        f"{DONOR_PRED_DIR}"
    )

    print(
        f"Saved 95% summary: "
        f"{selection_path}"
    )

    print(
        "\n=== 95% threshold summary ==="
    )

    print(
        selection_df.to_string(
            index=False
        )
    )

    print(
        "\n=== Performance summary ==="
    )

    display_cols = [
        "Cell_Type",
        "Pseudocell_Size",
        "Model",
        "N_Train_Pseudocells",
        "N_Train_Donors",
        "N_Features",
        "Donor_MAE",
        "Donor_RMSE",
        "Donor_R2",
        "Donor_Pearson_R",
        "Pseudocell_MAE",
        "Pseudocell_RMSE",
        "Pseudocell_R2",
        "Pseudocell_Pearson_R"
    ]

    existing_cols = [
        c for c in display_cols
        if c in result_df.columns
    ]

    print(
        result_df[
            existing_cols
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
