# Curated publication source. See docs/SOURCE_MANIFEST.json.

# Source notebook cell 0
import os
import gc
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata

from scipy import sparse
from scipy.stats import pearsonr

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split,
    GridSearchCV,
    KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==============================================================
# 1. Configuration
# ==============================================================

RANDOM_SEED = 260601
N_RESAMPLING = 100
N_THREADS = -1

CELLTYPES = ["NK", "CD8T", "CD4T"]

COL_DONOR = "donor_id"
COL_AGE = "age"
COL_SEX = "sex"
COL_CELLTYPE = "Modeling_annotation"

LAYER_NAME = "raw_counts"

PSEUDOCELL_SAMPLE_SIZE = 20
MIN_CELLS_PER_DONOR = 30

# 최소한의 alpha 후보
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]

N_CV_FOLDS = 5


# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------

RAW_DATA_PATH = "../raw_data/AIDA_pp.h5ad"
DEG_DIR = "./../2.DEG_analysis/data/DEG_analysis"
RESULT_DIR = "resampling_test_100"

os.makedirs(RESULT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


# ==============================================================
# 2. Load data
# ==============================================================

print(">>> Loading Raw AnnData...")

adata = sc.read_h5ad(RAW_DATA_PATH)


# ==============================================================
# 3. Helper functions
# ==============================================================

def split_donors_stratified(adata, test_size, seed):
    """
    Donor 단위 Train/Test split.

    Country + age group + sex를 이용하여 가능한 한
    Train/Test 분포가 유사하도록 stratification.
    """

    cols = ["donor_id", "age", "Country", "sex"]

    donors_df = (
        adata.obs[cols]
        .drop_duplicates(subset="donor_id")
        .set_index("donor_id")
    )

    donors_df["age_group"] = pd.cut(
        donors_df["age"],
        bins=range(20, 101, 10),
        right=False
    )

    stratify_parts = []

    if "Country" in donors_df.columns:
        stratify_parts.append(
            donors_df["Country"].astype(str)
        )

    stratify_parts.append(
        donors_df["age_group"].astype(str)
    )

    if COL_SEX:
        stratify_parts.append(
            donors_df[COL_SEX].astype(str)
        )

    donors_df["stratify_key"] = stratify_parts[0]

    for part in stratify_parts[1:]:
        donors_df["stratify_key"] = (
            donors_df["stratify_key"]
            + "_"
            + part
        )

    # singleton group 처리
    key_counts = donors_df["stratify_key"].value_counts()
    single_sample_keys = key_counts[key_counts < 2].index

    donors_df["stratify_key"] = (
        donors_df["stratify_key"]
        .replace(
            single_sample_keys,
            "single_sample_group"
        )
    )

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed
    )

    try:
        train_indices, test_indices = next(
            sss.split(
                donors_df.index,
                donors_df["stratify_key"]
            )
        )

    except ValueError:
        train_indices, test_indices = train_test_split(
            range(len(donors_df)),
            test_size=test_size,
            random_state=seed
        )

    train_donors = donors_df.index[train_indices]
    test_donors = donors_df.index[test_indices]

    adata_train = adata[
        adata.obs["donor_id"].isin(train_donors)
    ].copy()

    adata_test = adata[
        adata.obs["donor_id"].isin(test_donors)
    ].copy()

    return adata_train, adata_test


# ==============================================================
# DEG feature loading
# ==============================================================

def load_deg_features(ct, ad_tr, ad_te, deg_dir):
    """
    DESeq2 LRT + amplitude filter를 통과한
    significant_DEGs_{ct}.csv의 유전자를 feature로 사용.
    """

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
        if g in train_genes and g in test_genes
    ]

    return genes, deg_df, deg_path


# ==============================================================
# Pseudocell generation
# ==============================================================

def safe_sum_of_rows_layer(
    adata,
    indices,
    layer_name
):
    if (
        layer_name is not None
        and layer_name in adata.layers
    ):
        subset = adata.layers[layer_name][indices]
    else:
        subset = adata.X[indices]

    if sparse.issparse(subset):
        return np.asarray(
            subset.sum(axis=0)
        ).flatten()

    return np.sum(
        subset,
        axis=0
    )


def make_pseudo_cells(
    adata,
    cell_type_value,
    split_name,
    cell_type_col=COL_CELLTYPE,
    donor_col=COL_DONOR,
    age_col=COL_AGE,
    sex_col=COL_SEX,
    layer_name=LAYER_NAME
):

    if cell_type_col not in adata.obs:
        return None

    mask = (
        adata.obs[cell_type_col]
        == cell_type_value
    )

    if not mask.any():
        return None

    subset_obs = adata.obs[mask]
    target_indices = np.where(mask)[0]

    donors = subset_obs[donor_col].unique()

    rows = []
    metas = []

    for donor in donors:

        donor_mask = (
            subset_obs[donor_col]
            == donor
        )

        n_real = donor_mask.sum()

        if n_real < MIN_CELLS_PER_DONOR:
            continue

        donor_age = (
            subset_obs
            .loc[
                subset_obs[donor_col] == donor,
                age_col
            ]
            .iloc[0]
        )

        # ----------------------------------------------------------
        # Training:
        # <50 years  -> 100 pseudocells
        # >=50 years -> 200 pseudocells
        #
        # Test:
        # all donors -> 100 pseudocells
        # ----------------------------------------------------------

        if (
            split_name == "train"
            and donor_age >= 50
        ):
            current_n_pseudo = 200
        else:
            current_n_pseudo = 100

        donor_global_indices = (
            target_indices[donor_mask]
        )

        for i in range(current_n_pseudo):

            sampled_idx = np.random.choice(
                donor_global_indices,
                size=PSEUDOCELL_SAMPLE_SIZE,
                replace=True
            )

            sum_vec = safe_sum_of_rows_layer(
                adata,
                sampled_idx,
                layer_name
            )

            rows.append(sum_vec)

            metas.append({
                "pseudocell_id":
                    f"{cell_type_value}_"
                    f"{split_name}_"
                    f"{donor}_pc{i+1}",

                "donor_id":
                    donor,

                "cell_type":
                    cell_type_value,

                "n_real_cells":
                    n_real,

                "age":
                    donor_age,

                "sex":
                    subset_obs.loc[
                        subset_obs[donor_col] == donor,
                        sex_col
                    ].iloc[0]
                    if sex_col in subset_obs.columns
                    else "Unknown",

                "split":
                    split_name
            })

    if not rows:
        return None

    pseudo_X = np.vstack(rows)

    meta_df = pd.DataFrame(metas)

    meta_df.set_index(
        "pseudocell_id",
        inplace=True
    )

    df_pseudo = pd.DataFrame(
        pseudo_X,
        columns=adata.var_names,
        index=meta_df.index
    )

    return anndata.AnnData(
        X=df_pseudo,
        obs=meta_df
    )


# ==============================================================
# Evaluation
# ==============================================================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    if (
        len(y_pred) < 2
        or np.std(y_pred) < 1e-9
    ):
        pr = 0.0
    else:
        pr = pearsonr(
            y_true,
            y_pred
        )[0]

    return mae, rmse, r2, pr


# ==============================================================
# Donor-level CV split
# ==============================================================

def make_donor_cv_splits(
    pseudo_train,
    n_splits=5,
    seed=RANDOM_SEED
):
    """
    같은 donor에서 생성된 모든 pseudocell을
    동일한 CV fold에 배정.
    """

    donor_ids = (
        pseudo_train.obs["donor_id"]
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
        train_d_idx,
        val_d_idx
    ) in enumerate(
        kf.split(unique_donors),
        start=1
    ):

        train_donors = (
            unique_donors[train_d_idx]
        )

        val_donors = (
            unique_donors[val_d_idx]
        )

        # donor leakage 확인
        overlap = (
            set(train_donors)
            & set(val_donors)
        )

        if overlap:
            raise RuntimeError(
                f"Donor leakage detected "
                f"in fold {fold_idx}"
            )

        train_idx = np.where(
            np.isin(
                donor_ids,
                train_donors
            )
        )[0]

        val_idx = np.where(
            np.isin(
                donor_ids,
                val_donors
            )
        )[0]

        cv_splits.append(
            (
                train_idx,
                val_idx
            )
        )

    return cv_splits


# ==============================================================
# Donor-level RMSE scorer
# ==============================================================

class DonorRMSEScorer:
    """
    각 CV validation fold에서 pseudocell prediction을
    donor 단위 median으로 통합한 후 RMSE 계산.
    """

    def __init__(
        self,
        donor_series
    ):
        self.donor_map = (
            donor_series
            .astype(str)
            .copy()
        )

    def __call__(
        self,
        estimator,
        X,
        y_true
    ):

        y_pred = estimator.predict(X)

        donors = (
            self.donor_map
            .loc[X.index]
            .values
        )

        df = pd.DataFrame({
            "donor_id": donors,
            "true_age": y_true,
            "pred_age": y_pred
        })

        donor_df = (
            df
            .groupby(
                "donor_id",
                as_index=False
            )
            .agg(
                true_age=(
                    "true_age",
                    "mean"
                ),
                pred_age=(
                    "pred_age",
                    "median"
                )
            )
        )

        rmse = np.sqrt(
            mean_squared_error(
                donor_df["true_age"],
                donor_df["pred_age"]
            )
        )

        # sklearn scorer는 큰 값이 좋은 것으로 판단
        return -rmse


# ==============================================================
# 4. Main resampling loop
# ==============================================================

seeds = list(
    range(
        1,
        N_RESAMPLING + 1
    )
)

perf_results = []
feature_results = []


for random_seed in seeds:

    print(
        f"\n{'='*60}\n"
        f">>> Seed {random_seed} Start...\n"
        f"{'='*60}"
    )

    # ----------------------------------------------------------
    # pseudocell sampling도 seed별 재현 가능하도록 설정
    # ----------------------------------------------------------

    np.random.seed(
        random_seed
    )

    # ----------------------------------------------------------
    # 1. Outer Train / Test donor split
    # ----------------------------------------------------------

    adata_train, adata_test = (
        split_donors_stratified(
            adata,
            test_size=0.2,
            seed=random_seed
        )
    )

    print(
        f">>> Train donors: "
        f"{adata_train.obs['donor_id'].nunique()}"
    )

    print(
        f">>> Test donors : "
        f"{adata_test.obs['donor_id'].nunique()}"
    )


    # ----------------------------------------------------------
    # 2. Cell-type-specific modeling
    # ----------------------------------------------------------

    for ct in CELLTYPES:

        print(
            f"\n   📌 Celltype: {ct}"
        )

        try:

            # ==================================================
            # 2-1. DEG features
            # ==================================================

            deg_genes, _, _ = (
                load_deg_features(
                    ct,
                    adata_train,
                    adata_test,
                    DEG_DIR
                )
            )

            if not deg_genes:

                print(
                    f"      [Skip] "
                    f"No valid DEGs found for {ct}."
                )

                continue

            print(
                f"      >>> Phase 1: "
                f"{len(deg_genes)} DEG features loaded."
            )


            # ==================================================
            # 2-2. Pseudocell generation
            # ==================================================

            pseudo_train = make_pseudo_cells(
                adata_train,
                ct,
                "train"
            )

            pseudo_test = make_pseudo_cells(
                adata_test,
                ct,
                "test"
            )

            if (
                pseudo_train is None
                or pseudo_test is None
            ):

                print(
                    f"      [Skip] "
                    f"Not enough pseudocell data for {ct}."
                )

                continue


            # ==================================================
            # 2-3. Normalization
            # ==================================================

            sc.pp.normalize_total(
                pseudo_train,
                target_sum=1e4
            )

            sc.pp.log1p(
                pseudo_train
            )

            sc.pp.normalize_total(
                pseudo_test,
                target_sum=1e4
            )

            sc.pp.log1p(
                pseudo_test
            )


            # ==================================================
            # 2-4. Feature subset
            # ==================================================

            valid_genes_final = [
                g for g in deg_genes
                if (
                    g in pseudo_train.var_names
                    and g in pseudo_test.var_names
                )
            ]

            if not valid_genes_final:

                print(
                    f"      [Skip] "
                    f"No overlapping features for {ct}."
                )

                continue


            pseudo_train = (
                pseudo_train[
                    :,
                    valid_genes_final
                ]
                .copy()
            )

            pseudo_test = (
                pseudo_test[
                    :,
                    valid_genes_final
                ]
                .copy()
            )


            # ==================================================
            # 2-5. X / y
            # ==================================================

            X_train = (
                pseudo_train
                .to_df()
                .astype(np.float64)
            )

            X_test = (
                pseudo_test
                .to_df()
                .astype(np.float64)
            )

            y_train = (
                pseudo_train.obs["age"]
                .astype(float)
                .values
            )

            y_test = (
                pseudo_test.obs["age"]
                .astype(float)
                .values
            )

            test_donor_ids = (
                pseudo_test.obs[
                    "donor_id"
                ]
                .astype(str)
                .values
            )


            # ==================================================
            # 2-6. Donor-level 5-fold CV
            # ==================================================

            cv_splits = (
                make_donor_cv_splits(
                    pseudo_train,
                    n_splits=N_CV_FOLDS,
                    seed=random_seed
                )
            )

            donor_series = (
                pseudo_train.obs[
                    "donor_id"
                ]
                .astype(str)
                .copy()
            )

            donor_series.index = (
                X_train.index
            )

            donor_scorer = (
                DonorRMSEScorer(
                    donor_series
                )
            )


            # ==================================================
            # 2-7. ElasticNet
            # ==================================================

            pipeline = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    ElasticNet(
                        l1_ratio=0.5,
                        max_iter=10000,
                        random_state=RANDOM_SEED
                    )
                )
            ])


            search = GridSearchCV(
                estimator=pipeline,
                param_grid={
                    "model__alpha":
                        ALPHA_GRID
                },
                scoring=donor_scorer,
                cv=cv_splits,

                # 모든 CPU core 사용
                n_jobs=N_THREADS,

                refit=True,
                return_train_score=False,
                verbose=1
            )


            print(
                f"      >>> Phase 2: "
                f"ElasticNet donor-level "
                f"{N_CV_FOLDS}-fold CV "
                f"(Features: {len(valid_genes_final)})"
            )


            search.fit(
                X_train,
                y_train
            )


            final_model = (
                search.best_estimator_
            )

            best_alpha = (
                search.best_params_[
                    "model__alpha"
                ]
            )

            best_cv_rmse = (
                -search.best_score_
            )


            print(
                f"      Best alpha: "
                f"{best_alpha}"
            )

            print(
                f"      Best donor-level "
                f"CV RMSE: "
                f"{best_cv_rmse:.3f}"
            )


            # ==================================================
            # 2-8. Outer test prediction
            # ==================================================

            y_pred = (
                final_model.predict(
                    X_test
                )
            )


            # ==================================================
            # 2-9. Donor-level aggregation
            # ==================================================

            df_pred = pd.DataFrame({
                "donor_id":
                    test_donor_ids,

                "true_age":
                    y_test,

                "pred_age":
                    y_pred
            })


            donor_agg = (
                df_pred
                .groupby(
                    "donor_id",
                    as_index=False
                )
                .agg(
                    true_age=(
                        "true_age",
                        "mean"
                    ),
                    pred_age=(
                        "pred_age",
                        "median"
                    )
                )
            )


            (
                d_mae,
                d_rmse,
                d_r2,
                d_pr
            ) = evaluate(
                donor_agg[
                    "true_age"
                ].values,

                donor_agg[
                    "pred_age"
                ].values
            )


            # ==================================================
            # 2-10. Performance record
            # ==================================================

            perf_results.append({
                "Seed":
                    random_seed,

                "Cell_Type":
                    ct,

                "Donor_MAE":
                    d_mae,

                "Donor_RMSE":
                    d_rmse,

                "Donor_R2":
                    d_r2,

                "Donor_Pearson_R":
                    d_pr,

                "Alpha":
                    best_alpha,

                "CV_RMSE":
                    best_cv_rmse,

                "CV_Folds":
                    N_CV_FOLDS,

                "CV_Unit":
                    "donor",

                "N_Features_Used":
                    len(
                        valid_genes_final
                    ),

                "N_Train_Pseudocells":
                    len(
                        y_train
                    ),

                "N_Test_Donors":
                    len(
                        donor_agg
                    )
            })


            # ==================================================
            # 2-11. Selected features
            # ==================================================

            elasticnet_model = (
                final_model
                .named_steps[
                    "model"
                ]
            )

            final_coefs = (
                elasticnet_model.coef_
            )

            final_mask = (
                final_coefs != 0
            )

            retained_genes = (
                X_train.columns[
                    final_mask
                ]
            )

            retained_coefs = (
                final_coefs[
                    final_mask
                ]
            )


            for gene, coef in zip(
                retained_genes,
                retained_coefs
            ):

                feature_results.append({
                    "Seed":
                        random_seed,

                    "Cell_Type":
                        ct,

                    "Gene":
                        gene,

                    "Coefficient":
                        coef,

                    "Alpha":
                        best_alpha
                })


            print(
                f"      ✅ {ct} | "
                f"Donor MAE={d_mae:.2f}, "
                f"RMSE={d_rmse:.2f}, "
                f"R2={d_r2:.2f}, "
                f"r={d_pr:.2f}, "
                f"Genes={final_mask.sum()}"
            )


            # ==================================================
            # Clean up
            # ==================================================

            del (
                pseudo_train,
                pseudo_test,
                X_train,
                X_test,
                final_model,
                search
            )

            gc.collect()


        except FileNotFoundError as e:

            print(
                f"      ❌ {ct} File Error: "
                f"{e}"
            )


        except Exception as e:

            print(
                f"      ❌ {ct} Modeling Failed: "
                f"{e}"
            )

            gc.collect()


    del (
        adata_train,
        adata_test
    )

    gc.collect()


# ==============================================================
# 5. Save final results
# ==============================================================

print(
    "\n>>> Saving Final "
    "Resampling Results..."
)


# --------------------------------------------------------------
# Performance
# --------------------------------------------------------------

df_perf = pd.DataFrame(
    perf_results
)

df_perf.to_csv(
    os.path.join(
        RESULT_DIR,
        "resampling_performance_metrics.csv"
    ),
    index=False
)


# --------------------------------------------------------------
# Selected features
# --------------------------------------------------------------

df_feat = pd.DataFrame(
    feature_results
)

df_feat.to_csv(
    os.path.join(
        RESULT_DIR,
        "resampling_selected_features.csv"
    ),
    index=False
)


# ==============================================================
# 6. Summary
# ==============================================================

if not df_perf.empty:

    print(
        "\n=== Summary "
        "(Mean over seeds, Donor Level) ==="
    )

    summary = (
        df_perf
        .groupby(
            "Cell_Type"
        )[
            [
                "Donor_MAE",
                "Donor_RMSE",
                "Donor_R2",
                "Donor_Pearson_R"
            ]
        ]
        .agg(
            [
                "mean",
                "std"
            ]
        )
    )

    print(summary)


    summary.to_csv(
        os.path.join(
            RESULT_DIR,
            "resampling_performance_summary.csv"
        )
    )


# ==============================================================
# 7. Feature stability summary
# ==============================================================

if not df_feat.empty:

    feature_stability = (
        df_feat
        .groupby(
            [
                "Cell_Type",
                "Gene"
            ]
        )
        .agg(
            Selection_Count=(
                "Seed",
                "nunique"
            ),
            Mean_Coefficient=(
                "Coefficient",
                "mean"
            ),
            SD_Coefficient=(
                "Coefficient",
                "std"
            )
        )
        .reset_index()
    )


    feature_stability[
        "Selection_Frequency"
    ] = (
        feature_stability[
            "Selection_Count"
        ]
        / N_RESAMPLING
    )


    feature_stability.to_csv(
        os.path.join(
            RESULT_DIR,
            "resampling_feature_stability.csv"
        ),
        index=False
    )


    print(
        "\n>>> Feature stability "
        "summary saved."
    )


print(
    "\n>>> All resampling analyses completed."
)
