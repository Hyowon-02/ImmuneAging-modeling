# Curated publication source. See docs/SOURCE_MANIFEST.json.

# Source notebook cell 1
# -------------------------------------------------
# Donor-level 5-fold CV modeling
# trajectory DEG features
# -------------------------------------------------

import os
import json
import re
import warnings
import joblib

import numpy as np
import pandas as pd
import scanpy as sc
import lightgbm as lgb

from scipy.stats import pearsonr

from sklearn.linear_model import (
    Lasso,
    ElasticNet,
)

from sklearn.ensemble import (
    RandomForestRegressor,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==============================================================================
# 0. Config
# ==============================================================================

RANDOM_SEED = 260601

np.random.seed(
    RANDOM_SEED
)


# ------------------------------------------------------------------------------
# 기존 분석 대상 유지
# Figure 4에서 NK / CD4T / CD8T만 사용할 경우 Mono를 빼면 됨
# ------------------------------------------------------------------------------

CELLTYPES = ["NK", "CD4T", "CD8T"]


DATA_DIR = (
    "data/pseudocell_data_oversampling"
)

SAVE_DIR = (
    "Modeling_results"
)

DEG_DIR = (
    "./../2.DEG_analysis/data/DEG_analysis"
)


MODEL_SAVE_DIR = os.path.join(
    SAVE_DIR,
    "saved_models",
)

PRED_SAVE_DIR = os.path.join(
    SAVE_DIR,
    "cell_predictions",
)

CV_SAVE_DIR = os.path.join(
    SAVE_DIR,
    "cv_results",
)


os.makedirs(
    MODEL_SAVE_DIR,
    exist_ok=True,
)

os.makedirs(
    PRED_SAVE_DIR,
    exist_ok=True,
)

os.makedirs(
    CV_SAVE_DIR,
    exist_ok=True,
)


warnings.filterwarnings(
    "ignore"
)


# ==============================================================================
# 1. CV settings
# ==============================================================================

N_SPLITS = 5


# GridSearchCV 자체를 모든 CPU core로 병렬 실행
CV_N_JOBS = -1


# GridSearchCV가 이미 parallel하므로
# RF / LightGBM 내부는 1 thread로 제한
# -> nested parallelism 방지
MODEL_N_JOBS_DURING_CV = 1


# 최종 best model을 전체 training set으로 fitting할 때는
# 모든 CPU core 사용
FINAL_MODEL_N_JOBS = -1


# ==============================================================================
# 2. Evaluation
# ==============================================================================

def evaluate(
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


    if (
        len(y_pred) < 2
        or
        np.std(y_pred) < 1e-9
    ):

        pr = 0.0

    else:

        pr, _ = pearsonr(
            y_true,
            y_pred,
        )


    return (
        mae,
        rmse,
        r2,
        pr,
    )


# ==============================================================================
# 3. Donor column
# ==============================================================================

def get_donor_column(
    adata,
):

    candidates = [
        "donor_id",
        "donor",
        "batch",
    ]


    for column in candidates:

        if column in adata.obs.columns:

            return column


    raise KeyError(
        "Donor column not found. "
        "Expected donor_id, donor, or batch."
    )


# ==============================================================================
# 4. Donor-level 5-fold CV
# ==============================================================================

def make_donor_cv_splits(
    adata,
    n_splits=5,
    random_state=RANDOM_SEED,
):

    donor_col = (
        get_donor_column(
            adata
        )
    )


    donor_ids = (
        adata.obs[
            donor_col
        ]
        .astype(str)
        .to_numpy()
    )


    unique_donors = np.array(
        sorted(
            pd.unique(
                donor_ids
            )
        )
    )


    if len(
        unique_donors
    ) < n_splits:

        raise ValueError(
            f"Only {len(unique_donors)} donors "
            f"are available for "
            f"{n_splits}-fold CV."
        )


    # --------------------------------------------------------------------------
    # donor 자체를 5개 fold로 분할
    # --------------------------------------------------------------------------

    donor_kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


    cv_splits = []


    for fold_idx, (
        donor_train_idx,
        donor_valid_idx,
    ) in enumerate(
        donor_kfold.split(
            unique_donors
        ),
        start=1,
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


        # ----------------------------------------------------------------------
        # donor leakage check
        # ----------------------------------------------------------------------

        overlap = (
            train_donors
            & valid_donors
        )


        if overlap:

            raise RuntimeError(
                f"Donor leakage detected "
                f"in fold {fold_idx}: "
                f"{sorted(overlap)}"
            )


        # ----------------------------------------------------------------------
        # donor fold -> pseudocell row indices
        # ----------------------------------------------------------------------

        train_rows = np.where(
            np.isin(
                donor_ids,
                list(
                    train_donors
                ),
            )
        )[0]


        valid_rows = np.where(
            np.isin(
                donor_ids,
                list(
                    valid_donors
                ),
            )
        )[0]


        cv_splits.append(
            (
                train_rows,
                valid_rows,
            )
        )


        print(
            f"      Fold {fold_idx}: "
            f"{len(train_donors)} train donors / "
            f"{len(valid_donors)} validation donors"
        )


    return (
        cv_splits,
        donor_col,
    )


# ==============================================================================
# 5. Donor-level CV scorer
# ==============================================================================

def make_donor_rmse_scorer(
    donor_series,
):

    # index -> donor mapping
    donor_map = (
        donor_series
        .astype(str)
        .copy()
    )


    def donor_rmse_scorer(
        estimator,
        X,
        y_true,
    ):

        # ----------------------------------------------------------------------
        # pseudocell prediction
        # ----------------------------------------------------------------------

        y_pred = (
            estimator
            .predict(
                X
            )
        )


        # ----------------------------------------------------------------------
        # GridSearchCV가 DataFrame row index를 유지하므로
        # validation pseudocell의 donor ID를 가져옴
        # ----------------------------------------------------------------------

        donors = (
            donor_map
            .loc[
                X.index
            ]
            .to_numpy()
        )


        tmp = pd.DataFrame(
            {
                "donor":
                    donors,

                "true_age":
                    y_true,

                "pred_age":
                    y_pred,
            }
        )


        # ----------------------------------------------------------------------
        # donor별 pseudocell prediction median
        # ----------------------------------------------------------------------

        donor_df = (
            tmp
            .groupby(
                "donor",
                as_index=False,
            )
            .agg(
                true_age=(
                    "true_age",
                    "median",
                ),
                pred_age=(
                    "pred_age",
                    "median",
                ),
            )
        )


        rmse = np.sqrt(
            mean_squared_error(
                donor_df[
                    "true_age"
                ],
                donor_df[
                    "pred_age"
                ],
            )
        )


        # sklearn scorer는 클수록 좋은 값이어야 하므로
        # negative RMSE 반환
        return -rmse


    return donor_rmse_scorer
 
# ==============================================================================
# 6. Model specification
# ==============================================================================

def get_model_specs():

    # 최소한의 hyperparameter 후보
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]

    return {
        "Lasso": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=10000))
            ]),
            "param_grid": {
                "model__alpha": alpha_grid
            }
        },

        "ElasticNet": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(
                    l1_ratio=0.5,
                    max_iter=10000,
                    random_state=RANDOM_SEED
                ))
            ]),
            "param_grid": {
                "model__alpha": alpha_grid
            }
        },

        "RandomForest": {
            "estimator": RandomForestRegressor(
                n_estimators=100,
                n_jobs=MODEL_N_JOBS_DURING_CV,
                random_state=RANDOM_SEED
            ),
            "param_grid": {
                "max_depth": [None, 20]
            }
        },

        "LightGBM": {
            "estimator": lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                n_jobs=MODEL_N_JOBS_DURING_CV,
                random_state=RANDOM_SEED,
                verbosity=-1
            ),
            "param_grid": {
                "num_leaves": [15, 31]
            }
        }
    }

# ==============================================================================
# 7. DEG feature loading
# ==============================================================================

def load_deg_features(
    ct,
    ad_tr,
    ad_te,
    deg_dir,
):

    """
    DESeq2 LRT + amplitude filter를 통과한
    significant_DEGs_{ct}.csv의 gene을 feature로 사용.
    """

    deg_path = os.path.join(
        deg_dir,
        f"significant_DEGs_{ct}.csv",
    )


    if not os.path.exists(
        deg_path
    ):

        raise FileNotFoundError(
            f"DEG file not found: "
            f"{deg_path}"
        )


    deg_df = pd.read_csv(
        deg_path
    )


    if "gene" not in deg_df.columns:

        raise ValueError(
            f"'gene' column not found "
            f"in {deg_path}"
        )


    if "padj" in deg_df.columns:

        deg_df = (
            deg_df
            .sort_values(
                "padj",
                ascending=True,
            )
        )


    deg_genes = (
        deg_df[
            "gene"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


    train_genes = set(
        ad_tr.var_names
    )

    test_genes = set(
        ad_te.var_names
    )


    genes = [
        gene
        for gene
        in deg_genes
        if (
            gene in train_genes
            and
            gene in test_genes
        )
    ]


    return (
        genes,
        deg_df,
        deg_path,
    )


# ==============================================================================
# 8. Clean parameter names
# ==============================================================================

def clean_best_params(
    params,
):

    cleaned = {}


    for key, value in params.items():

        key = key.replace(
            "model__",
            "",
        )


        if isinstance(
            value,
            np.generic,
        ):

            value = (
                value.item()
            )


        cleaned[
            key
        ] = value


    return cleaned


# ==============================================================================
# 9. Build final estimator
#
# RF / LightGBM은 최종 fitting에서 n_jobs=-1 사용
# ==============================================================================

def build_final_estimator(
    model_name,
    best_params,
):

    # --------------------------------------------------------------------------
    # Lasso
    # --------------------------------------------------------------------------

    if model_name == "Lasso":

        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "model",
                    Lasso(
                        alpha=
                            best_params[
                                "alpha"
                            ],

                        max_iter=10000,
                    ),
                ),
            ]
        )


    # --------------------------------------------------------------------------
    # ElasticNet
    # --------------------------------------------------------------------------

    elif model_name == "ElasticNet":

        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "model",
                    ElasticNet(
                        alpha=
                            best_params[
                                "alpha"
                            ],

                        l1_ratio=0.5,

                        max_iter=10000,

                        random_state=
                            RANDOM_SEED,
                    ),
                ),
            ]
        )


    # --------------------------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------------------------

    elif model_name == "RandomForest":

        return RandomForestRegressor(
            n_estimators=100,

            max_depth=
                best_params[
                    "max_depth"
                ],

            n_jobs=
                FINAL_MODEL_N_JOBS,

            random_state=
                RANDOM_SEED,
        )


    # --------------------------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------------------------

    elif model_name == "LightGBM":

        return lgb.LGBMRegressor(
            n_estimators=100,

            learning_rate=0.1,

            num_leaves=
                best_params[
                    "num_leaves"
                ],

            n_jobs=
                FINAL_MODEL_N_JOBS,

            random_state=
                RANDOM_SEED,

            verbosity=-1,
        )


    else:

        raise ValueError(
            f"Unknown model: "
            f"{model_name}"
        )


# ==============================================================================
# 10. Main
# ==============================================================================

def main():

    all_results = []


    for ct in CELLTYPES:

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Cell type: {ct}"
        )

        print(
            "=" * 80
        )


        # ======================================================================
        # Load train / test
        # ======================================================================

        tr_path = os.path.join(
            DATA_DIR,
            f"{ct}_train_pseudo.h5ad",
        )

        te_path = os.path.join(
            DATA_DIR,
            f"{ct}_test_pseudo.h5ad",
        )


        if not (
            os.path.exists(
                tr_path
            )
            and
            os.path.exists(
                te_path
            )
        ):

            print(
                f"[Skip] Missing "
                f"train/test for {ct}"
            )

            continue


        ad_tr = sc.read_h5ad(
            tr_path
        )

        ad_te = sc.read_h5ad(
            te_path
        )


        if (
            "age"
            not in ad_tr.obs.columns
            or
            "age"
            not in ad_te.obs.columns
        ):

            print(
                f"[Skip] Missing age "
                f"column for {ct}"
            )

            continue


        # ======================================================================
        # Normalization
        # ======================================================================

        print(
            "Processing normalization..."
        )


        tr_max = (
            ad_tr.X.max()
        )

        te_max = (
            ad_te.X.max()
        )


        if hasattr(
            tr_max,
            "toarray",
        ):

            tr_max = (
                tr_max
                .toarray()
                .item()
            )


        if hasattr(
            te_max,
            "toarray",
        ):

            te_max = (
                te_max
                .toarray()
                .item()
            )


        if max(
            tr_max,
            te_max,
        ) > 100:

            sc.pp.normalize_total(
                ad_tr,
                target_sum=1e4,
            )

            sc.pp.log1p(
                ad_tr
            )


            sc.pp.normalize_total(
                ad_te,
                target_sum=1e4,
            )

            sc.pp.log1p(
                ad_te
            )

        else:

            print(
                "Data appears already normalized."
            )


        # ======================================================================
        # Feature selection
        # ======================================================================

        try:

            (
                genes,
                deg_df,
                deg_path,
            ) = load_deg_features(
                ct=ct,
                ad_tr=ad_tr,
                ad_te=ad_te,
                deg_dir=DEG_DIR,
            )


        except Exception as e:

            print(
                f"[Skip] DEG feature "
                f"loading failed: {e}"
            )

            continue


        print(
            f"DEG file: "
            f"{deg_path}"
        )

        print(
            f"Selected features: "
            f"{len(genes):,}"
        )


        if len(
            genes
        ) == 0:

            continue


        ad_tr = ad_tr[
            :,
            genes,
        ].copy()


        ad_te = ad_te[
            :,
            genes,
        ].copy()


        # ======================================================================
        # X / y
        # ======================================================================

        X_train = (
            ad_tr
            .to_df()
            .astype(
                np.float64
            )
        )


        X_test = (
            ad_te
            .to_df()
            .astype(
                np.float64
            )
        )


        y_train = (
            ad_tr.obs[
                "age"
            ]
            .astype(float)
            .to_numpy()
        )


        y_test = (
            ad_te.obs[
                "age"
            ]
            .astype(float)
            .to_numpy()
        )


        # ======================================================================
        # Donor information
        # ======================================================================

        donor_col = (
            get_donor_column(
                ad_tr
            )
        )


        train_donor_series = (
            ad_tr.obs[
                donor_col
            ]
            .astype(str)
            .copy()
        )


        # DataFrame index와 맞춰줌
        train_donor_series.index = (
            X_train.index
        )


        # ======================================================================
        # Donor-level 5-fold CV
        # ======================================================================

        print(
            "\nCreating donor-level "
            "5-fold CV..."
        )


        (
            cv_splits,
            donor_col,
        ) = make_donor_cv_splits(
            ad_tr,
            n_splits=
                N_SPLITS,
            random_state=
                RANDOM_SEED,
        )


        donor_scorer = (
            make_donor_rmse_scorer(
                train_donor_series
            )
        )


        # ======================================================================
        # LightGBM-safe gene names
        # ======================================================================

        X_train_lgb = (
            X_train.rename(
                columns=lambda x:
                    re.sub(
                        r"[^\w]",
                        "_",
                        str(x),
                    )
            )
        )


        X_test_lgb = (
            X_test.rename(
                columns=lambda x:
                    re.sub(
                        r"[^\w]",
                        "_",
                        str(x),
                    )
            )
        )


        # donor map의 index는 동일하게 유지되므로
        # LightGBM에서도 donor scorer 사용 가능


        # ======================================================================
        # Models
        # ======================================================================

        model_specs = (
            get_model_specs()
        )


        for (
            model_name,
            spec,
        ) in model_specs.items():

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"{ct} | "
                f"{model_name}"
            )

            print(
                "-" * 70
            )


            # ------------------------------------------------------------------
            # LightGBM column names
            # ------------------------------------------------------------------

            if model_name == "LightGBM":

                X_tr_cv = (
                    X_train_lgb
                )

                X_te_final = (
                    X_test_lgb
                )

            else:

                X_tr_cv = (
                    X_train
                )

                X_te_final = (
                    X_test
                )


            # ==================================================================
            # Donor-level GridSearchCV
            # ==================================================================

            search = GridSearchCV(

                estimator=
                    spec[
                        "estimator"
                    ],

                param_grid=
                    spec[
                        "param_grid"
                    ],

                scoring=
                    donor_scorer,

                cv=
                    cv_splits,

                # 모든 CPU core 사용
                n_jobs=
                    CV_N_JOBS,

                refit=
                    False,

                return_train_score=
                    False,

                verbose=1,
            )


            try:

                # --------------------------------------------------------------
                # Hyperparameter search
                # --------------------------------------------------------------

                search.fit(
                    X_tr_cv,
                    y_train,
                )


                # --------------------------------------------------------------
                # Save CV result
                # --------------------------------------------------------------

                cv_result_df = (
                    pd.DataFrame(
                        search.cv_results_
                    )
                )


                cv_result_path = (
                    os.path.join(
                        CV_SAVE_DIR,

                        f"{ct}_"
                        f"{model_name}_"
                        f"cv_results.csv",
                    )
                )


                cv_result_df.to_csv(
                    cv_result_path,
                    index=False,
                )


                # --------------------------------------------------------------
                # Best parameter
                #
                # score = negative donor-level RMSE
                # --------------------------------------------------------------

                best_index = int(
                    np.nanargmax(
                        search.cv_results_[
                            "mean_test_score"
                        ]
                    )
                )


                raw_best_params = (
                    search.cv_results_[
                        "params"
                    ][
                        best_index
                    ]
                )


                best_params = (
                    clean_best_params(
                        raw_best_params
                    )
                )


                best_cv_rmse = (
                    -search.cv_results_[
                        "mean_test_score"
                    ][
                        best_index
                    ]
                )


                print(
                    f"Best parameters: "
                    f"{best_params}"
                )


                print(
                    f"Best donor-level "
                    f"CV RMSE: "
                    f"{best_cv_rmse:.3f}"
                )


                # ==================================================================
                # Fit final model on all training pseudocells
                # ==================================================================

                final_model = (
                    build_final_estimator(
                        model_name=
                            model_name,

                        best_params=
                            best_params,
                    )
                )


                final_model.fit(
                    X_tr_cv,
                    y_train,
                )


                # ==================================================================
                # Independent test pseudocell prediction
                # ==================================================================

                y_pred = (
                    final_model
                    .predict(
                        X_te_final
                    )
                )


                (
                    mae,
                    rmse,
                    r2,
                    pr,
                ) = evaluate(
                    y_test,
                    y_pred,
                )


                print(
                    f"Test pseudocell-level: "
                    f"MAE={mae:.3f}, "
                    f"RMSE={rmse:.3f}, "
                    f"R2={r2:.3f}, "
                    f"Pearson r={pr:.3f}"
                )


                # ==================================================================
                # Results
                # ==================================================================

                all_results.append(
                    {
                        "Cell_Type":
                            ct,

                        "Feature_Set":
                            "trajectory_DEG",

                        "Model":
                            model_name,

                        "N_Samples_Train":
                            X_train.shape[0],

                        "N_Samples_Test":
                            X_test.shape[0],

                        "N_Features_Total":
                            X_train.shape[1],

                        "CV_Folds":
                            N_SPLITS,

                        "CV_Unit":
                            "donor",

                        "Best_CV_RMSE":
                            best_cv_rmse,

                        "MAE":
                            mae,

                        "RMSE":
                            rmse,

                        "R2":
                            r2,

                        "Pearson_R":
                            pr,
                    }
                )


                # ==================================================================
                # Prediction file
                # ==================================================================

                pred_df = pd.DataFrame(
                    {
                        "cell_id":
                            ad_te.obs_names,

                        "true_age":
                            y_test,

                        "pred_age":
                            y_pred,
                    }
                )


                test_donor_col = (
                    get_donor_column(
                        ad_te
                    )
                )


                pred_df[
                    "batch"
                ] = (
                    ad_te.obs[
                        test_donor_col
                    ]
                    .astype(str)
                    .to_numpy()
                )


                base_filename = (
                    f"{ct}_"
                    f"{model_name}"
                )


                pred_df.to_csv(
                    os.path.join(
                        PRED_SAVE_DIR,

                        f"{base_filename}"
                        "_predictions.csv",
                    ),

                    index=False,
                )


                # ==================================================================
                # Save model
                #
                # 기존 external validation 코드와 호환되도록
                # linear model은 model + scaler를 따로 저장
                # ==================================================================

                is_linear_model = (
                    model_name
                    in [
                        "Lasso",
                        "ElasticNet",
                    ]
                )


                if is_linear_model:

                    fitted_scaler = (
                        final_model
                        .named_steps[
                            "scaler"
                        ]
                    )


                    fitted_model = (
                        final_model
                        .named_steps[
                            "model"
                        ]
                    )


                    joblib.dump(
                        fitted_model,

                        os.path.join(
                            MODEL_SAVE_DIR,

                            f"{base_filename}"
                            ".pkl",
                        ),
                    )


                    joblib.dump(
                        fitted_scaler,

                        os.path.join(
                            MODEL_SAVE_DIR,

                            f"{base_filename}"
                            "_scaler.pkl",
                        ),
                    )


                else:

                    joblib.dump(
                        final_model,

                        os.path.join(
                            MODEL_SAVE_DIR,

                            f"{base_filename}"
                            ".pkl",
                        ),
                    )


                # ==================================================================
                # Feature metadata
                # ==================================================================

                feature_meta = {

                    "cell_type":
                        ct,

                    "feature_set":
                        "trajectory_DEG",

                    "deg_file":
                        deg_path,

                    "model":
                        model_name,

                    "feature_names":
                        X_train.columns.tolist(),

                    "n_features":
                        X_train.shape[1],

                    "scaled":
                        is_linear_model,

                    "cv_folds":
                        N_SPLITS,

                    "cv_unit":
                        "donor",

                    "cv_metric":
                        "donor-level RMSE",

                    "best_cv_rmse":
                        float(
                            best_cv_rmse
                        ),

                    "best_params":
                        best_params,

                    "metrics": {

                        "MAE":
                            mae,

                        "RMSE":
                            rmse,

                        "R2":
                            r2,

                        "Pearson_R":
                            pr,
                    },
                }


                with open(
                    os.path.join(
                        MODEL_SAVE_DIR,

                        f"{base_filename}"
                        "_features.json",
                    ),

                    "w",
                ) as f:

                    json.dump(
                        feature_meta,
                        f,
                        indent=4,
                    )


            except Exception as e:

                print(
                    f"[Fail] "
                    f"{model_name}: "
                    f"{e}"
                )


    # ==========================================================================
    # Save benchmark
    # ==========================================================================

    if len(
        all_results
    ) > 0:

        result_df = (
            pd.DataFrame(
                all_results
            )
        )


        raw_path = os.path.join(
            SAVE_DIR,
            "benchmark_raw.csv",
        )


        summary_path = os.path.join(
            SAVE_DIR,
            "benchmark_summary.csv",
        )


        result_df.to_csv(
            raw_path,
            index=False,
        )


        result_df.to_csv(
            summary_path,
            index=False,
        )


        print(
            "\n"
            + "=" * 80
        )

        print(
            "Finished."
        )

        print(
            f"Raw result: "
            f"{raw_path}"
        )

        print(
            f"Summary: "
            f"{summary_path}"
        )

        print(
            "=" * 80
        )


    else:

        print(
            "No results generated."
        )


# ==============================================================================
# Run
# ==============================================================================

if __name__ == "__main__":

    main()
