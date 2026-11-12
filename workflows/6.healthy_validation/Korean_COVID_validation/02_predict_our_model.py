# Curated publication source. See docs/SOURCE_MANIFEST.json.

# Source notebook cell 0
import os
import json
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

# ==============================================================
# 0) Config
# ==============================================================

MODEL_DIR = "./../../4.prediction_model/Modeling_results/saved_models"
VAL_DATA_DIR = "pseudocells_validation"
VAL_RESULT_DIR = "validation_results"

# gene symbol -> Ensembl ID mapping file
GENE_MAP_PATH = "../../raw_data/gene_id_mapping.csv"

CELLTYPES = ["NK", "CD8T", "CD4T"]
SEX_MODE = "General"
TARGET_MODEL = "ElasticNet"

DONOR_COL = "sampleID"
STATE_COL = "state"

# validation cohort 내부에서 StandardScaler를 새로 적용
# 파일명은 기존과 동일하게 저장
SCALER_MODE = "validation"

# feature match가 너무 낮으면 validation skip
MIN_FEATURE_MATCH_RATIO = 0.70

# log-normalized expression이면 mean 권장
DUPLICATE_GENE_AGG = "mean"   # "mean" or "sum"

os.makedirs(VAL_RESULT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


# ==============================================================
# Helper functions
# ==============================================================

def strip_ensembl_version(x):
    """
    ENSG00000123456.12 -> ENSG00000123456
    """
    if pd.isna(x):
        return x
    return str(x).split(".")[0]


def calculate_metrics(y_true, y_pred):
    """
    MAE, RMSE, R2, Pearson R 계산
    NaN / Inf 제거 포함
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan, np.nan, np.nan, np.nan

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        pr = 0.0
    else:
        pr, _ = pearsonr(y_true, y_pred)

    return mae, rmse, r2, pr


def load_gene_mapping(mapping_path):
    """
    gene_id_mapping.csv 로드
    필요한 컬럼:
      - ensembl_id
      - gene_symbol
    """
    gene_map = pd.read_csv(mapping_path)

    required_cols = {"ensembl_id", "gene_symbol"}
    missing_cols = required_cols - set(gene_map.columns)

    if missing_cols:
        raise ValueError(
            f"Mapping file에 필요한 컬럼이 없습니다: {missing_cols}\n"
            f"현재 컬럼: {gene_map.columns.tolist()}"
        )

    gene_map = gene_map[["ensembl_id", "gene_symbol"]].copy()

    gene_map["ensembl_id"] = (
        gene_map["ensembl_id"]
        .astype(str)
        .map(strip_ensembl_version)
    )

    gene_map["gene_symbol"] = gene_map["gene_symbol"].astype(str)

    gene_map = gene_map.dropna(subset=["ensembl_id", "gene_symbol"])

    gene_map = gene_map[
        (gene_map["ensembl_id"] != "") &
        (gene_map["gene_symbol"] != "") &
        (gene_map["ensembl_id"].str.lower() != "nan") &
        (gene_map["gene_symbol"].str.lower() != "nan")
    ]

    gene_map = gene_map.drop_duplicates(subset=["ensembl_id", "gene_symbol"])

    return gene_map


def convert_symbol_to_ensembl_df(
    adata,
    mapping_path,
    aggregate="mean"
):
    """
    validation AnnData의 var_names가 gene symbol일 때,
    모델 feature인 Ensembl ID 기준 DataFrame으로 변환.

    Returns
    -------
    df_ens : pd.DataFrame
        rows = cells/pseudocells
        columns = Ensembl IDs
    """

    df = adata.to_df()

    print(f"   - Validation matrix: {df.shape[0]:,} cells x {df.shape[1]:,} genes")
    print(f"   - Example validation genes: {list(df.columns[:5])}")

    gene_map = load_gene_mapping(mapping_path)

    # gene_symbol -> ensembl_id dictionary
    # 같은 symbol이 여러 Ensembl로 매핑되는 경우 첫 번째 사용
    symbol_to_ens_df = gene_map.drop_duplicates(subset=["gene_symbol"], keep="first")

    symbol_to_ens = dict(
        zip(symbol_to_ens_df["gene_symbol"], symbol_to_ens_df["ensembl_id"])
    )

    matched_symbols = [g for g in df.columns if g in symbol_to_ens]

    if len(matched_symbols) == 0:
        raise ValueError(
            "Validation gene symbol이 mapping file과 하나도 매칭되지 않았습니다.\n"
            "ad_val.var_names가 정말 gene symbol인지 확인하세요."
        )

    print(
        f"   - Gene symbol -> Ensembl mapping: "
        f"{len(matched_symbols):,} / {df.shape[1]:,} validation genes matched "
        f"({len(matched_symbols) / df.shape[1]:.1%})"
    )

    df = df[matched_symbols]
    df.columns = [symbol_to_ens[g] for g in matched_symbols]

    n_before = df.shape[1]

    if aggregate == "mean":
        df_ens = df.T.groupby(level=0).mean().T
    elif aggregate == "sum":
        df_ens = df.T.groupby(level=0).sum().T
    else:
        raise ValueError("aggregate는 'mean' 또는 'sum'이어야 합니다.")

    n_after = df_ens.shape[1]

    if n_before != n_after:
        print(
            f"   - Duplicated Ensembl IDs aggregated: "
            f"{n_before:,} -> {n_after:,}"
        )

    return df_ens


def get_model_coefficients(model, feature_names):
    """
    ElasticNet/Lasso 계수 추출.
    Pipeline 형태까지 일부 대응.
    """
    if hasattr(model, "coef_"):
        coeffs = model.coef_

    elif hasattr(model, "named_steps"):
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, "coef_"):
            coeffs = last_step.coef_
        else:
            raise AttributeError("Pipeline 마지막 모델에 coef_가 없습니다.")

    else:
        raise AttributeError("현재 모델에서 coef_를 찾을 수 없습니다.")

    coeffs = np.asarray(coeffs).ravel()

    if len(coeffs) != len(feature_names):
        raise ValueError(
            f"Coefficient length와 feature 개수가 다릅니다: "
            f"{len(coeffs)} vs {len(feature_names)}"
        )

    return pd.Series(coeffs, index=feature_names)


def apply_scaling(X_val, scaler_path=None, scaler_mode="validation"):
    """
    Scaling 적용 함수.

    scaler_mode == "validation":
        validation cohort 내부에서 StandardScaler를 새로 fit_transform.

    scaler_mode == "train":
        저장된 training scaler를 load해서 transform.
    """

    if scaler_mode == "validation":
        val_scaler = StandardScaler()
        X_val_scaled = val_scaler.fit_transform(X_val)

        print(
            "   - Applied validation-specific StandardScaler "
            "(fit_transform on validation data)."
        )

        return X_val_scaled

    elif scaler_mode == "train":
        if scaler_path is None or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Training scaler file not found: {scaler_path}"
            )

        scaler = joblib.load(scaler_path)
        X_val_scaled = scaler.transform(X_val)

        print(
            "   - Applied saved training StandardScaler "
            "(transform only)."
        )

        return X_val_scaled

    else:
        raise ValueError("SCALER_MODE은 'validation' 또는 'train'이어야 합니다.")


# ==============================================================
# Main Validation Pipeline
# ==============================================================

def main():
    summary_metrics = []

    print(f"🚀 Starting Validation for Model: {TARGET_MODEL}")
    print(f"📂 Model Dir: {MODEL_DIR}")
    print(f"📂 Data Dir: {VAL_DATA_DIR}")
    print(f"📂 Gene Map: {GENE_MAP_PATH}")
    print(f"⚙️  Scaler Mode: {SCALER_MODE}")
    print("💾 Output filenames: original naming scheme\n")

    for ct in CELLTYPES:
        print("=" * 70)
        print(f"📌 Processing Celltype: {ct}")

        # ----------------------------------------------------------
        # 1. 모델 및 메타데이터 파일 경로
        # ----------------------------------------------------------
        base_filename = f"{ct}_{TARGET_MODEL}"

        model_path = os.path.join(MODEL_DIR, f"{base_filename}.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{base_filename}_scaler.pkl")
        feature_path = os.path.join(MODEL_DIR, f"{base_filename}_features.json")

        # validation scaler mode에서는 saved scaler가 필수 아님
        required_files = [model_path, feature_path]

        if SCALER_MODE == "train":
            required_files.append(scaler_path)

        missing_files = [p for p in required_files if not os.path.exists(p)]

        if missing_files:
            print(f"   [Skip] Missing required files for {ct}")
            for p in missing_files:
                print(f"      missing: {p}")
            continue

        if SCALER_MODE == "validation":
            if os.path.exists(scaler_path):
                print(
                    "   - Note: saved scaler exists but will NOT be used "
                    "because SCALER_MODE='validation'."
                )
            else:
                print(
                    "   - Note: saved scaler not found, but it is not required "
                    "because SCALER_MODE='validation'."
                )

        # ----------------------------------------------------------
        # 2. 모델, feature 로드
        # ----------------------------------------------------------
        model = joblib.load(model_path)

        with open(feature_path, "r") as f:
            feature_meta = json.load(f)

        train_features = feature_meta["feature_names"]

        # 모델 feature가 Ensembl version을 갖고 있을 수 있으므로 제거
        train_features = [strip_ensembl_version(g) for g in train_features]

        print(f"   - Loaded model: {os.path.basename(model_path)}")
        print(f"   - Number of model features: {len(train_features):,}")
        print(f"   - Example model features: {train_features[:5]}")

        # ----------------------------------------------------------
        # 3. Validation 데이터 로드
        # ----------------------------------------------------------
        val_file_path = os.path.join(VAL_DATA_DIR, f"validation_{ct}.h5ad")

        if not os.path.exists(val_file_path):
            print(f"   [Skip] Validation data not found: {val_file_path}")
            continue

        ad_val = sc.read_h5ad(val_file_path)

        print(
            f"   - Loaded validation data: "
            f"{ad_val.shape[0]:,} cells x {ad_val.shape[1]:,} genes "
            f"from {os.path.basename(val_file_path)}"
        )

        # 필수 obs 컬럼 확인
        required_obs_cols = ["age", STATE_COL]

        if DONOR_COL:
            required_obs_cols.append(DONOR_COL)

        missing_obs_cols = [
            c for c in required_obs_cols
            if c not in ad_val.obs.columns
        ]

        if missing_obs_cols:
            print(f"   [Skip] Missing obs columns: {missing_obs_cols}")
            print(f"   - Available obs columns: {ad_val.obs.columns.tolist()}")
            continue

        # ----------------------------------------------------------
        # 4. Gene symbol -> Ensembl ID 변환
        # ----------------------------------------------------------
        try:
            ad_val_df = convert_symbol_to_ensembl_df(
                adata=ad_val,
                mapping_path=GENE_MAP_PATH,
                aggregate=DUPLICATE_GENE_AGG
            )

        except Exception as e:
            print(f"   [Skip] Gene ID conversion failed for {ct}: {e}")
            continue

        # ----------------------------------------------------------
        # 5. Feature alignment
        # ----------------------------------------------------------
        matched_features = [
            g for g in train_features
            if g in ad_val_df.columns
        ]

        missing_genes = [
            g for g in train_features
            if g not in ad_val_df.columns
        ]

        match_ratio = len(matched_features) / len(train_features)

        print(
            f"   - Matched model features: "
            f"{len(matched_features):,} / {len(train_features):,} "
            f"({match_ratio:.1%})"
        )

        if len(missing_genes) > 0:
            print(
                f"   - Warning: {len(missing_genes):,} model features missing "
                "in validation data. Filling with 0."
            )

        if match_ratio < MIN_FEATURE_MATCH_RATIO:
            print(
                f"   [Skip] Too few matched features for {ct}: "
                f"{match_ratio:.1%} < {MIN_FEATURE_MATCH_RATIO:.1%}"
            )

            missing_save_path = os.path.join(
                VAL_RESULT_DIR,
                f"{ct}_{TARGET_MODEL}_missing_features.txt"
            )

            with open(missing_save_path, "w") as f:
                for g in missing_genes:
                    f.write(f"{g}\n")

            print(f"   - Missing feature list saved to: {missing_save_path}")
            continue

        # missing feature는 0으로 채움
        if missing_genes:
            zeros_df = pd.DataFrame(
                0,
                index=ad_val_df.index,
                columns=missing_genes
            )

            ad_val_df = pd.concat([ad_val_df, zeros_df], axis=1)

        # 학습 때 사용한 feature 순서와 동일하게 정렬
        X_val = ad_val_df.loc[:, train_features]

        # ----------------------------------------------------------
        # 6. Scaling
        # ----------------------------------------------------------
        X_val_scaled = apply_scaling(
            X_val=X_val,
            scaler_path=scaler_path,
            scaler_mode=SCALER_MODE
        )

        X_val_final = pd.DataFrame(
            X_val_scaled,
            index=X_val.index,
            columns=X_val.columns
        )

        # ----------------------------------------------------------
        # 7. Prediction
        # ----------------------------------------------------------
        y_pred = model.predict(X_val_final)

        # ----------------------------------------------------------
        # 8. Contribution analysis
        # ----------------------------------------------------------
        print(f"   - Calculating feature contribution for {ct}...")

        try:
            coef_series = get_model_coefficients(
                model=model,
                feature_names=X_val_final.columns
            )

            nonzero_coef = coef_series[coef_series != 0]

            if len(nonzero_coef) == 0:
                print("   - [Skip] No non-zero coefficients found.")

            else:
                top_genes = (
                    nonzero_coef
                    .abs()
                    .sort_values(ascending=False)
                    .head(50)
                    .index
                )

                # contribution = scaled expression * coefficient
                contrib_df = X_val_final[top_genes].multiply(
                    coef_series[top_genes],
                    axis=1
                )

                contrib_df[STATE_COL] = ad_val.obs[STATE_COL].values

                avg_contrib = contrib_df.groupby(STATE_COL).mean()

                # 기존 파일명 유지
                save_path_contrib = os.path.join(
                    VAL_RESULT_DIR,
                    f"{ct}_{TARGET_MODEL}_top50_contribution.csv"
                )

                avg_contrib.to_csv(save_path_contrib)

                print(
                    f"   - ✅ Saved contribution analysis to "
                    f"{save_path_contrib}"
                )

        except Exception as e:
            print(f"   - [Skip] Contribution analysis failed: {e}")

        # ----------------------------------------------------------
        # 9. 결과 DataFrame 생성
        # ----------------------------------------------------------
        y_true = ad_val.obs["age"].astype(float).values

        res_df = pd.DataFrame({
            "cell_id": ad_val.obs_names,
            "donor": (
                ad_val.obs[DONOR_COL].values
                if DONOR_COL
                else ["Unknown"] * len(y_pred)
            ),
            "state": ad_val.obs[STATE_COL].values,
            "true_age": y_true,
            "pred_age": y_pred
        })

        # 기존 파일명 유지
        save_path_cell = os.path.join(
            VAL_RESULT_DIR,
            f"{ct}_{TARGET_MODEL}_validation_predictions_cell.csv"
        )

        res_df.to_csv(save_path_cell, index=False)

        print(
            f"   - ✅ Saved cell-level predictions to "
            f"{save_path_cell}"
        )

        # ----------------------------------------------------------
        # 10. Donor-level aggregation
        # ----------------------------------------------------------
        if DONOR_COL:
            res_df["donor"] = res_df["donor"].astype(str)
            res_df["state"] = res_df["state"].astype(str)

            donor_df = (
                res_df
                .groupby(["donor", "state"], observed=True)
                .agg({
                    "true_age": "mean",
                    "pred_age": "median"
                })
                .reset_index()
            )

            # 기존 파일명 유지
            save_path_donor = os.path.join(
                VAL_RESULT_DIR,
                f"{ct}_{TARGET_MODEL}_validation_predictions_donor_median.csv"
            )

            donor_df.to_csv(save_path_donor, index=False)

            print(
                f"   - ✅ Saved donor-level predictions to "
                f"{save_path_donor}"
            )

        else:
            donor_df = pd.DataFrame()

        # ----------------------------------------------------------
        # 11. Metric calculation by state
        # ----------------------------------------------------------
        states = ["All"] + sorted(
            res_df["state"].astype(str).unique().tolist()
        )

        for lvl_name, data_df in [
            ("Cell_Level", res_df),
            ("Donor_Level", donor_df)
        ]:
            if data_df.empty:
                continue

            data_df = data_df.copy()
            data_df["state"] = data_df["state"].astype(str)

            for st in states:
                if st == "All":
                    subset = data_df
                else:
                    subset = data_df[data_df["state"] == st]

                if len(subset) < 2:
                    continue

                mae, rmse, r2, pr = calculate_metrics(
                    subset["true_age"],
                    subset["pred_age"]
                )

                summary_metrics.append({
                    "Cell_Type": ct,
                    "Model": TARGET_MODEL,
                    "Scaler_Mode": SCALER_MODE,
                    "Level": lvl_name,
                    "State": st,
                    "N_Samples": len(subset),
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "Pearson_R": pr,
                    "Matched_Features": len(matched_features),
                    "Total_Features": len(train_features),
                    "Feature_Match_Ratio": match_ratio
                })

    # --------------------------------------------------------------
    # 12. Summary 저장
    # --------------------------------------------------------------
    if summary_metrics:
        summary_df = pd.DataFrame(summary_metrics)

        summary_df = summary_df.sort_values(
            by=["Cell_Type", "Level", "State"]
        )

        # 기존 파일명 유지
        summary_save_path = os.path.join(
            VAL_RESULT_DIR,
            f"{TARGET_MODEL}_validation_summary_by_state.csv"
        )

        summary_df.to_csv(summary_save_path, index=False)

        print("\n" + "=" * 70)
        print("✅ Validation Finished.")
        print(f"📄 Summary saved to: {summary_save_path}")
        print(summary_df)

    else:
        print("\n❌ No metrics calculated.")


if __name__ == "__main__":
    main()
