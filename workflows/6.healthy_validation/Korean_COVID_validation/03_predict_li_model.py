# Input is loaded from a raw-count file below; normalization occurs once.
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

GENE_MAPPING_PATH = "../../raw_data/gene_id_mapping_full.csv"
SEED = 251219
import scanpy as sc
from scipy import sparse
adata = sc.read_h5ad("data/covid_pp_annotation.h5ad")
values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X).ravel()
if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
    raise ValueError("Li input must be raw UMI counts, not normalized/scaled values.")
if np.any(np.asarray(adata.X.sum(axis=1)).ravel() <= 0):
    raise ValueError("Remove zero-library cells before normalization.")
adata.X = adata.X.astype(np.float64)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[adata.obs["state"].astype(str).eq("Normal")].copy()
if adata.n_obs == 0:
    raise ValueError("No Normal controls in input.")


# ==============================================================
# 1. Helper Function (작성하신 코드와 동일)
# ==============================================================
def calculate_metrics(y_true, y_pred):
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
    
    if np.std(y_pred) < 1e-9:
        pr = 0.0
    else:
        pr, _ = pearsonr(y_true, y_pred)
        
    return mae, rmse, r2, pr


def generate_pseudocells_py(adata, donor_col, ct_col, n_cells=15, n_pseudo=100):
    pseudo_results = {}
    rng = np.random.default_rng(SEED)
    valid_cts = ['NK', 'CD8T', 'CD4T','Mono']
    
    for ct in valid_cts:
        print(f"Processing {ct}...")
        ct_adata = adata[adata.obs[ct_col] == ct]
        donors = ct_adata.obs[donor_col].unique()
        ct_dict = {}
        
        for donor in donors:
            donor_adata = ct_adata[ct_adata.obs[donor_col] == donor]
            
            # 세포 수가 적어도 복원추출로 pseudocell 생성
            if donor_adata.n_obs == 0: 
                continue
            
            expr_matrix = donor_adata.X
            if hasattr(expr_matrix, "toarray"): 
                expr_matrix = expr_matrix.toarray()
            
            # 세포별 log-normalized 발현값의 평균을 사용합니다.
            # 공개 R 구현: 세포 수 <=15일 때만 복원추출합니다.
            p_vectors = [
                expr_matrix[rng.choice(donor_adata.n_obs, n_cells, replace=(donor_adata.n_obs <= n_cells))].mean(axis=0) 
                for _ in range(n_pseudo)
            ]
            ct_dict[donor] = np.array(p_vectors, dtype=np.float64)
            
        pseudo_results[ct] = ct_dict
    return pseudo_results
# ==============================================================
# 2. Main Validation Function for External Model
# ==============================================================
def validate_li_et_al_model(
    pseudo_data,
    coef_df,
    adata,
    val_result_dir,
    target_model="Li_et_al",
    donor_col="sampleID",
    state_col="state",
    gene_mapping=GENE_MAPPING_PATH,
):
    """
    adata.var_names와 adata.var의 gene ID/symbol 컬럼을 확인해 모델 유전자를 매칭합니다.

    gene_mapping:
        필요하면 gene_id_mapping_full.csv의 경로 또는 DataFrame을 전달합니다.
        Ensembl ID 컬럼과 gene symbol 컬럼을 자동으로 탐색합니다.

    기본 donor 결과: median (논문 Methods).
    추가 donor 결과: round(mean) (공개 R 패키지).
    """
    import os
    import re
    import warnings
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    os.makedirs(val_result_dir, exist_ok=True)

    def clean_gene(value):
        if pd.isna(value):
            return ""
        value = str(value).strip()
        # Ensembl 버전 번호만 제거합니다.
        return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", value)

    def clean_ct(value):
        value = re.sub(r"[^A-Z0-9]", "", str(value).upper())
        return {"MONOCYTE": "MONO", "MONOCYTES": "MONO"}.get(value, value)

    coef_df = coef_df.copy()
    if "gene_clean" not in coef_df.columns:
        if "gene" not in coef_df.columns:
            raise ValueError("계수 표에 gene 또는 gene_clean 컬럼이 필요합니다.")
        coef_df = coef_df.rename(columns={"gene": "gene_clean"})

    coef_df["gene_clean"] = coef_df["gene_clean"].map(clean_gene)
    coef_df["coef"] = pd.to_numeric(coef_df["coef"], errors="raise")
    coef_df["_ct"] = coef_df["celltype"].map(clean_ct)

    # 하나의 ID/symbol이 여러 열에 대응하면 임의로 선택하지 않습니다.
    aliases = {}

    def add_alias(name, position):
        name = clean_gene(name)
        if name:
            aliases.setdefault(name, set()).add(int(position))

    for position, name in enumerate(adata.var_names):
        add_alias(name, position)

    gene_columns = []
    for column in adata.var.columns:
        key = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if (
            key in {
                "gene", "genes", "geneid", "geneids",
                "genename", "genenames", "genesymbol", "genesymbols",
                "symbol", "symbols", "ensembl", "ensemblid",
                "ensemblgeneid", "featureid", "featurename",
                "featurenames",
            }
            or "ensembl" in key
        ):
            gene_columns.append(column)
            for position, name in enumerate(adata.var[column]):
                add_alias(name, position)

    # 별도 mapping 파일을 제공한 경우 ID ↔ symbol 연결을 추가합니다.
    if gene_mapping is not None:
        mapping = (
            gene_mapping.copy()
            if isinstance(gene_mapping, pd.DataFrame)
            else pd.read_csv(gene_mapping)
        )

        id_columns = [
            c for c in mapping.columns
            if mapping[c].astype(str).str.match(r"^ENSG\d+", na=False).any()
        ]
        symbol_columns = [
            c for c in mapping.columns
            if re.sub(r"[^a-z0-9]", "", str(c).lower())
            in {
                "symbol", "genesymbol", "hgncsymbol",
                "genename", "externalgenename",
            }
        ]

        if not id_columns or not symbol_columns:
            raise ValueError(
                "mapping의 Ensembl/symbol 컬럼을 식별하지 못했습니다. "
                f"현재 컬럼: {list(mapping.columns)}"
            )

        # mapping을 확장하기 전의 실제 데이터 유전자 기준으로 연결합니다.
        base_aliases = {k: set(v) for k, v in aliases.items()}
        for _, row in mapping.iterrows():
            names = {
                clean_gene(row[c])
                for c in id_columns + symbol_columns
            } - {""}
            positions = set()
            for name in names:
                positions.update(base_aliases.get(name, set()))
            for name in names:
                if positions:
                    aliases.setdefault(name, set()).update(positions)

    obs = adata.obs[[donor_col, state_col, "age"]].copy()
    if obs.isna().any().any():
        raise ValueError("donor/state/age에 결측값이 있습니다.")

    obs[donor_col] = obs[donor_col].astype(str)
    obs[state_col] = obs[state_col].astype(str)
    obs["age"] = pd.to_numeric(obs["age"], errors="raise")

    if (
        obs.groupby(donor_col)[[state_col, "age"]]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("동일 donor에 서로 다른 state 또는 age가 있습니다.")

    meta_map = obs.drop_duplicates(donor_col).set_index(donor_col)
    summary_rows = []

    print("유전자 매칭에 사용한 var 컬럼:", gene_columns)

    for ct, donors_dict in pseudo_data.items():
        ct_coefs = coef_df.loc[coef_df["_ct"].eq(clean_ct(ct))].copy()
        if ct_coefs.empty:
            warnings.warn(f"{ct}: 해당 세포형 계수가 없어 건너뜁니다.")
            continue

        intercept_rows = ct_coefs.loc[
            ct_coefs["gene_clean"].eq("(Intercept)"), "coef"
        ]
        if len(intercept_rows) != 1:
            raise ValueError(f"{ct}: 절편이 정확히 하나여야 합니다.")

        intercept = float(intercept_rows.iloc[0])
        genes = ct_coefs.loc[
            ~ct_coefs["gene_clean"].eq("(Intercept)")
        ].copy()

        # 0 계수 유전자는 예측에 영향을 주지 않습니다.
        genes = genes.loc[genes["coef"].ne(0)].copy()
        if genes.empty:
            raise ValueError(f"{ct}: 0이 아닌 유전자 계수가 없습니다.")
        if genes["gene_clean"].duplicated().any():
            raise ValueError(f"{ct}: 중복된 모델 유전자 ID가 있습니다.")
        if not np.isfinite(genes["coef"]).all() or not np.isfinite(intercept):
            raise ValueError(f"{ct}: 계수 또는 절편에 비정상 값이 있습니다.")

        match_rows = []
        model_positions = []
        data_positions = []

        for j, gene in enumerate(genes["gene_clean"]):
            positions = aliases.get(gene, set())
            if len(positions) > 1:
                raise ValueError(
                    f"{ct}: {gene}이 데이터의 여러 유전자 열과 매칭됩니다. "
                    "중복 ID/symbol을 먼저 확인해야 합니다."
                )

            position = next(iter(positions)) if positions else None
            match_rows.append({
                "model_gene": gene,
                "matched": position is not None,
                "data_gene": (
                    str(adata.var_names[position])
                    if position is not None else ""
                ),
            })
            if position is not None:
                model_positions.append(j)
                data_positions.append(position)

        coverage = pd.DataFrame(match_rows)
        coverage.to_csv(
            os.path.join(val_result_dir, f"{ct}_gene_matching.csv"),
            index=False,
        )

        n_matched = len(data_positions)
        print(f"{ct}: 모델 유전자 {len(genes)}개 중 {n_matched}개 매칭")

        if n_matched == 0:
            raise ValueError(
                f"{ct}: 유전자 매칭이 0개입니다.\n"
                f"모델 ID 예시: {genes['gene_clean'].head().tolist()}\n"
                f"데이터 ID 예시: {list(adata.var_names[:5])}\n"
                "ID↔symbol 정보가 adata.var에 없다면 함수 호출에 "
                "gene_mapping='gene_id_mapping_full.csv의 실제 경로'를 "
                "추가하세요."
            )

        if n_matched < len(genes):
            warnings.warn(
                f"{ct}: 누락 유전자 {len(genes) - n_matched}개는 "
                "공식 구현처럼 0으로 처리했습니다. gene_matching.csv를 확인하세요."
            )

        beta = genes["coef"].to_numpy(dtype=float)
        records = []

        for donor, matrix in donors_dict.items():
            donor = str(donor)
            if donor not in meta_map.index:
                raise ValueError(f"{ct}: {donor}의 메타데이터가 없습니다.")

            matrix = np.asarray(matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != adata.n_vars:
                raise ValueError(
                    f"{ct}/{donor}: pseudocell 열 수가 adata.n_vars와 다릅니다. "
                    "pseudocell 생성 후 adata의 유전자 순서/구성을 바꾸면 안 됩니다."
                )

            # 매칭되지 않은 유전자는 0으로 남깁니다.
            aligned = np.zeros((matrix.shape[0], len(genes)), dtype=float)
            aligned[:, model_positions] = matrix[:, data_positions]
            predictions = aligned @ beta + intercept

            if not np.isfinite(predictions).all():
                raise ValueError(f"{ct}/{donor}: 예측값에 NaN/Inf가 있습니다.")

            records.extend({
                "cell_id": f"{donor}_pseudo_{i + 1}",
                "donor": donor,
                "state": meta_map.at[donor, state_col],
                "true_age": float(meta_map.at[donor, "age"]),
                "pred_age": float(prediction),
            } for i, prediction in enumerate(predictions))

        if not records:
            continue

        cell_df = pd.DataFrame(records)
        prefix = os.path.join(val_result_dir, f"{ct}_{target_model}")

        cell_df.to_csv(
            f"{prefix}_validation_predictions_cell.csv", index=False
        )

        # 논문 Methods 기준: donor median.
        donor_df = cell_df.groupby(
            ["donor", "state"], observed=True, as_index=False
        ).agg(true_age=("true_age", "first"), pred_age=("pred_age", "median"))

        donor_df.to_csv(
            f"{prefix}_validation_predictions_donor_median.csv",
            index=False,
        )

        # 공개 R 패키지 기준: round(mean).
        package_df = cell_df.groupby(
            ["donor", "state"], observed=True, as_index=False
        ).agg(true_age=("true_age", "first"), pred_age=("pred_age", "mean"))
        package_df["pred_age_unrounded_mean"] = package_df["pred_age"]
        package_df["pred_age"] = np.rint(package_df["pred_age"])
        package_df.to_csv(
            f"{prefix}_validation_predictions_donor_mean_rounded.csv",
            index=False,
        )

        for level, frame in [
            ("Cell_Level", cell_df),
            ("Donor_Level", donor_df),
            ("Donor_Level_Package", package_df),
        ]:
            for state in ["All"] + sorted(frame["state"].unique().tolist()):
                subset = frame if state == "All" else frame.loc[
                    frame["state"].eq(state)
                ]
                if len(subset) < 2:
                    continue

                actual = subset["true_age"].to_numpy(dtype=float)
                predicted = subset["pred_age"].to_numpy(dtype=float)
                errors = predicted - actual
                denominator = np.sum((actual - actual.mean()) ** 2)

                summary_rows.append({
                    "Cell_Type": ct,
                    "Model": target_model,
                    "Level": level,
                    "State": state,
                    "N_Samples": len(subset),
                    "MAE": np.mean(np.abs(errors)),
                    "RMSE": np.sqrt(np.mean(errors ** 2)),
                    "R2": (
                        1 - np.sum(errors ** 2) / denominator
                        if denominator > 0 else np.nan
                    ),
                    "Pearson_R": (
                        pearsonr(actual, predicted)[0]
                        if np.std(actual) > 0 and np.std(predicted) > 0
                        else np.nan
                    ),
                })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Cell_Type", "Level", "State"])
        summary_df.to_csv(
            os.path.join(
                val_result_dir,
                f"{target_model}_validation_summary_by_state.csv",
            ),
            index=False,
        )

    return summary_df


# 매핑 CSV를 읽어 정확한 두 컬럼만 사용합니다.
gene_mapping_df = pd.read_csv(GENE_MAPPING_PATH, dtype=str)
gene_mapping_df.columns = gene_mapping_df.columns.str.strip()
required = {"ensembl_id", "gene_symbol"}
if not required.issubset(gene_mapping_df.columns):
    raise ValueError(f"매핑 CSV에 {required} 컬럼이 필요합니다.")
gene_mapping_df = gene_mapping_df[["ensembl_id", "gene_symbol"]].dropna().copy()
for column in gene_mapping_df.columns:
    gene_mapping_df[column] = gene_mapping_df[column].str.strip()
gene_mapping_df = gene_mapping_df.loc[
    gene_mapping_df["ensembl_id"].ne("") & gene_mapping_df["gene_symbol"].ne("")
].drop_duplicates()
print(f"Gene mapping loaded: {len(gene_mapping_df):,} pairs")

# 1. 파일 경로 지정 및 데이터 불러오기
coef_path = "../../raw_data/li_reference/model_coefficients.csv"
coef_df = pd.read_csv(coef_path)
# Coefficients must be exported from the published glmnet model at s='lambda.min'.

# 2. 기존 검증 함수와의 호환성을 위한 컬럼명 변경 ('gene' -> 'gene_clean')
if 'gene' in coef_df.columns:
    coef_df = coef_df.rename(columns={'gene': 'gene_clean'})

# 3. 데이터 구조 사전 점검 (선택 사항)
print("✅ Coefficient Data Loaded:")
print(coef_df.head())

# ==============================================================
# 4. 검증 파이프라인 실행
# ==============================================================
pseudo_data = generate_pseudocells_py(adata, donor_col='sampleID', ct_col='Modeling_annotation')

summary_df = validate_li_et_al_model(
    pseudo_data=pseudo_data, 
    coef_df=coef_df,  # 로드 및 전처리된 데이터프레임 전달
    adata=adata, 
    val_result_dir="Li_et_al_covid_validation_result",
    target_model="Li_et_al",
    donor_col='sampleID', 
    state_col='state',
    gene_mapping=gene_mapping_df,
)