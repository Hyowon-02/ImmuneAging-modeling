# Curated publication source. See docs/SOURCE_MANIFEST.json.

# Source notebook cell 2
import os
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
from scipy import sparse
import anndata

# -------------------------
# 1. 전역 설정 (Global Config)
# -------------------------

RANDOM_SEED = 260601
np.random.seed(RANDOM_SEED)

PSEUDOCELL_SAMPLE_SIZE = 20      # K (하나의 슈도셀을 만드는 데 합칠 세포 수)
MIN_CELLS_PER_DONOR = 30         # 최소 세포 기준 (이보다 적으면 제외)
LAYER_NAME = 'raw_counts'

# 경로 설정
PSEUDO_SAVE_DIR = "data/pseudocell_data_oversampling/"
INPUT_RDS_DIR = "data/train_test"

os.makedirs(PSEUDO_SAVE_DIR, exist_ok=True)

# 컬럼명 설정
COL_DONOR = 'donor_id'
COL_AGE = 'age'
COL_SEX = 'sex' 
COL_CELLTYPE = 'Modeling_annotation'

# 통합 Config 리스트
CELLTYPE_LIST = ['NK', "CD4T", "CD8T",'Mono']

# -------------------------
# 2. 헬퍼 함수 (Sum)
# -------------------------

def safe_sum_of_rows_layer(adata_obj, row_idx, layer_name=LAYER_NAME):
    """합계(Sum) 계산 함수 (Sparse/Dense 지원)"""
    if layer_name == "X" or layer_name is None:
        X = adata_obj.X[row_idx]
    elif layer_name in adata_obj.layers:
        X = adata_obj.layers[layer_name][row_idx]
    else:
        raise ValueError(f"Layer '{layer_name}' not found in adata.")

    if sparse.issparse(X):
        sum_vec = X.sum(axis=0)
        return np.asarray(sum_vec).ravel()
    else:
        return np.sum(X, axis=0)

# -------------------------
# 3. 슈도셀 생성 코어 로직
# ------------------------
def make_pseudo_cells(adata, cell_type_value, split_name, cell_type_col=COL_CELLTYPE, 
                      donor_col=COL_DONOR, age_col=COL_AGE, sex_col=COL_SEX, 
                      layer_name=LAYER_NAME):
    """
    모든 도너로부터 슈도셀 생성 
    - train: 50대 이상 200개, 그 외 100개 (오버샘플링)
    - test: 모든 연령대 동일하게 100개
    """

    if cell_type_col not in adata.obs:
        return None, None

    mask = (adata.obs[cell_type_col] == cell_type_value)
    if not mask.any():
        return None, None

    subset_obs = adata.obs[mask]
    target_indices = np.where(mask)[0] 

    donors = subset_obs[donor_col].unique()
    rows, metas = [], []

    for donor in donors:
        donor_mask = (subset_obs[donor_col] == donor)
        n_real = donor_mask.sum()
        if n_real < MIN_CELLS_PER_DONOR: continue

        # 해당 도너의 연령 확인
        donor_age = subset_obs.loc[subset_obs[donor_col] == donor, age_col].iloc[0]

        # [수정 로직] split_name에 따른 슈도셀 생성 개수 결정
        if split_name == "train":
            if donor_age >= 50:
                current_n_pseudo = 200  # 50대 이상 오버샘플링
            else:
                current_n_pseudo = 100
        else:
            # test 셋은 연령에 관계없이 동일하게 100개 생성
            current_n_pseudo = 100

        donor_global_indices = target_indices[donor_mask]

        for i in range(current_n_pseudo):
            sampled_idx = np.random.choice(
                donor_global_indices, 
                size=PSEUDOCELL_SAMPLE_SIZE, 
                replace=True 
            )

            sum_vec = safe_sum_of_rows_layer(adata, sampled_idx, layer_name)
            rows.append(sum_vec)

            metas.append({
                "pseudocell_id": f"{cell_type_value}_{split_name}_{donor}_pc{i+1}",
                "donor_id": donor,
                "cell_type": cell_type_value,
                "n_real_cells": n_real,
                "age": donor_age,
                "sex": subset_obs.loc[subset_obs[donor_col] == donor, sex_col].iloc[0] if sex_col in subset_obs.columns else "Unknown",
                "split": split_name
            })

    if not rows:
        return None, None

    pseudo_X = np.vstack(rows)
    meta_df = pd.DataFrame(metas)
    meta_df.set_index('pseudocell_id', inplace=True)
    
    df_pseudo = pd.DataFrame(pseudo_X, columns=adata.var_names, index=meta_df.index)

    return df_pseudo, meta_df

# -------------------------
# 4. 파이프라인 실행 함수
# -------------------------
def run_processing_pipeline(ct_list, save_dir):

    summary_records = []

    print(f"=== STARTING PIPELINE: PSEUDO CELL GENERATION ===")

    for ct in tqdm(ct_list):
        print(f"\n>> JOB: {ct}")
        try:
            path_train = os.path.join(INPUT_RDS_DIR, f"{ct}_train.h5ad")
            path_test = os.path.join(INPUT_RDS_DIR, f"{ct}_test.h5ad")

            adata_train = sc.read_h5ad(path_train)
            adata_test = sc.read_h5ad(path_test)

        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        n_real_total = adata_train.n_obs + adata_test.n_obs
        datasets = {"train": adata_train, "test": adata_test}
        
        saved_counts = {"train": 0, "test": 0}

        for split_name, adata_obj in datasets.items():
            df_X, df_meta = make_pseudo_cells(
                adata_obj, 
                ct, 
                split_name=split_name,  # split_name 전달
                cell_type_col=COL_CELLTYPE, 
                donor_col=COL_DONOR, 
                age_col=COL_AGE, 
                sex_col=COL_SEX, 
                layer_name=LAYER_NAME 
            )
            
            if df_X is not None:
                # AnnData 생성 및 저장
                out_adata = anndata.AnnData(X=df_X, obs=df_meta)
                save_path = os.path.join(save_dir, f"{ct}_{split_name}_pseudo.h5ad")
                out_adata.write_h5ad(save_path)
                
                saved_counts[split_name] = out_adata.n_obs

        if sum(saved_counts.values()) > 0:
            print(f"   [Saved] Train: {saved_counts['train']}, Test: {saved_counts['test']}")
            summary_records.append({
                "cell_type": ct,
                "n_real_total": n_real_total,
                "n_pseudo_train": saved_counts['train'],
                "n_pseudo_test": saved_counts['test']
            })
        else:
            print(f"   [Warn] No outputs generated for {ct}")

    if summary_records:
        print("\n=== SUMMARY ===")
        print(pd.DataFrame(summary_records))
    print("\n=== COMPLETE ===")

# -------------------------
# 5. 실행
# -------------------------
if __name__ == "__main__":
    run_processing_pipeline(CELLTYPE_LIST, PSEUDO_SAVE_DIR)
