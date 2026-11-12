# Curated publication source. See docs/SOURCE_MANIFEST.json.

# Source notebook cell 0
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
import anndata as ad
import os
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", 
    "NUMEXPR_NUM_THREADS": "1"
})

sc.settings.verbosity = 3
sc.set_figure_params(dpi=100, frameon=False)

# Source notebook cell 1
adata = sc.read_h5ad("data/covid_pp_annotation.h5ad")

# Source notebook cell 2
adata.layers['raw_counts'] = adata.X.copy()

# Source notebook cell 3
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy import sparse

# -------------------------
# 1. 전역 설정 (Global Config)
# -------------------------
RANDOM_SEED = 260312
np.random.seed(RANDOM_SEED)

PSEUDOCELL_SAMPLE_SIZE = 20      # K (합칠 세포 수)
PSEUDOCELLS_PER_DONOR = 100       # [수정] 모든 도너당 고정적으로 생성할 슈도셀 수
MIN_CELLS_PER_DONOR = 30         # 최소 세포 기준
LAYER_NAME = 'raw_counts'        # 합계를 구할 레이어

# 컬럼명 설정 (AIDA 기준)
COL_DONOR = 'sampleID'
COL_AGE = 'age'
COL_SEX = 'sex' 
COL_CELLTYPE = 'Modeling_annotation'
COL_STATE = 'state'              # Normal, Disease 등 상태 정보 

TARGET_CELL_TYPES = ["NK", "CD8T", "CD4T", "Mono"]

# -------------------------
# 2. 슈도셀 생성 코어 로직 (Fixed Sampling)
# -------------------------
def make_pseudo_cells(adata, cell_type_value, cell_type_col, 
                      donor_col=COL_DONOR, age_col=COL_AGE, sex_col=COL_SEX, 
                      state_col=COL_STATE, layer_name=LAYER_NAME):
    """
    단일 Cell Type에 대한 슈도셀 생성.
    - population(인종) 관련 로직 제거됨.
    - 모든 도너로부터 고정된 개수(100개)의 슈도셀을 추출하여 데이터 균형 확보.
    """
    
    if cell_type_col not in adata.obs:
        print(f"[Error] Column '{cell_type_col}' not found.")
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
        
        if n_real < MIN_CELLS_PER_DONOR:
            continue
            
        donor_global_indices = target_indices[donor_mask]

        # 모든 도너에게서 고정된 개수(PSEUDOCELLS_PER_DONOR)만큼 추출
        for i in range(PSEUDOCELLS_PER_DONOR):
            sampled_idx = np.random.choice(
                donor_global_indices, 
                size=PSEUDOCELL_SAMPLE_SIZE, 
                replace=True # 고정 개수 생성을 위해 복원 추출 허용
            )

            # Sum aggregation (Sparse/Dense 지원)
            X_subset = adata.X[sampled_idx] if (layer_name == "X" or layer_name is None) else adata.layers[layer_name][sampled_idx]
            sum_vec = np.asarray(X_subset.sum(axis=0)).ravel() if sparse.issparse(X_subset) else np.sum(X_subset, axis=0)
            
            rows.append(sum_vec)
            metas.append({
                "pseudocell_id": f"{cell_type_value}__{donor}__pc{i+1}",
                "sampleID": donor,
                "cell_type": cell_type_value,
                "n_real_cells": n_real
            })

    if not rows:
        return None, None

    pseudo_X = np.vstack(rows)
    meta_df = pd.DataFrame(metas)
    
    # 메타데이터 매핑 (Age, Sex, State만 포함)
    cols_to_map = [donor_col, age_col]
    for c in [state_col, sex_col]:
        if c in adata.obs.columns:
            cols_to_map.append(c)

    donor_meta_map = adata.obs[cols_to_map].drop_duplicates(subset=donor_col).set_index(donor_col)
    
    # 데이터프레임에 정보 병합
    meta_df['age'] = meta_df['sampleID'].map(donor_meta_map[age_col])
    if state_col in donor_meta_map.columns:
        meta_df['state'] = meta_df['sampleID'].map(donor_meta_map[state_col])
    if sex_col in donor_meta_map.columns:
        meta_df['sex'] = meta_df['sampleID'].map(donor_meta_map[sex_col])
    
    meta_df.set_index('pseudocell_id', inplace=True)
    df_pseudo = pd.DataFrame(pseudo_X, columns=adata.var_names, index=meta_df.index)
    
    return df_pseudo, meta_df

# -------------------------
# 3. 파이프라인 실행 함수
# -------------------------
def run_validation_pipeline(adata, save_dir, target_types=TARGET_CELL_TYPES):
    os.makedirs(save_dir, exist_ok=True)
    print(f"=== Starting AIDA Validation Pipeline ===")

    for cell_type in target_types:
        print(f"\n>> Processing Validation: {cell_type}")
        df_X, df_meta = make_pseudo_cells(adata, cell_type, COL_CELLTYPE)

        if df_X is None:
            continue

        val_adata = anndata.AnnData(X=df_X, obs=df_meta)
        
        # 정규화 및 로그 변환 (학습 시와 동일한 기준 적용)
        sc.pp.normalize_total(val_adata, target_sum=1e4)
        sc.pp.log1p(val_adata)

        save_path = os.path.join(save_dir, f"validation_{cell_type}.h5ad")
        val_adata.write_h5ad(save_path)
        print(f"   [Saved] {save_path} (n_pseudocells={val_adata.n_obs})")

    print("\n=== COMPLETE ===")

# Source notebook cell 4
# -------------------------
# 4. 실행
# -------------------------
run_validation_pipeline(adata, "pseudocells_validation")
