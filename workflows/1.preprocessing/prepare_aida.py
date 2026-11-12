# Curated publication source. See docs/SOURCE_MANIFEST.json.

# Source notebook cell 0
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import multipletests
from scipy.stats import t as tdist
sc.settings.n_jobs = 32

# Source notebook cell 1
adata = sc.read_h5ad("../raw_data/AIDA_Freeze_v2.h5ad")

# Source notebook cell 2
are_all_integers = np.all(adata.raw.X.data == np.floor(adata.raw.X.data))
if are_all_integers:
    print("✅ 확인 완료: adata.raw.X의 모든 count 값은 정수입니다.")
else:
    print("⚠️ 발견: adata.raw.X에 정수가 아닌 count 값이 포함되어 있습니다.")

# Source notebook cell 3
country_key = 'Country' 
original_age_key = 'development_stage' 
# 새로 생성할 숫자 나이 컬럼
age_key = 'age'

adata.obs[age_key] =  adata.obs[original_age_key].str.split('-').str[0].astype(int)
print(adata.obs[[original_age_key, age_key]].head(10))

# 여러 조건을 '&' (AND) 연산자로 결합하여 한 번에 필터링
adata = adata[
    (~adata.obs['self_reported_ethnicity'].isin(['European', 'Indian'])) & 
    (adata.obs['Annotation_Level1'] != 'Platelet')
].copy()
adata.obs['self_reported_ethnicity'].value_counts()

# Source notebook cell 5
conditions = [
    adata.obs['Annotation_Level2'].str.contains('NK'),
    adata.obs['Annotation_Level2'].str.contains('CD4'),
    adata.obs['Annotation_Level2'].str.contains('CD8'),
    adata.obs['Annotation_Level2'].str.contains('Monocyte'),
    adata.obs['Annotation_Level2'].str.contains('B')
]

# 2. 각 조건에 해당할 때 부여할 값을 리스트로 정의합니다. (순서가 중요)
choices = ['NK', 'CD4T', 'CD8T', 'Mono', 'B']

# 3. np.select를 사용하여 'Modeling_annotation' 컬럼 생성
# conditions에 해당하지 않는 모든 경우는 default 값인 'else'가 부여됩니다.
adata.obs['Modeling_annotation'] = np.select(conditions, choices, default='else')

# Source notebook cell 6
adata.layers['raw_counts'] = adata.raw.X.copy()

# Source notebook cell 7
adata.raw = None

# Source notebook cell 9
import scanpy as sc
import pandas as pd

# =========================================================
# 0. Gene symbol column 설정
# =========================================================
GENE_SYMBOL_COL = "feature_name"

if GENE_SYMBOL_COL not in adata.var.columns:
    raise ValueError(f"{GENE_SYMBOL_COL} column not found in adata.var")

# feature_name을 문자열로 변환
gene_symbols = adata.var[GENE_SYMBOL_COL].astype(str)

# =========================================================
# 1. 미토콘드리아, 리보솜, 혈색소 유전자 확인
# =========================================================
adata.var["mt"] = gene_symbols.str.startswith("MT-")

adata.var["ribo"] = gene_symbols.str.startswith(("RPS", "RPL"))

# HBP pseudogene 제외 목적
# 기존 '^HB[^(P)]'는 정규식이 조금 애매해서 아래처럼 명확히 처리
adata.var["hb"] = gene_symbols.str.match(r"^HB(?!P)")

# =========================================================
# 2. BCR/TCR 유전자 플래깅
# =========================================================
adata.var["bcr_tcr"] = gene_symbols.str.contains(
    r"^TRAV|^TRBV|^TRDV|^TRGV|^TRAJ|^TRBJ|^TRDJ|^TRGJ|"
    r"^TRAC|^TRBC|^TRDC|^TRGC|"
    r"^IGHV|^IGHD|^IGHJ|^IGHC|"
    r"^IGLV|^IGLJ|^IGLC|"
    r"^IGKV|^IGKJ|^IGKC",
    regex=True
)

# =========================================================
# 3. 성별 편향적 유전자 플래깅
# =========================================================
sex_specific_genes = [
    "XIST", "TSIX",
    "RPS4Y1", "EIF1AY", "DDX3Y", "KDM5D",
    "UTY", "USP9Y", "ZFY", "PRKY"
]

adata.var["sex_biased"] = gene_symbols.isin(sex_specific_genes)

# =========================================================
# 4. 제거할 유전자 추출
# =========================================================
remove_mask = (
    adata.var["mt"] |
    adata.var["ribo"] |
    adata.var["hb"] |
    adata.var["bcr_tcr"] |
    adata.var["sex_biased"]
)

genes_to_remove = adata.var.index[remove_mask]

# 확인용 summary
print("제거 대상 유전자 수")
print(f"  mt          : {adata.var['mt'].sum()}")
print(f"  ribo        : {adata.var['ribo'].sum()}")
print(f"  hb          : {adata.var['hb'].sum()}")
print(f"  bcr_tcr     : {adata.var['bcr_tcr'].sum()}")
print(f"  sex_biased  : {adata.var['sex_biased'].sum()}")
print(f"  total unique: {len(genes_to_remove)}")

# 제거되는 gene symbol 확인
removed_gene_table = adata.var.loc[remove_mask, [GENE_SYMBOL_COL]].copy()
removed_gene_table["ensembl_id"] = removed_gene_table.index

print("\n제거 예시:")
print(removed_gene_table.head(20))

# =========================================================
# 5. filtering
# =========================================================
adata  = adata[:, ~remove_mask].copy() 
print(f"정제 후 adata 형태: {adata_filtered.shape}")

# Source notebook cell 10
adata.X = adata.layers['raw_counts'].copy()

# Source notebook cell 11
batch_info = pd.read_excel("./../raw_data/Batch_info.xlsx")  # 컬럼: DCP_ID, scRNA-seq Experimental Batch
print(batch_info.head())

# batch_info를 dict로 변환
batch_dict = dict(zip(batch_info['DCP_ID'], batch_info['scRNA-seq Experimental Batch']))

# donor_id에 매칭
adata.obs['batch'] = adata.obs['donor_id'].map(batch_dict)

# 4. 확인
print(adata.obs[['donor_id', 'batch']].head())

# Source notebook cell 12
adata.write_h5ad("./../raw_data/AIDA_pp.h5ad")
