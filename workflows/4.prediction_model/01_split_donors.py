# Curated publication source. See docs/SOURCE_MANIFEST.json.
import os
os.makedirs("data/train_test", exist_ok=True)
# Source notebook cell 0
import os
import pandas as pd
import numpy as np
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import scanpy as sc
import anndata
import random
from sklearn.model_selection import StratifiedShuffleSplit
import gc

# --- 기본 설정 ---
random_state = 251219
target_celltypes = ['NK','CD4T','CD8T','Mono']

# Source notebook cell 1
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split # fallback용

def split_donors_stratified(adata, test_size=0.2, seed=random_state): 
    sex_col = "sex"

    # 1-2. 필요한 컬럼 정의
    cols_to_fetch = ['donor_id', 'age']
    
    if 'Country' in adata.obs.columns:
        cols_to_fetch.append('Country')
    else:
        print("⚠️ 경고: 'Country' 컬럼이 없습니다.")

    if sex_col:
        cols_to_fetch.append(sex_col)
        print(f"✅ 성별 컬럼 감지됨: '{sex_col}'")
    else:
        print("⚠️ 경고: 성별 관련 컬럼('sex', 'gender' 등)을 찾을 수 없습니다.")

    # Donor 중복 제거 (한 Donor는 고유한 메타데이터를 가진다고 가정)
    donors_df = adata.obs[cols_to_fetch].drop_duplicates(subset='donor_id').set_index('donor_id')
    
    print("2. 계층화 그룹(Stratification Group) 생성 중...")
    
    # 2-1. 나이 그룹 (Age Binning: 10세 단위)
    donors_df['age_group'] = pd.cut(donors_df['age'], bins=range(20, 101, 10), right=False)
    
    # 2-2. Stratify Key 생성 (Country + Age Group + Sex)
    # 예: "SouthKorea_20.0_Female"
    # 정보가 없는 컬럼은 제외하고 키를 생성합니다.
    stratify_parts = []
    
    if 'Country' in donors_df.columns:
        stratify_parts.append(donors_df['Country'].astype(str))
    
    stratify_parts.append(donors_df['age_group'].astype(str))
    
    if sex_col:
        stratify_parts.append(donors_df[sex_col].astype(str))
    
    # 리스트 안의 시리즈들을 '_'로 결합
    donors_df['stratify_key'] = stratify_parts[0]
    for part in stratify_parts[1:]:
        donors_df['stratify_key'] = donors_df['stratify_key'] + "_" + part
    
    # 2-3. 소수 그룹(Singleton) 처리
    # 그룹 내 샘플이 1개뿐이면 Stratified Split 불가 -> 'single_sample_group'으로 통합
    key_counts = donors_df['stratify_key'].value_counts()
    single_sample_keys = key_counts[key_counts < 2].index 
    
    n_singletons = len(donors_df[donors_df['stratify_key'].isin(single_sample_keys)])
    print(f"  - Stratify 그룹 수 (조합): {len(key_counts)}")
    print(f"  - 샘플이 1개뿐인 그룹 수: {len(single_sample_keys)} (총 {n_singletons}명) -> 'single_sample_group'으로 통합")

    donors_df['stratify_key'] = donors_df['stratify_key'].replace(single_sample_keys, 'single_sample_group')
    
    print("3. StratifiedShuffleSplit 실행 중...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    try:
        train_indices, test_indices = next(sss.split(donors_df.index, donors_df['stratify_key']))
    except ValueError as e:
        print(f"⚠️ Stratification 실패 (이유: {e}) -> 단순 Random Split으로 대체합니다.")
        # fallback
        train_indices, test_indices = train_test_split(range(len(donors_df)), test_size=test_size, random_state=seed)

    train_donors = donors_df.index[train_indices]
    test_donors = donors_df.index[test_indices]
    
    print(f"  - 총 {len(donors_df)}명의 Donor를 Train: {len(train_donors)}명, Test: {len(test_donors)}명으로 분할 완료.")
    
    # ================= [검증 출력] =================
    print("\n[검증] Train/Test 데이터 분포 비율 확인:")
    
    # 1) 국가별 분포
    if 'Country' in donors_df.columns:
        print("\n >> 국가(Country) 분포:")
        dist_country = pd.DataFrame({
            'Train': donors_df.loc[train_donors, 'Country'].value_counts(normalize=True),
            'Test': donors_df.loc[test_donors, 'Country'].value_counts(normalize=True)
        })
        print(dist_country.round(3))

    # 2) 성별 분포
    if sex_col:
        print(f"\n >> 성별({sex_col}) 분포:")
        dist_sex = pd.DataFrame({
            'Train': donors_df.loc[train_donors, sex_col].value_counts(normalize=True),
            'Test': donors_df.loc[test_donors, sex_col].value_counts(normalize=True)
        })
        print(dist_sex.round(3))
    # ===============================================

    print("4. AnnData 객체 분할 및 반환 중...")
    result = {}
    
    # 전체 데이터에서 donor_id 기준으로 subset
    adata_train_full = adata[adata.obs['donor_id'].isin(train_donors)].copy()
    adata_test_full = adata[adata.obs['donor_id'].isin(test_donors)].copy()
     
    for ct in target_celltypes:
        # 해당 cell type만 추출
        train_sub = adata_train_full[adata_train_full.obs["Modeling_annotation"] == ct].copy()
        test_sub = adata_test_full[adata_test_full.obs["Modeling_annotation"] == ct].copy()
            
        # 데이터가 있는 경우만 결과에 포함
        if train_sub.n_obs > 0:
            result[f"{ct}_train"] = train_sub
        if test_sub.n_obs > 0:
            result[f"{ct}_test"] = test_sub
        
    return result, train_donors, test_donors, donors_df

# Source notebook cell 2
adata = sc.read_h5ad("./../raw_data/AIDA_pp.h5ad")

# Source notebook cell 3
splits, train_d, test_d, donors_meta = split_donors_stratified(adata, test_size=0.2, seed=random_state)
# h5ad 파일로 저장
for name, ad in splits.items():
    if ad.n_obs > 0: # 세포가 없는 AnnData는 저장하지 않음
        print(f"  - 저장 중: {name}.h5ad ({ad.n_obs} cells)")
        ad.write(f"data/train_test/{name}.h5ad")
        
del adata
del splits
pd.DataFrame({"donor_id": train_d}).to_csv("data/train_test/train_donors.csv", index=False)
pd.DataFrame({"donor_id": test_d}).to_csv("data/train_test/test_donors.csv", index=False)
donors_meta.to_csv("data/train_test/donor_metadata.csv")
del train_d, test_d, donors_meta
gc.collect()

print("모든 작업이 완료되었습니다.")
