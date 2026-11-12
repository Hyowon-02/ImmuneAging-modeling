# Curated publication source. See docs/SOURCE_MANIFEST.json.
import scanpy as sc
# Source notebook cell 0
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import os

# Source notebook cell 2
adata = sc.read_h5ad("./../raw_data/AIDA_pp.h5ad")

# Source notebook cell 4
adata.obs['Modeling_annotation'] = pd.Categorical(
    adata.obs['Modeling_annotation'],
    categories=['NK', 'Mono', 'CD4T', 'CD8T', 'B','else'],
    ordered=True
)

# Source notebook cell 8
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
import os

def generate_donor_pseudobulk(
    adata,
    cell_type_col,
    target_cell_types,
    donor_col,
    layer="raw_counts",
    min_cells_per_donor=10
):
    pseudobulk_results = {}
    filtering_summary = []

    for cell_type in target_cell_types:
        print(f"\nProcessing Cell Type: {cell_type}...")

        subset = adata[adata.obs[cell_type_col] == cell_type].copy()

        cell_counts = subset.obs[donor_col].value_counts()
        valid_donors = cell_counts[cell_counts >= min_cells_per_donor].index
        excluded_donors = cell_counts[cell_counts < min_cells_per_donor].index

        n_before = len(cell_counts)
        n_after = len(valid_donors)
        n_excluded = len(excluded_donors)

        if n_excluded > 0:
            excluded_counts = cell_counts[excluded_donors]
            print(f"  ⚠ Excluding {n_excluded} donors with < {min_cells_per_donor} cells")
            print(
                f"    Excluded donor cell counts: "
                f"min={excluded_counts.min()}, "
                f"max={excluded_counts.max()}, "
                f"median={excluded_counts.median():.0f}"
            )

        print(f"  Donors: {n_before} → {n_after} (excluded {n_excluded})")

        subset = subset[subset.obs[donor_col].isin(valid_donors)].copy()

        if layer in subset.layers:
            counts = subset.layers[layer]
        else:
            counts = subset.X

        count_df = pd.DataFrame(
            counts.toarray() if hasattr(counts, "toarray") else counts,
            index=subset.obs[donor_col].values,
            columns=subset.var_names
        )

        pb_matrix = count_df.groupby(level=0).sum()

        covariates = ["age", "sex", "Country"]
        available_covs = [c for c in covariates if c in subset.obs.columns]

        metadata = subset.obs.drop_duplicates(subset=[donor_col])
        metadata = metadata.set_index(donor_col)[available_covs]

        metadata["n_cells"] = cell_counts[metadata.index].values

        common_donors = pb_matrix.index.intersection(metadata.index)
        pb_matrix = pb_matrix.loc[common_donors]
        metadata = metadata.loc[common_donors]

        # ================================================================
        # AIDA-style low-expression gene filtering
        # 조건:
        #   1) 전체 donor의 10% 이상에서 발현
        #   2) total UMI >= donor 수
        # ================================================================
        n_donors = len(common_donors)

        min_donors = max(1, int(np.ceil(n_donors * 0.10)))
        genes_expressed = (pb_matrix > 0).sum(axis=0)
        gene_total_counts = pb_matrix.sum(axis=0)

        genes_to_keep = (
            (genes_expressed >= min_donors) &
            (gene_total_counts >= n_donors)
        )

        n_genes_before = pb_matrix.shape[1]
        n_genes_after = int(genes_to_keep.sum())

        pb_matrix = pb_matrix.loc[:, genes_to_keep]

        print(
            f"  Genes: {n_genes_before} → {n_genes_after} "
            f"(removed {n_genes_before - n_genes_after} low-expression genes)"
        )
        print(
            f"    Criteria:"
            f"\n      - expressed in >= {min_donors} donors (10%)"
            f"\n      - total UMI >= {n_donors}"
        )

        filtering_summary.append({
            "cell_type": cell_type,
            "donors_before": n_before,
            "donors_after": n_after,
            "donors_excluded": n_excluded,
            "min_cells_threshold": min_cells_per_donor,
            "genes_before_filtering": n_genes_before,
            "genes_after_filtering": n_genes_after,
            "min_donors_gene_filter": min_donors,
            "total_umi_threshold": n_donors
        })

        pseudobulk_results[cell_type] = {
            "counts": pb_matrix.T,
            "metadata": metadata
        }

    summary_df = pd.DataFrame(filtering_summary)
    pseudobulk_results["_filtering_summary"] = summary_df

    print(f"\n{'=' * 50}")
    print("Donor / Gene Filtering Summary:")
    print(summary_df.to_string(index=False))
    print(f"{'=' * 50}")

    return pseudobulk_results


# --- 실행 부분 ---

os.makedirs("data/pseudobulk", exist_ok=True)

target_types = ["NK", "CD8T", "CD4T", "Mono", "B"]

results = generate_donor_pseudobulk(
    adata=adata,
    cell_type_col="Modeling_annotation",
    target_cell_types=target_types,
    donor_col="donor_id",
    layer="raw_counts",
    min_cells_per_donor=10
)

for ct in target_types:
    results[ct]["counts"].to_csv(f"data/pseudobulk/pseudobulk_counts_{ct}.csv")
    results[ct]["metadata"].to_csv(f"data/pseudobulk/metadata_{ct}.csv")

results["_filtering_summary"].to_csv(
    "data/pseudobulk/donor_filtering_summary.csv",
    index=False
)

print("\nPhase 1 Completed: All pseudobulk matrices and metadata saved.")
