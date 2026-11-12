dir.create("data/DEG_analysis", recursive=TRUE, showWarnings=FALSE)
library(DESeq2) 
library(splines) 
library(ggplot2) 
library(dplyr) 
library(limma) 

dir.create("data/trajectory", showWarnings = FALSE, recursive = TRUE) 
dir.create("data/expression", showWarnings = FALSE, recursive = TRUE) 
dir.create("data/adjusted_expression", showWarnings = FALSE, recursive = TRUE) 

cell_types <- c("NK", "CD8T", "CD4T", "B", "Mono") 

for (ct in cell_types) { 
  message(paste0("\n--- Processing Cell Type: ", ct, " ---")) 
  
  counts_path <- paste0("data/pseudobulk/pseudobulk_counts_", ct, ".csv") 
  meta_path <- paste0("data/pseudobulk/metadata_", ct, ".csv") 
  
  counts <- read.csv(counts_path, row.names = 1, check.names = FALSE) 
  metadata <- read.csv(meta_path, row.names = 1, check.names = FALSE) 
  
  common_donors <- intersect(colnames(counts), rownames(metadata)) 
  counts <- counts[, common_donors, drop = FALSE] 
  metadata <- metadata[common_donors, , drop = FALSE] 
  
  metadata$age <- as.numeric(metadata$age) 
  metadata$sex <- droplevels(as.factor(metadata$sex)) 
  metadata$Country <- droplevels(as.factor(metadata$Country)) 
  
  counts <- round(as.matrix(counts)) 
  counts <- counts[rowSums(counts) > 0, , drop = FALSE] 
  
  
  # ============================================================================ 
  # DESeq2 LRT 
  # ============================================================================ 
  
  design_formula <- ~ Country + sex + ns(age, df = 3) 
  reduced_formula <- ~ Country + sex 
  
  dds <- DESeqDataSetFromMatrix( 
    countData = counts, 
    colData = metadata, 
    design = design_formula 
  ) 
  
  message("Running DESeq2 LRT...") 
  
  dds_lrt <- DESeq( 
    dds, 
    test = "LRT", 
    reduced = reduced_formula 
  ) 
  
  res <- results(dds_lrt) 
  res_df <- as.data.frame(res) 
  res_df$gene <- rownames(res_df) 
  
  sig_genes <- res_df %>% 
    filter(!is.na(padj) & padj < 0.05) %>% 
    arrange(padj) 
  
  message( 
    paste0( 
      ">>> Found ", 
      nrow(sig_genes), 
      " significant trajectory DEGs for ", 
      ct 
    ) 
  ) 
  
  
  # ============================================================================ 
  # VST normalization 
  # ============================================================================ 
  
  vsd <- vst( 
    dds_lrt, 
    blind = FALSE 
  ) 
  
  vst_counts <- assay(vsd) 
  
  write.csv( 
    vst_counts, 
    paste0( 
      "data/expression/vst_expression_", 
      ct, 
      ".csv" 
    ) 
  ) 
  
  
  # ============================================================================ 
  # Covariate adjustment for ALL genes 
  # 
  # Remove: 
  #   - Country 
  #   - sex 
  # 
  # Preserve: 
  #   - nonlinear age effect: ns(age, df = 3) 
  # 
  # This adjusted matrix is intended for PCA / WGCNA. 
  # ============================================================================ 
  
  message("Removing Country and sex effects using limma...") 
  
  age_design <- model.matrix( 
    ~ ns(age, df = 3), 
    data = metadata 
  ) 
  
  adjusted_vst_counts <- removeBatchEffect( 
    vst_counts, 
    batch = metadata$Country, 
    batch2 = metadata$sex, 
    design = age_design 
  ) 
  
  write.csv( 
    adjusted_vst_counts, 
    paste0( 
      "data/adjusted_expression/adjusted_vst_expression_", 
      ct, 
      ".csv" 
    ) 
  ) 
  
  message( 
    paste0( 
      ">>> Saved covariate-adjusted VST for ALL ", 
      nrow(adjusted_vst_counts), 
      " genes" 
    ) 
  ) 
  
  
  # ============================================================================ 
  # Spline trajectory fitting 
  # ============================================================================ 
  
  age_grid <- seq( 
    min(metadata$age, na.rm = TRUE), 
    max(metadata$age, na.rm = TRUE), 
    length.out = 50 
  ) 
  
  ref_country <- names( 
    sort( 
      table(metadata$Country), 
      decreasing = TRUE 
    ) 
  )[1] 
  
  ref_sex <- names( 
    sort( 
      table(metadata$sex), 
      decreasing = TRUE 
    ) 
  )[1] 
  
  newdata_grid <- data.frame( 
    age = age_grid, 
    Country = factor( 
      ref_country, 
      levels = levels(metadata$Country) 
    ), 
    sex = factor( 
      ref_sex, 
      levels = levels(metadata$sex) 
    ) 
  ) 
  
  message( 
    paste0( 
      "Reference condition: Country=", 
      ref_country, 
      ", sex=", 
      ref_sex 
    ) 
  ) 
  
  
  # ============================================================================ 
  # Fit spline curves for significant DEGs 
  # ============================================================================ 
  
  fitted_list <- list() 
  amplitude_vec <- c() 
  
  for (gene in sig_genes$gene) { 
    
    df <- metadata 
    
    df$expr <- as.numeric( 
      vst_counts[ 
        gene, 
        rownames(metadata) 
      ] 
    ) 
    
    fit <- lm( 
      expr ~ Country + sex + ns(age, df = 3), 
      data = df 
    ) 
    
    fitted_line <- predict( 
      fit, 
      newdata = newdata_grid 
    ) 
    
    fitted_list[[gene]] <- fitted_line 
    
    amplitude_vec[gene] <- ( 
      max(fitted_line) - 
        min(fitted_line) 
    ) 
  } 
  
  
  # ============================================================================ 
  # Combine spline curves 
  # ============================================================================ 
  
  fitted_curves <- do.call( 
    rbind, 
    fitted_list 
  ) 
  
  colnames(fitted_curves) <- paste0( 
    "Age_", 
    round(age_grid, 1) 
  ) 
  
  
  # ============================================================================ 
  # Amplitude filtering 
  # ============================================================================ 
  
  sig_genes$amplitude <- amplitude_vec[ 
    sig_genes$gene 
  ] 
  
  sig_genes <- sig_genes %>% 
    filter(amplitude >= 0.2) %>% 
    arrange(padj) 
  
  fitted_curves <- fitted_curves[ 
    sig_genes$gene, 
    , 
    drop = FALSE 
  ] 
  
  message( 
    paste0( 
      ">>> After VST-scale amplitude filter (>=0.2): ", 
      nrow(sig_genes), 
      " trajectory DEGs for ", 
      ct 
    ) 
  ) 
  
  
  # ============================================================================ 
  # Save DEG / trajectory results 
  # ============================================================================ 
  
  write.csv( 
    res_df, 
    paste0( 
      "data/DEG_analysis/DEGs_", 
      ct, 
      ".csv" 
    ), 
    row.names = FALSE 
  ) 
  
  write.csv( 
    sig_genes, 
    paste0( 
      "data/DEG_analysis/significant_DEGs_", 
      ct, 
      ".csv" 
    ), 
    row.names = FALSE 
  ) 
  
  write.csv( 
    fitted_curves, 
    paste0( 
      "data/DEG_analysis/spline_fitted_curves_", 
      ct, 
      ".csv" 
    ) 
  ) 
} 

message("\n[SUCCESS] Phase 2 Completed.")
