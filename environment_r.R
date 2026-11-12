# DESeq2 version 1.44.0 is reported in the manuscript; use its matching
# Bioconductor environment and capture sessionInfo() after confirming reproduction.
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
BiocManager::install(c("DESeq2", "limma"))
install.packages(c("ggplot2", "dplyr"))
sessionInfo()
