import scanpy as sc
import logging

logging.basicConfig(level=logging.INFO)

input_file = "data/processed/sea_ad_proc.h5ad"
try:
    adata = sc.read_h5ad(input_file, backed='r')
    print("Obs columns:", adata.obs.columns.tolist())
    if 'Diagnosis' in adata.obs.columns:
        print("Found 'Diagnosis'")
    if 'diagnosis' in adata.obs.columns:
        print("Found 'diagnosis'")
    
    # Print first few rows of potential label columns
    for col in adata.obs.columns:
        if 'diag' in col.lower() or 'class' in col.lower() or 'type' in col.lower():
            print(f"{col}: {adata.obs[col].unique().tolist()[:5]}")

except Exception as e:
    print(f"Error: {e}")
