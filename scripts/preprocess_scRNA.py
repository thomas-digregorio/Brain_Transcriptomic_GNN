import scanpy as sc
import pandas as pd
import numpy as np
import logging
import os
import argparse

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser()
    # Reduced arguments since we only support SEA-AD now
    args = parser.parse_args()

    # Paths
    # Find the first h5ad in raw that looks like SEA-AD
    raw_dir = "data/raw"
    possible_files = [f for f in os.listdir(raw_dir) if f.endswith('.h5ad') and 'SEAAD' in f]
    if not possible_files:
        logging.error("No SEA-AD data found in data/raw. Run scripts/download_sea_ad.py first.")
        return
    input_file = os.path.join(raw_dir, possible_files[0]) # Take the first one
    logging.info(f"Detected SEA-AD input file: {input_file}")
    
    output_file = "data/processed/sea_ad_proc.h5ad"
    
    logging.info(f"Loading data from {input_file}")
    
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found.")
        return

    # Use backed mode for large files to avoid OOM
    # For SEA-AD (30GB), this is essential on standard machines
    try:
        adata = sc.read_h5ad(input_file, backed='r')
    except:
         adata = sc.read_h5ad(input_file) # Fallback if backed fails (e.g. not h5ad)

    logging.info(f"Initial shape: {adata.shape}")

    # --- Dataset Specific Homogenization ---
    # Note: modifying .obs in backed mode is allowed and loads into memory
    print("Standardizing SEA-AD metadata...")
    if True: # Always run SEA-AD logic
        logging.info("Standardizing SEA-AD metadata...")
        
        # Load obs to memory (it's small compared to X)
        obs = adata.obs.copy()
        
        # 1. Cell Type
        if 'Subclass' in obs.columns:
            obs['cell_type'] = obs['Subclass']
        elif 'Class' in obs.columns:
            obs['cell_type'] = obs['Class']
            
        # 2. Key: Diagnosis
        if 'Cognitive Status' in obs.columns:
             # Convert to string to avoid Categorical errors with new values
             obs['diagnosis'] = obs['Cognitive Status'].astype(str).map({
                 'No dementia': 'Control', 
                 'Dementia': 'AD', # Check if 'Dementia' covers AD in SEA-AD
                 'MCI': 'MCI' 
             })
             # Map remaining unmapped or NaNs to Unknown
             obs['diagnosis'] = obs['diagnosis'].fillna('Unknown')

        else:
             obs['diagnosis'] = 'Unknown'

        # 3. Batch
        if 'Donor ID' in obs:
            obs['batch'] = obs['Donor ID']
            
        adata.obs = obs # update obs

    # For strict preprocessing of massive data, we usually need to subset to memory eventually
    # or use specialized tools. Here, we will take a heavy subsample if the file is huge
    # to allow the rest of the pipeline to run on a laptop demo.
    if adata.n_obs > 50000:
        logging.warning(f"Dataset is very large ({adata.n_obs} cells). Subsampling to 10k cells for demo feasibility...")
        # sc.pp.subsample doesn't work inplace on backed objects. 
        # Manual random sampling + slice to load into memory
        import numpy as np
        indices = np.random.choice(adata.n_obs, 10000, replace=False)
        # Sort indices for potentially faster h5 reading
        indices.sort()
        adata = adata[indices].to_memory()
    else:
        # If it fits in memory/is small, just load it to process efficiently
        if adata.isbacked:
            adata = adata.to_memory()

    # --- Standard QC & Preprocessing ---
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-') 
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Filter
    logging.info(f"Filtering: min_genes=200, max_genes=8000, max_mt_pct=10.0")
    # Robust filtering
    if 'n_genes_by_counts' in adata.obs:
        adata = adata[adata.obs.n_genes_by_counts < 8000, :]
    if 'pct_counts_mt' in adata.obs:
        adata = adata[adata.obs.pct_counts_mt < 10.0, :]
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    logging.info(f"Shape after filtering: {adata.shape}")

    # Normalize
    logging.info("Normalizing to 1e4 counts per cell...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG
    logging.info("Selecting top 2000 highly variable genes...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    except Exception as e:
        logging.warning(f"HVG selection failed: {e}. Skipping subset.")
    
    # PCA
    logging.info("Scaling data and running PCA...")
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)

    # Save
    logging.info(f"Saving processed data to {output_file}")
    adata.write(output_file)
    logging.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
