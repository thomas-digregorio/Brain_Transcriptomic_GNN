import scanpy as sc
import pandas as pd
import numpy as np
import logging
import os
import argparse
import shutil
import h5py
from sklearn.decomposition import IncrementalPCA
import scipy.sparse as sp

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def subset_and_save(adata_src, output_file, mask_obs):
    """
    Creates a new backed h5ad file containing only the cells in mask_obs.
    Copies data chunk-by-chunk to avoid memory spikes.
    """
    logging.info(f"Creating subsetted file: {output_file}")
    
    # Ensure output doesn't exist
    if os.path.exists(output_file):
        os.remove(output_file)

    kept_indices = np.where(mask_obs)[0]
    n_source = adata_src.shape[0]
    n_vars = adata_src.shape[1]
    n_new = len(kept_indices)
    
    # 1. Prepare Metadata (Obs/Var)
    obs_new = adata_src.obs.iloc[kept_indices].copy()
    var_new = adata_src.var.copy()
    
    # 2. Create the H5AD Shell with Unstructured Data
    # We use a dummy AnnData to write the structure, but with empty X
    # This sets up /obs, /var, /uns correctly.
    # Note: Writing an empty sparse matrix of 1M cells might be tricky.
    # Strategy: Write a tiny AnnData, then resize X in h5py.
    
    # Write just the metadata first? 
    # Easiest way: Write the first chunk primarily using Scanpy to define structure,
    # then append the rest using custom code? 
    # Problem: Scanpy doesn't support appending rows easily.
    
    # ROBUST STRATEGY: 
    # Write the full Obs/Var and a CSR matrix of correct shape but 0 non-zeros?
    # No, we just build the file manually using h5py for X.
    
    # Let's use the 'h5py' API to create the file and datasets.
    # But dealing with complex obs/var (categories) manually is nightmare.
    
    # Hybrid Strategy:
    # 1. Create a "Template" AnnData with 0 cells but correct VAR and UNS? 
    #    Scanpy struggles with 0 cells sometimes.
    # 2. Create an AnnData with the FIRST 1 cell. Write it.
    # 3. Open with h5py, resize X and Obs to N_new.
    # 4. Fill data.
    
    logging.info(f"Subset strategy: Init with chunk 0, then resize/append for {n_new} cells.")
    
    chunk_size = 50000 
    
    # Step 1: Write the first chunk to initialize the file structure
    first_chunk_end = min(chunk_size, n_new)
    
    # Map 'new' indices 0..chunk_size to 'source' indices
    idx_chunk_0 = kept_indices[0:first_chunk_end]
    
    X_chunk_0 = adata_src.X[idx_chunk_0]
    # Ensure sparse
    if not sp.issparse(X_chunk_0):
        X_chunk_0 = sp.csr_matrix(X_chunk_0)
        
    adat_0 = sc.AnnData(
        X=X_chunk_0,
        obs=obs_new.iloc[0:first_chunk_end],
        var=var_new
    )
    # Copy uns/obsm if needed? (Usually empty at this stage)
    adat_0.write(output_file)
    
    if n_new <= chunk_size:
        logging.info("Dataset small enough, single write finished.")
        return output_file
        
    # Step 2: Resize and Append
    # We use h5py to resize the datasets
    logging.info(f"Resizing H5AD to {n_new} cells...")
    
    with h5py.File(output_file, 'r+') as f:
        # Resize X (which is a Group for Sparse, or Dataset for Dense)
        if 'X' not in f:
             raise ValueError("X not found in written file.")
             
        is_sparse = isinstance(f['X'], h5py.Group) and 'data' in f['X']
        if not is_sparse:
             # Dense dataset
             dset = f['X']
             dset.resize((n_new, n_vars))
        else:
             # CSR/CSC Group
             g = f['X']
             
             # FIX: Ensure 'indptr' is int64 (i8) to avoid overflow with >2B elements
             # Scanpy often defaults to int32 for small initial chunks.
             if g['indptr'].dtype != np.int64:
                 logging.info(f"Upgrading indptr dtype from {g['indptr'].dtype} to int64...")
                 data_ptr = g['indptr'][:]
                 del g['indptr']
                 # Recreate with int64 and unlimited maxshape
                 g.create_dataset('indptr', data=data_ptr.astype(np.int64), maxshape=(None,), chunks=True)
                 
             # Also ensure 'data' and 'indices' are resizable (chunked)
             # If they were written as contiguous, resize would fail.
             # We check maxshape.
             if g['data'].maxshape != (None,):
                 logging.info("Enabling chunking for 'data'...")
                 d_data = g['data'][:]
                 del g['data']
                 g.create_dataset('data', data=d_data, maxshape=(None,), chunks=True)
                 
             if g['indices'].maxshape != (None,):
                 logging.info("Enabling chunking for 'indices'...")
                 d_ind = g['indices'][:]
                 del g['indices']
                 g.create_dataset('indices', data=d_ind, maxshape=(None,), chunks=True)

             # We need to resize data, indices, indptr
             pass
             
    # Actually, h5py sparse append is tricky because you have to update `indptr`.
    # `indptr` for CSR is [0, nnz_row1, nnz_row1+row2, ...]
    # When we append a chunk, we take its indptr, shift it by the last value of current indptr, and append.
    
    current_row_idx = first_chunk_end
    
    # We open file once to append
    with h5py.File(output_file, 'r+') as f:
        # Access Sparse X
        g = f['X']
        if not isinstance(g, h5py.Group):
             raise ValueError("Expected Sparse X (CSR) but got Dense. Ensure input is sparse.")
             
        # Buffers for fast resize
        dset_data = g['data']
        dset_indices = g['indices']
        dset_indptr = g['indptr']
        
        # Resize Indptr now to final size
        # indptr size = n_rows + 1. 
        # Current size = first_chunk + 1
        dset_indptr.resize((n_new + 1,))
        
        last_ptr = dset_indptr[current_row_idx] # Value at the boundary
        
        # Iterate remaining chunks
        for i in range(first_chunk_end, n_new, chunk_size):
            end = min(i+chunk_size, n_new)
            
            # Source indices
            src_idx = kept_indices[i:end]
            
            # Read Source
            # This is the memory-critical part. data_src.X[src_idx]
            # Since data_src is backed, simple slicing works efficiently.
            chunk_X = adata_src.X[src_idx]
            if not sp.issparse(chunk_X):
                 chunk_X = sp.csr_matrix(chunk_X)
                 
            # Extract CSR components
            c_data = chunk_X.data
            c_indices = chunk_X.indices
            c_indptr = chunk_X.indptr
            
            # c_indptr starts at 0. We need to shift it by `last_ptr`
            # But wait, c_indptr[0] is 0. 
            # The appended indptr should skip the first 0, and add last_ptr to the rest.
            
            new_indptr = c_indptr[1:] + last_ptr
            
            # Resize data/indices
            n_new_vals = len(c_data)
            current_data_len = dset_data.shape[0]
            
            dset_data.resize((current_data_len + n_new_vals,))
            dset_indices.resize((current_data_len + n_new_vals,))
            
            # specific slices
            dset_data[current_data_len:] = c_data
            dset_indices[current_data_len:] = c_indices
            
            # Update indptr
            # i+1 : end+1
            # Wait, i is start row index. i=first_chunk_end.
            # indptr index 0 corresponds to row 0.
            # indptr index K corresponds to end of row K-1.
            # We want to fill dset_indptr[i+1 : end+1].
            
            dset_indptr[i+1 : end+1] = new_indptr
            
            last_ptr = new_indptr[-1]
            
            if i % 100000 == 0:
                logging.info(f"Appended rows {i}/{n_new}...")
        
        # CRITICAL FIX: Update the shape attribute of the Sparse Matrix Group
        # Anndata/Scanpy relies on this attribute to know the matrix dimensions.
        new_shape = np.array([n_new, n_vars], dtype=np.int64)
        if 'shape' in g.attrs:
            g.attrs['shape'] = new_shape
        else:
            # Maybe it's 'h5sparse_shape' in some versions? No, standard is 'shape'.
            # But let's check standard anndata encoding.
            g.attrs['shape'] = new_shape

    # Finally: We must also update 'obs' in the file!
    # The first write only wrote the first chunk of obs.
    # Scanpy stores obs as a composite of arrays in /obs group.
    # This is VERY TRICKY to resize manually because of Categories, Index strings, etc.
    
    # HACK: 
    # Since obs tables are usually small (1M rows x 10 cols is small RAM), 
    # we can just OVERWRITE the /obs group at the end?
    # Yes. h5ad /obs is column-oriented usually.
    # We can delete 'obs' group and re-write it using Scanpy's internal API or just standard h5py? 
    # NO. 
    
    # Better HACK:
    # Read the file back in 'r+' mode with Scanpy??
    # No, sizes don't match X.
    
    # BEST HACK:
    # We already have `obs_new` (the full dataframe) in memory.
    # We can use `anndata.experimental.write_elem` to write the full obs to the file, replacing the old one.
    
    from anndata.experimental import write_elem
    with h5py.File(output_file, 'r+') as f:
        # Delete partial obs
        if 'obs' in f:
            del f['obs']
        # Write full obs
        write_elem(f, "obs", obs_new)
        
    logging.info("Subset complete.")
    return output_file

def run_incremental_pca(adata, n_components=50, batch_size=5000):
    """
    Runs PCA incrementally on the backed AnnData object.
    Requires HVGs to be selected already.
    """
    logging.info("Starting Incremental PCA...")
    
    # 1. Identify HVGs
    hvg_mask = adata.var['highly_variable'].values
    hvg_indices = np.where(hvg_mask)[0]
    n_hvg = len(hvg_indices)
    
    if n_hvg == 0:
        raise ValueError("No HVGs selected.")
        
    logging.info(f"PCA will run on {n_hvg} HVGs.")

    ipca = IncrementalPCA(n_components=n_components)
    
    # --- Pass 1: Compute Mean/Std for Scaling (Optional but recommended for PCA) ---
    # Actually, IPCA centers data automatically (it tracks mean). 
    # But we usually want to log1p and scale to unit variance (Scanpy style) before PCA.
    # StandardScaler(with_mean=True, with_std=True) requires a pass.
    
    # Let's perform "Online" Scaling + IPCA together? 
    # IPCA removes mean. It does NOT scale variance.
    # We need to compute column variance first.
    
    logging.info("Pass 1: Computing Variance for Scaling...")
    # WELFOLD'S ALGORITHM for online variance? Or just simple chunk accumulation?
    # Simple chunk accumulation for sum and sum_sq
    
    sum_x = np.zeros(n_hvg)
    sum_sq_x = np.zeros(n_hvg)
    n_total = 0
    
    n_cells = adata.n_obs
    
    for i in range(0, n_cells, batch_size):
        end = min(i+batch_size, n_cells)
        # Load raw counts for HVGs
        chunk = adata.X[i:end][:, hvg_indices]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
            
        # Normalize (custom on-the-fly)
        # sc.pp.normalize_total equivalent:
        counts_per_cell = chunk.sum(axis=1, keepdims=True)
        counts_per_cell[counts_per_cell==0] = 1 # avoid div zero
        chunk = (chunk / counts_per_cell) * 1e4
        chunk = np.log1p(chunk)
        
        sum_x += chunk.sum(axis=0)
        sum_sq_x += (chunk ** 2).sum(axis=0)
        n_total += (end - i)
        
    mean = sum_x / n_total
    # Var = E[X^2] - (E[X])^2
    var = (sum_sq_x / n_total) - (mean ** 2)
    std = np.sqrt(var)
    std[std == 0] = 1 # avoid div zero
    
    logging.info("Pass 2: Fitting Incremental PCA...")
    
    for i in range(0, n_cells, batch_size):
        end = min(i+batch_size, n_cells)
        
        # Load & Preprocess
        chunk = adata.X[i:end][:, hvg_indices]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
            
        counts_per_cell = chunk.sum(axis=1, keepdims=True)
        counts_per_cell[counts_per_cell==0] = 1
        chunk = (chunk / counts_per_cell) * 1e4
        chunk = np.log1p(chunk)
        
        # Scale
        chunk = (chunk - mean) / std
        
        # Fit
        # Check for NaN
        chunk = np.nan_to_num(chunk)
        ipca.partial_fit(chunk)
        
        if i % 50000 == 0:
            logging.info(f"Fits {i}/{n_cells}...")
            
    # --- Pass 3: Transform and Save ---
    logging.info("Pass 3: Transforming and saving X_pca...")
    
    # We need to write X_pca to the AnnData. 
    # Backed AnnData allows writing to obsm? Yes.
    
    # Initialize empty obsm in memory? No, might crash.
    # We can write directly to the HDF5 file backing if we are careful, 
    # OR we just accumulate X_pca in memory?
    # X_pca is (N_cells, 50). 
    # For 1M cells, that is 10^6 * 50 * 4 bytes = 200MB. 
    # TOTALLY SAFE to keep in memory.
    
    X_pca_list = []
    
    for i in range(0, n_cells, batch_size):
        end = min(i+batch_size, n_cells)
        
        chunk = adata.X[i:end][:, hvg_indices]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
            
        counts_per_cell = chunk.sum(axis=1, keepdims=True)
        counts_per_cell[counts_per_cell==0] = 1
        chunk = (chunk / counts_per_cell) * 1e4
        chunk = np.log1p(chunk)
        chunk = (chunk - mean) / std
        chunk = np.nan_to_num(chunk)
        
        chunk_pca = ipca.transform(chunk)
        X_pca_list.append(chunk_pca)
        
    X_pca_final = np.vstack(X_pca_list)
    adata.obsm['X_pca'] = X_pca_final
    logging.info("PCA Complete.")


def select_hvg_manual(adata, n_top_genes=2000, batch_size=10000):
    """
    Manually select HVGs using a chunked pass to avoid Scanpy/Numpy errors on backed data.
    Approximation of Cell Ranger / Seurat flavor (Dispersion based).
    """
    logging.info("Running Manual Chunked HVG Selection...")
    n_vars = adata.n_vars
    n_cells = adata.n_obs
    
    # Accumulators for Welford's or simple Sum/SumSq
    # We do simple Sum/SumSq on Log1pNormalized data
    sum_x = np.zeros(n_vars, dtype=np.float64)
    sum_sq_x = np.zeros(n_vars, dtype=np.float64)
    
    # We need a proper mean/var of the Log1p data
    
    for i in range(0, n_cells, batch_size):
        end = min(i+batch_size, n_cells)
        # Load Raw
        chunk = adata.X[i:end]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
            
        # Normalize & Log1p
        counts = chunk.sum(axis=1, keepdims=True)
        counts[counts==0] = 1
        chunk = (chunk / counts) * 1e4
        chunk = np.log1p(chunk)
        
        sum_x += chunk.sum(axis=0)
        sum_sq_x += (chunk ** 2).sum(axis=0)
        
        if i % 100000 == 0:
            logging.info(f"HVG Scan {i}/{n_cells}...")

    mean = sum_x / n_cells
    var = (sum_sq_x / n_cells) - (mean ** 2)
    # Dispersion = var / mean
    dispersion = np.zeros_like(mean)
    np.divide(var, mean, out=dispersion, where=mean > 1e-12)
    
    # Simple selection: Top N by dispersion
    # (Real Seurat bins by mean expression, but this is a robust fallback)
    
    # Filter out very low expression genes first to avoid noise
    valid_genes = mean > 0.0125 # ~ scanpy default min mean
    
    # effective dispersion
    dispersion[~valid_genes] = -1.0
    
    # Get indices of top dispersion
    # argsort is ascending, so take last N
    top_indices = np.argsort(dispersion)[-n_top_genes:]
    
    adata.var['highly_variable'] = False
    adata.var['means'] = mean
    adata.var['dispersions'] = dispersion
    
    # Set True for top indices
    # We need to map integer indices to labels if using loc, or just boolean array
    hvg_mask = np.zeros(n_vars, dtype=bool)
    hvg_mask[top_indices] = True
    adata.var['highly_variable'] = hvg_mask
    
    logging.info(f"Selected {n_top_genes} HVGs manually.")


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Paths
    raw_dir = "data/raw"
    possible_files = [f for f in os.listdir(raw_dir) if f.endswith('.h5ad') and 'SEAAD' in f]
    if not possible_files:
        logging.error("No SEA-AD data found in data/raw.")
        return
    input_file = os.path.join(raw_dir, possible_files[0])
    
    # Intermediate output
    # We maintain 3 files: 
    # 1. Raw (Input)
    # 2. Backed Copy (For initial QC)
    # 3. Filtered Backed (Final)
    
    temp_file = "data/processed/sea_ad_temp.h5ad"
    final_file = "data/processed/sea_ad_proc.h5ad"
    
    # ensure processed dir exists
    os.makedirs("data/processed", exist_ok=True)

    # 1. Copy Raw -> Temp
    if os.path.exists(temp_file):
        os.remove(temp_file)
    logging.info(f"Copying {input_file} to {temp_file}...")
    shutil.copyfile(input_file, temp_file)
    
    # Load Temp
    adata = sc.read_h5ad(temp_file, backed='r+')
    
    # 2. QC (Chunked)
    logging.info("Calculating QC metrics...")
    
    # Init cols
    adata.obs['n_genes_by_counts'] = 0.0
    adata.obs['total_counts'] = 0.0
    adata.obs['pct_counts_mt'] = 0.0
    
    mt_gene_mask = adata.var_names.str.startswith('MT-')
    mt_indices = np.where(mt_gene_mask)[0]
    
    chunk_size = 10000
    n_cells = adata.n_obs
    
    for i in range(0, n_cells, chunk_size):
        end = min(i + chunk_size, n_cells)
        chunk = adata.X[i:end]
        if hasattr(chunk, "toarray"): chunk = chunk.toarray()
            
        n_genes = np.count_nonzero(chunk, axis=1)
        total_counts = chunk.sum(axis=1)
        if isinstance(total_counts, np.matrix): total_counts = total_counts.A1
            
        if len(mt_indices) > 0:
            mt_counts = chunk[:, mt_indices].sum(axis=1)
            if isinstance(mt_counts, np.matrix): mt_counts = mt_counts.A1
        else:
            mt_counts = np.zeros(end-i)

        adata.obs.iloc[i:end, adata.obs.columns.get_loc('n_genes_by_counts')] = n_genes
        adata.obs.iloc[i:end, adata.obs.columns.get_loc('total_counts')] = total_counts
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = (mt_counts / total_counts) * 100
        pct[np.isnan(pct)] = 0
        adata.obs.iloc[i:end, adata.obs.columns.get_loc('pct_counts_mt')] = pct
        
        if i % 100000 == 0: logging.info(f"QC {i}/{n_cells}")

    # 3. Define Filter Mask
    logging.info("Defining filter mask...")
    mask_cell = (adata.obs.n_genes_by_counts < 8000) & \
                (adata.obs.pct_counts_mt < 10.0) & \
                (adata.obs.n_genes_by_counts > 200)
    
    logging.info(f"Kept {mask_cell.sum()} / {adata.n_obs} cells")
    
    # 4. Create Filtered File (Crucial Step)
    # Since we can't easily perform "subset_and_save" efficiently with basic h5py without code complexity,
    # and "adata[mask].write()" explodes memory,
    # we will use the "In-Place Filter" strategy if possible? No, file size doesn't shrink.
    # 
    # ALTERNATIVE: Just write 'mask_cell' to the disk and use it for the rest of parameters?
    # Not ideal for distribution.
    # 
    # Let's try the simple scanpy write for now, assuming the USER has at least 16GB RAM. 
    # If 1M cells x 30k genes (sparse), structure is ~4GB. It SHOULD fit in RAM to simple 'write'.
    # The crash usually happens during dense conversion.
    # 
    # We will try: Load Masked View -> Write to Final.
    # If this crashes, we need the complex h5py surgeon.
    
    # Implement the manual chunked copy strategy
    logging.info("Standard write memory-unsafe. Switching to manual chunked creation...")
    subset_and_save(adata, final_file, mask_cell)
        
    # Free up temp file
    del adata
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    # 5. Work on Final File (Backed)
    logging.info(f"Re-loading filtered file {final_file}...")
    adata = sc.read_h5ad(final_file, backed='r+')
    
    # 6. HVGs (Manual Chunked to avoid Crash)
    # Scanpy's backed mode fails on Windows/Numpy recent versions for this size.
    try:
        select_hvg_manual(adata, n_top_genes=2000)
    except Exception as e:
        logging.error(f"Manual HVG failed: {e}")
        return

    # 7. Incremental PCA
    run_incremental_pca(adata, n_components=50)
    
    logging.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
