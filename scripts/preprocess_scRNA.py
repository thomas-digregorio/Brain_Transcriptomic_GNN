import scanpy as sc
import pandas as pd
import numpy as np
import logging
import os
import argparse
import shutil
import h5py
# from sklearn.decomposition import IncrementalPCA # Not used in GPU mode
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
    
    logging.info(f"Subset strategy: Init with chunk 0, then resize/append for {n_new} cells.")
    
    chunk_size = 50000 
    
    # Step 1: Write the first chunk to initialize the file structure
    first_chunk_end = min(chunk_size, n_new)
    
    idx_chunk_0 = kept_indices[0:first_chunk_end]
    
    X_chunk_0 = adata_src.X[idx_chunk_0]
    if not sp.issparse(X_chunk_0):
        X_chunk_0 = sp.csr_matrix(X_chunk_0)
        
    adat_0 = sc.AnnData(
        X=X_chunk_0,
        obs=obs_new.iloc[0:first_chunk_end],
        var=var_new
    )
    adat_0.write(output_file)
    
    if n_new <= chunk_size:
        logging.info("Dataset small enough, single write finished.")
        return output_file
        
    # Step 2: Resize and Append
    logging.info(f"Resizing H5AD to {n_new} cells...")
    
    with h5py.File(output_file, 'r+') as f:
        if 'X' not in f:
             raise ValueError("X not found in written file.")
             
        is_sparse = isinstance(f['X'], h5py.Group) and 'data' in f['X']
        if not is_sparse:
             dset = f['X']
             dset.resize((n_new, n_vars))
        else:
             g = f['X']
             # FIX: Ensure 'indptr' is int64 (i8) to avoid overflow
             if g['indptr'].dtype != np.int64:
                 logging.info(f"Upgrading indptr dtype from {g['indptr'].dtype} to int64...")
                 data_ptr = g['indptr'][:]
                 del g['indptr']
                 g.create_dataset('indptr', data=data_ptr.astype(np.int64), maxshape=(None,), chunks=True)
                 
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

    current_row_idx = first_chunk_end
    
    with h5py.File(output_file, 'r+') as f:
        g = f['X']
        if not isinstance(g, h5py.Group):
             raise ValueError("Expected Sparse X (CSR) but got Dense.")
             
        dset_data = g['data']
        dset_indices = g['indices']
        dset_indptr = g['indptr']
        
        dset_indptr.resize((n_new + 1,))
        
        last_ptr = dset_indptr[current_row_idx]
        
        for i in range(first_chunk_end, n_new, chunk_size):
            end = min(i+chunk_size, n_new)
            src_idx = kept_indices[i:end]
            
            chunk_X = adata_src.X[src_idx]
            if not sp.issparse(chunk_X):
                 chunk_X = sp.csr_matrix(chunk_X)
                 
            c_data = chunk_X.data
            c_indices = chunk_X.indices
            c_indptr = chunk_X.indptr
            
            new_indptr = c_indptr[1:] + last_ptr
            
            n_new_vals = len(c_data)
            current_data_len = dset_data.shape[0]
            
            dset_data.resize((current_data_len + n_new_vals,))
            dset_indices.resize((current_data_len + n_new_vals,))
            
            dset_data[current_data_len:] = c_data
            dset_indices[current_data_len:] = c_indices
            
            dset_indptr[i+1 : end+1] = new_indptr
            
            last_ptr = new_indptr[-1]
            
            if i % 100000 == 0:
                logging.info(f"Appended rows {i}/{n_new}...")
        
        new_shape = np.array([n_new, n_vars], dtype=np.int64)
        g.attrs['shape'] = new_shape
    
    from anndata.experimental import write_elem
    with h5py.File(output_file, 'r+') as f:
        if 'obs' in f:
            del f['obs']
        write_elem(f, "obs", obs_new)
        
    logging.info("Subset complete.")
    return output_file

def select_hvg_gpu(adata, n_top_genes=2000, batch_size=10000):
    """
    Selects Highly Variable Genes using GPU acceleration (PyTorch).
    Calculates Mean and Dispersion (Variance/Mean) in chunks.
    Replaces slow CPU-based scan.
    """
    import torch
    
    if not torch.cuda.is_available():
        logging.warning("No GPU for HVG selection. Falling back to CPU manual scan.")
        # We could keep the CPU fallback here or fail. 
        # Given the "GPU Pipeline" goal, we should fail or warn heavily.
        return select_hvg_manual(adata, n_top_genes)

    device = torch.device("cuda")
    logging.info(f"Running GPU HVG Selection on {device}...")
    
    n_vars = adata.n_vars
    n_cells = adata.n_obs
    
    # Accumulators (Float64 for stability)
    sum_x = torch.zeros(n_vars, device=device, dtype=torch.float64)
    sum_sq_x = torch.zeros(n_vars, device=device, dtype=torch.float64)
    
    # Iterate in chunks
    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        
        # Load chunk (CPU)
        chunk = adata.X[i:end]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
            
        # Move to GPU
        x_batch = torch.from_numpy(chunk).to(device, dtype=torch.float32)
        
        # Normalize (Log1p) on GPU - mimics Seurat/CellRanger Flavor
        # norm = (count / total) * 10000 -> log1p
        scaling_factor = 10000.0
        row_sums = x_batch.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        x_batch.div_(row_sums).mul_(scaling_factor).log1p_()
        
        x_batch_64 = x_batch.double()
        
        # Update Stats
        sum_x += x_batch_64.sum(dim=0)
        sum_sq_x += (x_batch_64 ** 2).sum(dim=0)
        
        if i % 100000 == 0:
            logging.info(f"  HVG Scan {i}/{n_cells}...")
            
    # Finalize Stats
    mean = (sum_x / n_cells).cpu().numpy()
    mean_sq = (sum_sq_x / n_cells).cpu().numpy()
    var = mean_sq - (mean ** 2)
    
    # Calculate Dispersion (Var / Mean)
    # Handle zeros: if mean is very small, dispersion is typically 0 or ignored
    dispersion = np.zeros_like(mean)
    np.divide(var, mean, out=dispersion, where=mean > 1e-12)
    
    # Filter very low expression genes (noise)
    valid_genes = mean > 0.0125
    dispersion[~valid_genes] = -1.0
    
    # Select Top N
    top_indices = np.argsort(dispersion)[-n_top_genes:]
    
    # Construct Results
    hvg_mask = np.zeros(n_vars, dtype=bool)
    hvg_mask[top_indices] = True
    
    adata.var['highly_variable'] = hvg_mask
    adata.var['means'] = mean
    adata.var['dispersions'] = dispersion
    
    logging.info(f"Selected {n_top_genes} HVGs on GPU.")
    
    # EXPLICITLY SAVE TO FILE using write_elem (Scanpy/AnnData compatible)
    if adata.isbacked and adata.filename:
        logging.info("Explicitly saving HVG metadata to HDF5 file (using write_elem)...")
        from anndata.experimental import write_elem
        # We need to ensure we write the WHOLE .var dataframe to be safe
        try:
            with h5py.File(adata.filename, 'r+') as f:
                # Delete existing 'var' to allow full overwrite
                if 'var' in f: 
                    del f['var']
                
                # Write the updated .var dataframe from adata
                write_elem(f, "var", adata.var)
                
            logging.info("HVG metadata securely saved via write_elem.")
        except Exception as e:
             logging.error(f"Failed to save HVG metadata: {e}")

# =============================================================================
# GPU PCA IMPLEMENTATION (PyTorch)
# =============================================================================

def run_gpu_pca(adata, n_components=50, batch_size=20000, output_file=None):
    """
    Performs PCA using PyTorch on the GPU via the Covariance Method (X^T X).
    Uses Double Precision (float64) for stability.
    DIRECTLY writes to HDF5 to ensure persistence.
    """
    import torch
    
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available! Cannot run GPU PCA.")
        
    device = torch.device("cuda")
    
    # 1. Select HVGs (Critical for Memory/Speed)
    if 'highly_variable' in adata.var.columns:
        hvg_mask = adata.var['highly_variable'].values
        hvg_indices = np.where(hvg_mask)[0]
        n_vars = len(hvg_indices)
        logging.info(f"Starting GPU PCA on {device} using {n_vars} HVGs...")
    else:
        logging.warning("No 'highly_variable' genes found. Running PCA on ALL genes (High Memory Usage!).")
        hvg_indices = np.arange(adata.n_vars)
        n_vars = adata.n_vars
        
    n_cells = adata.n_obs
    
    # ---------------------------------------------------------
    # PASS 1: Calculate Mean and Covariance (X^T X)
    # ---------------------------------------------------------
    logging.info("Pass 1: Computing Covariance Matrix on GPU (Float64)...")
    
    # ACCUMULATORS IN FLOAT64
    cov_matrix = torch.zeros((n_vars, n_vars), device=device, dtype=torch.float64)
    sum_x = torch.zeros(n_vars, device=device, dtype=torch.float64)
    
    total_cells = 0
    
    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        
        # Load chunk (CPU) and Subset to HVGs
        chunk = adata.X[i:end]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
            
        chunk = chunk[:, hvg_indices]
            
        # Move to GPU
        x_batch = torch.from_numpy(chunk).to(device, dtype=torch.float32)
        
        # 1. Normalize (Log1p) on GPU in-place
        scaling_factor = 10000.0
        row_sums = x_batch.sum(dim=1, keepdim=True)
        # Avoid div by zero
        row_sums[row_sums == 0] = 1.0
        x_batch.div_(row_sums).mul_(scaling_factor).log1p_()
        
        # Cast to float64
        x_batch_64 = x_batch.double()
        
        # 2. Update stats
        sum_x += x_batch_64.sum(dim=0)
        
        # Update Uncentered Covariance (X^T X)
        cov_matrix += torch.matmul(x_batch_64.T, x_batch_64)
        
        total_cells += (end - i)
        
        if i % 100000 == 0:
            logging.info(f"  Aggregated {i}/{n_cells} cells...")
            
    # Compute Final Mean
    mean_x = sum_x / total_cells
    
    # Correct term: N * mu^T * mu
    correction = total_cells * torch.outer(mean_x, mean_x)
    final_cov = (cov_matrix - correction) / (total_cells - 1)
    
    # SANITIZE
    if torch.isnan(final_cov).any():
        logging.warning("NaNs detected in covariance matrix! Replacing with zeros.")
        final_cov = torch.nan_to_num(final_cov, nan=0.0)
    
    logging.info("  Covariance matrix computed.")
    
    # ---------------------------------------------------------
    # PASS 2: Eigendecomposition (PCA Training)
    # ---------------------------------------------------------
    logging.info("Pass 2: Eigendecomposition on GPU...")
    
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(final_cov)
    except Exception as e:
        logging.error(f"Eigendecomposition failed: {e}")
        final_cov_cpu = final_cov.cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(final_cov_cpu)
        eigenvalues = eigenvalues.to(device)
        eigenvectors = eigenvectors.to(device)
    
    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    components = eigenvectors[:, :n_components].float() # (n_vars, n_components)
    
    total_var = torch.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[:n_components] / total_var
    
    logging.info(f"  Top 50 components explain {explained_variance_ratio.sum().item():.4f} of variance.")
    
    # ---------------------------------------------------------
    # PASS 3: Project Data (Transform)
    # ---------------------------------------------------------
    logging.info("Pass 3: Projecting Data (Transform) and Saving...")
    
    X_pca_all = np.zeros((n_cells, n_components), dtype=np.float32)
    mean_x_32 = mean_x.float()
    
    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        
        chunk = adata.X[i:end]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
            
        chunk = chunk[:, hvg_indices]
            
        x_batch = torch.from_numpy(chunk).to(device, dtype=torch.float32)
        
        row_sums = x_batch.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        x_batch.div_(row_sums).mul_(scaling_factor).log1p_()
        
        x_batch -= mean_x_32
        
        projected = torch.matmul(x_batch, components)
        
        X_pca_all[i:end] = projected.cpu().numpy()
        
        if i % 100000 == 0:
            logging.info(f"  Projected {i}/{n_cells} cells...")
            
    # ---------------------------------------------------------
    # EXPLCIT SAVE with h5py (Robustness fix)
    # ---------------------------------------------------------
    logging.info("Saving PCA results explicitly to disk...")
    
    # Prepare data arrays
    var_arr = eigenvalues[:n_components].cpu().numpy()
    var_ratio_arr = explained_variance_ratio.cpu().numpy()
    
    pcs_full = np.zeros((adata.n_vars, n_components), dtype=np.float32)
    pcs_full[hvg_indices] = components.cpu().numpy()
    
    # Determine output file
    if output_file is None:
        if adata.filename:
            output_file = str(adata.filename)
        else:
            logging.warning("No output file specified and AnnData has no filename. Results will barely persist in memory.")
            
    if output_file:
         logging.info(f"Writing to HDF5: {output_file}")
         # We need to close adata if it's holding a lock? 
         # backed='r+' might lock it. But h5py often allows concurrent read/write if careful.
         # Safer: rely on h5py if possible, or just flush. 
         # Actually, adata.obsm assignment SHOULD work. The failure suggests we need to be violent.
         
         # Let's try writing to a separate 'obsm' if needed, or simply force open in 'r+'
         try:
             with h5py.File(output_file, 'r+') as f:
                # X_pca (obsm)
                if 'obsm' not in f: f.create_group('obsm')
                if 'X_pca' in f['obsm']: del f['obsm']['X_pca']
                f['obsm'].create_dataset('X_pca', data=X_pca_all)
                
                # uns/pca
                if 'uns' not in f: f.create_group('uns')
                if 'pca' not in f['uns']: f['uns'].create_group('pca')
                g_pca = f['uns']['pca']
                
                if 'variance' in g_pca: del g_pca['variance']
                g_pca.create_dataset('variance', data=var_arr)
                
                if 'variance_ratio' in g_pca: del g_pca['variance_ratio']
                g_pca.create_dataset('variance_ratio', data=var_ratio_arr)
                
                # varm/PCs
                if 'varm' not in f: f.create_group('varm')
                if 'PCs' in f['varm']: del f['varm']['PCs']
                f['varm'].create_dataset('PCs', data=pcs_full)
                
             logging.info("Successfully wrote PCA to HDF5 via h5py.")
         except Exception as e:
             logging.error(f"Failed to write HDF5 direct: {e}")
             logging.info("Falling back to Scanpy assignment...")
             adata.obsm['X_pca'] = X_pca_all
             adata.uns['pca'] = {'variance': var_arr, 'variance_ratio': var_ratio_arr}
             adata.varm['PCs'] = pcs_full

    else:
        # Fallback
        adata.obsm['X_pca'] = X_pca_all
        adata.uns['pca'] = {'variance': var_arr, 'variance_ratio': var_ratio_arr}
        adata.varm['PCs'] = pcs_full

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
    
    temp_file = "data/processed/sea_ad_temp.h5ad"
    final_file = "data/processed/sea_ad_proc.h5ad"
    
    os.makedirs("data/processed", exist_ok=True)

    # 1. Copy Raw -> Temp (if needed)
    if not os.path.exists(temp_file):
        logging.info(f"Copying {input_file} to {temp_file}...")
        shutil.copyfile(input_file, temp_file)
    
    # 2. Check if processing is already done
    if os.path.exists(final_file):
        logging.info(f"Found {final_file}. Loading for continued processing...")
        adata = sc.read_h5ad(final_file, backed='r+')
    else:
        # Load Temp
        adata = sc.read_h5ad(temp_file, backed='r+')
        
        # QC (Chunked)
        logging.info("Calculating QC metrics...")
        
        adata.obs['n_genes_by_counts'] = 0.0
        adata.obs['total_counts'] = 0.0
        adata.obs['pct_counts_mt'] = 0.0
        
        mt_indices = np.where(adata.var_names.str.startswith('MT-'))[0]
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

        # Define Filter Mask
        logging.info("Defining filter mask...")
        mask_cell = (adata.obs.n_genes_by_counts < 8000) & \
                    (adata.obs.pct_counts_mt < 10.0) & \
                    (adata.obs.n_genes_by_counts > 200)
        
        logging.info(f"Kept {mask_cell.sum()} / {adata.n_obs} cells")
        
        # Create Filtered File
        logging.info("Standard write memory-unsafe. Switching to manual chunked creation...")
        # Close adata before subsetting to release file handle if needed? 
        # Actually subset_and_save reads from adata_src (which is open) and writes to NEW file.
        subset_and_save(adata, final_file, mask_cell)
            
        del adata
        if os.path.exists(temp_file):
            # os.remove(temp_file) # Keep temp for safety or debug
            pass
            
        # Reload
        logging.info(f"Re-loading filtered file {final_file}...")
        adata = sc.read_h5ad(final_file, backed='r+')
    
    # 6. HVGs
    if 'highly_variable' not in adata.var.columns:
        try:
            select_hvg_gpu(adata, n_top_genes=2000)
        except Exception as e:
            logging.error(f"GPU HVG failed: {e}")
            return
    else:
        logging.info("HVGs already present.")

    # 7. GPU PCA
    logging.info("Running GPU PCA...")
    try:
        # Pass final_file explicitely to enforce writing
        run_gpu_pca(adata, n_components=50, output_file=final_file)
    except Exception as e:
        logging.error(f"GPU PCA failed: {e}")
        return
    
    logging.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
