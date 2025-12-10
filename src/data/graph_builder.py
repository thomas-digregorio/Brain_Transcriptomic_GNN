"""
Graph Builder Module
--------------------
Constructs PyTorch Geometric (PyG) Data objects from AnnData.

1. Cell-Cell Graph:
   - Nodes: Cells
   - Edges: KNN connectivities (from Scanpy)
   - node_features: PCA embeddings

2. Gene-Gene Graph:
   - Nodes: Genes
   - Edges: Co-expression (Pearson correlation > threshold)
   - node_features: Gene properties (dispersion, mean expression) or learned embeddings
"""


import torch
from torch_geometric.data import Data
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import logging

logger = logging.getLogger(__name__)

def build_cell_graph(adata, use_rep='X_pca'):
    """
    Builds a cell-cell graph from the KNN connectivities.
    Strictly avoids accessing adata.X if possible to support backed mode.
    """
    logger.info("Building Cell-Cell Graph...")
    
    # Ensure neighbors are computed
    if 'connectivities' not in adata.obsp:
        logger.info("Computing neighbors (using PCA)...")
        # Ensure we represent the data efficiently. accessing .X on backed data is fatal.
        if use_rep not in adata.obsm:
             raise ValueError(f"{use_rep} not found in adata.obsm. PCA is required for large datasets.")
             
        # Scanpy's neighbors function works on backed data if use_rep is in obsm
        sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep)
    
    # Extract adjacency matrix (sparse)
    adj = adata.obsp['connectivities']
    
    # Convert to COO format for PyG
    coo = coo_matrix(adj)
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    
    # Edge weights (connectivity strength)
    edge_attr = torch.from_numpy(coo.data.astype(np.float32))
    
    # Node features: PCA embeddings
    # IMPORTANT: Do not fallback to X for large datasets
    if use_rep in adata.obsm:
        # Load PCA embeddings into memory (small: 1M x 50 float32 = 200MB)
        x = torch.from_numpy(adata.obsm[use_rep][:]).float() 
    else:
        raise ValueError(f"Feature representation {use_rep} missing. Cannot convert to graph features safely.")

            
    # Labels
    # priority: Cognitive Status > ADNC > diagnosis
    y = None
    label_col = None
    
    if 'Cognitive Status' in adata.obs:
        label_col = 'Cognitive Status'
    elif 'ADNC' in adata.obs:
        label_col = 'ADNC'
    elif 'diagnosis' in adata.obs:
        label_col = 'diagnosis'
        
    if label_col:
        # Filter out NaNs if necessary? Factorize handles it usually (as -1)
        # We need to ensure we don't have -1 in y for CrossEntropy usually, or ignore_index.
        # Let's map strict classes.
        
        # Simple factorization
        codes, uniques = pd.factorize(adata.obs[label_col])
        
        # Check for -1 (NaNs)
        if (codes == -1).any():
            logger.warning(f"Found missing labels in {label_col}. These will be -1.")
            
        y = torch.from_numpy(codes).long()
        logger.info(f"Encoded labels from '{label_col}': {dict(enumerate(uniques))}")
    else:
        logger.warning("No known label column found (Cognitive Status, ADNC, diagnosis). y will be None.")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Metadata for reference
    data.num_nodes = x.shape[0]
    data.num_features = x.shape[1]
    
    logger.info(f"Cell Graph created: {data}")
    return data

def build_gene_graph(adata, threshold=0.1):
    """
    Builds a gene-gene co-expression graph.
    Uses chunked processing or subsetting to avoid OOM.
    """
    logger.info(f"Building Gene-Gene Graph (Threshold={threshold})...")
    
    # 1. Select HVGs to reduce dimensionality
    if 'highly_variable' in adata.var:
        # Create a view or memory-safe subset
        hvg_mask = adata.var['highly_variable'].values
        n_hvg = np.sum(hvg_mask)
        logger.info(f"Computing correlation on {n_hvg} Highly Variable Genes...")
        
        # Load ONLY the HVG columns into memory. 
        # For 1M cells x 2000 genes = 2 billion items = ~8GB RAM (float32).
        # This is on the edge for 16GB RAM if other things are loaded, but feasible.
        # Ideally we'd calculate correlation chunk-wise, but for now loading 8GB is safer than 128GB (full X).
        
        # If the file is backed, we need to be careful.
        # Reading by column in HDF5 is slow. Reading by row is fast.
        # We can iterate chunks of rows, compute sufficient stats for correlation (mean, covariance).
        
        # Optimized chunked correlation:
        mean_expr, cov_matrix = compute_chunked_covariance(adata, hvg_mask)
        
        # Convert covariance to correlation
        diag = np.diag(cov_matrix)
        std_dev = np.sqrt(diag)
        outer_std = np.outer(std_dev, std_dev)
        corr_matrix = cov_matrix / (outer_std + 1e-8) # Avoid div zero
        
        var_names = adata.var_names[hvg_mask].tolist()
        
    else:
        logger.warning("No HVGs found. This might crash if using all genes.")
        # Fallback (dangerous for large data)
        mean_expr, cov_matrix = compute_chunked_covariance(adata, np.ones(adata.n_vars, dtype=bool))
        # ... (same calculation)
        diag = np.diag(cov_matrix)
        std_dev = np.sqrt(diag)
        outer_std = np.outer(std_dev, std_dev)
        corr_matrix = cov_matrix / (outer_std + 1e-8)
        var_names = adata.var_names.tolist()

    # 2. Thresholding to get edges
    np.fill_diagonal(corr_matrix, 0)
    
    mask = np.abs(corr_matrix) > threshold
    
    row, col = np.where(mask)
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    
    # Edge weights = correlation coefficient
    edge_values = corr_matrix[row, col]
    edge_attr = torch.tensor(edge_values, dtype=torch.float)
    
    # Node features
    # Use mean expression and variance (which is the diagonal of covariance)
    var_expr = np.diag(cov_matrix)
    x = torch.tensor(np.stack([mean_expr, var_expr], axis=1), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.gene_names = var_names
    
    logger.info(f"Gene Graph created: {data}")
    return data

def compute_chunked_covariance(adata, gene_mask, chunk_size=50000):
    """
    Computes covariance matrix and mean expression w.r.t to features (genes)
    iterating over samples (cells). Optimized for GPU.
    """
    n_cells = adata.n_obs
    n_genes = np.sum(gene_mask)
    gene_indices = np.where(gene_mask)[0]
    
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU Detected. Using {device} for covariance calculation.")
    else:
        device = torch.device('cpu')
        logger.warning("No GPU found. Falling back to CPU.")

    # Accumulators (on Device)
    # Double precision avoids numerical instability in variance calculation
    sum_x = torch.zeros(n_genes, device=device, dtype=torch.float64)
    sum_xxT = torch.zeros((n_genes, n_genes), device=device, dtype=torch.float64)
    
    start = 0
    count = 0
    
    logger.info(f"Computing covariance chunk-wise (Chunk size: {chunk_size})...")
    
    while start < n_cells:
        end = min(start + chunk_size, n_cells)
        
        # Load chunk (CPU RAM) -> Move to GPU
        # Handling backed vs memory
        chunk_data = adata[start:end, gene_indices].X
        
        if hasattr(chunk_data, "toarray"):
            chunk_data = chunk_data.toarray()
            
        # Convert to tensor and move to GPU
        # Check alignment for large chunks
        chunk = torch.from_numpy(chunk_data).to(device, dtype=torch.float64)
            
        chunk_sum = chunk.sum(dim=0)
        sum_x += chunk_sum
        
        # X^T X calculation
        sum_xxT += torch.matmul(chunk.T, chunk)
        
        count += (end - start)
        start = end
        
        if count % 100000 == 0 or count == n_cells:
            logger.info(f"  Processed {count}/{n_cells} cells...")
        
    # Finalize
    mean_x = sum_x / count
    
    # Covariance formula: E[XX^T] - E[X]E[X]^T
    term1 = sum_xxT / count
    term2 = torch.outer(mean_x, mean_x)
    
    cov_matrix = term1 - term2
    
    # Return as numpy for compatibility
    return mean_x.cpu().numpy(), cov_matrix.cpu().numpy()
