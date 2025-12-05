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
    """
    logger.info("Building Cell-Cell Graph...")
    
    # Ensure neighbors are computed
    if 'connectivities' not in adata.obsp:
        logger.info("Computing neighbors...")
        sc.pp.neighbors(adata, use_rep=use_rep)
    
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
    if use_rep in adata.obsm:
        x = torch.from_numpy(adata.obsm[use_rep]).float()
    else:
        logger.warning(f"{use_rep} not found in adata.obsm. Using X.")
        if isinstance(adata.X, np.ndarray):
             x = torch.from_numpy(adata.X).float()
        else:
             x = torch.from_numpy(adata.X.toarray()).float()
            
    # Labels (Diagnosis) - Map string labels to integers
    # Only if available
    y = None
    if 'diagnosis' in adata.obs:
        codes, uniques = pd.factorize(adata.obs['diagnosis'])
        y = torch.from_numpy(codes).long()
        logger.info(f"Encoded labels: {dict(enumerate(uniques))}")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Metadata for reference
    data.num_nodes = x.shape[0]
    data.num_features = x.shape[1]
    
    logger.info(f"Cell Graph created: {data}")
    return data

def build_gene_graph(adata, threshold=0.1):
    """
    Builds a gene-gene co-expression graph.
    """
    logger.info(f"Building Gene-Gene Graph (Threshold={threshold})...")
    
    # 1. Compute correlation matrix on HVGs
    # Genes are columns in adata, so we transpose X to get (genes x cells)
    # Actually, we want correlation of gene expression profiles across cells.
    # X shape: (n_cells, n_genes) -> columns are genes.
    
    # Use sparse matrix support if needed, or densify if n_genes is small (~2k)
    # adata_hvg should already be subsetted or we select HVGs
    if 'highly_variable' in adata.var:
        adata_sub = adata[:, adata.var['highly_variable']]
    else:
        adata_sub = adata
        
    df = pd.DataFrame(adata_sub.X.toarray(), columns=adata_sub.var_names)
    
    # Correlation (Pearson)
    corr_matrix = df.corr(method='pearson')
    
    # 2. Thresholding to get edges
    # Remove self-loops (diagonal)
    np.fill_diagonal(corr_matrix.values, 0)
    
    # Mask low correlations
    # Use absolute value? Or only positive co-expression?
    # Usually absolute for "module" detection, but sign matters for regulation.
    # Let's keep absolute > threshold for edges, store sign in edge_attr
    mask = np.abs(corr_matrix.values) > threshold
    
    row, col = np.where(mask)
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    
    # Edge weights = correlation coefficient
    edge_values = corr_matrix.values[row, col]
    edge_attr = torch.tensor(edge_values, dtype=torch.float)
    
    # Node features
    # What represents a gene? Mean expression, Dispersion?
    # For now: Just use 1-hot or simple stats
    # Ideally: Learn gene embeddings via GNN, but we need input features.
    # Let's use mean expression and variance across cells.
    mean_expr = np.mean(adata_sub.X.toarray(), axis=0)
    var_expr = np.var(adata_sub.X.toarray(), axis=0)
    x = torch.tensor(np.stack([mean_expr, var_expr], axis=1), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.gene_names = adata_sub.var_names.tolist()
    
    logger.info(f"Gene Graph created: {data}")
    return data
