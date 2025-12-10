"""
Graph Building Script
---------------------
Constructs Cell-Cell (KNN) and Gene-Gene (Co-expression) graphs from processed data.
Usage:
    python scripts/build_graphs.py --input data/processed/sea_ad_proc.h5ad --out_dir data/processed/graphs
"""

import argparse
import logging
from pathlib import Path
import scanpy as sc
import torch
import sys
import os
import numpy as np
import scipy.sparse as sp

# Add src to path to import graph_builder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.graph_builder import build_cell_graph, build_gene_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/sea_ad_proc.h5ad")
    parser.add_argument("--out_dir", type=str, default="data/processed/graphs")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input {input_path} does not exist.")
        return

    logger.info(f"Loading processed AnnData from {input_path} (backed='r')...")
    # Strict mode: Preprocessing issues should be fixed upstream
    adata = sc.read_h5ad(input_path, backed='r')
    
    if 'highly_variable' not in adata.var.columns:
        logger.error("'highly_variable' genes not found. Preprocessing incomplete.")
        return
    
    # 1. Build Cell Graph
    if 'X_pca' not in adata.obsm:
        logger.error("X_pca not found in adata.obsm. Please run preprocessing first.")
        return

    cell_graph = build_cell_graph(adata, use_rep='X_pca')
    torch.save(cell_graph, out_dir / "cell_graph.pt")
    logger.info(f"Saved cell_graph.pt to {out_dir}")
    
    # 2. Build Gene Graph
    # Note: Threshold is a hyperparam.
    gene_graph = build_gene_graph(adata, threshold=0.15) 
    torch.save(gene_graph, out_dir / "gene_graph.pt")
    logger.info(f"Saved gene_graph.pt to {out_dir}")

if __name__ == "__main__":
    main()
