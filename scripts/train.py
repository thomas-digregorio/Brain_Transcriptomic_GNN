"""
Training Script
---------------
Trains GNN models using PyTorch Lightning.
Usage:
    python scripts/train.py --model cell --data_path data/processed/graphs/cell_graph.pt --epochs 10
"""

import argparse
import sys
import os

# Fix OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.gnn_models import CellGNN, GeneGNN
from src.training.trainers import GNNLightningModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['cell', 'gene'], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1 if torch.cuda.is_available() else 0)
    parser.add_argument("--batch_size", type=int, default=1) # Full match often uses batch_size=1
    args = parser.parse_args()
    
    
    # Load Data
    print(f"Loading data from {args.data_path}...")
    # PyTorch 2.6+ default weights_only=True breaks complex objects like PyG Data
    try:
        data = torch.load(args.data_path, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions where weights_only arg doesn't exist
        data = torch.load(args.data_path)
    
    # For full-graph training (transductive), we often treat the whole graph as one 'batch' instance
    # DataLoader with batch_size=1
    loader = DataLoader([data], batch_size=args.batch_size)
    
    # Init Model
    if args.model == 'cell':
        # data.x shape: [num_nodes, num_features]
        in_channels = data.num_features
        hidden_channels = 64
        # Dynamically determine number of classes
        if data.y is not None:
             out_channels = len(torch.unique(data.y))
        else:
             out_channels = 3 # Default fallback
        print(f"Initializing CellGNN with {out_channels} output classes")
        
        # REDUCED COMPLEXITY: heads=1 to fit 1M nodes on 16GB GPU
        backbone = CellGNN(in_channels, hidden_channels, out_channels, heads=1)
        task = 'classification'
    else:
        in_channels = data.x.shape[1]
        backbone = GeneGNN(in_channels, 64, 32)
        task = 'embedding'
        
    module = GNNLightningModule(backbone, task=task)
    
    # Init Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=1,
        precision='16-mixed' # FIX: Critical for memory savings on large graphs
    )
    
    print("Starting training...")
    trainer.fit(module, loader)
    
    # Save checkpoint
    os.makedirs("outputs", exist_ok=True)
    trainer.save_checkpoint(f"outputs/{args.model}_model.ckpt")
    print(f"Model saved to outputs/{args.model}_model.ckpt")

if __name__ == "__main__":
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    main()
