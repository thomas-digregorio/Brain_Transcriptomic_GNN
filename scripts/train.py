"""
Training Script
---------------
Trains GNN models using PyTorch Lightning.
Usage:
    python scripts/train.py --model cell --data_path data/processed/graphs/cell_graph.pt --epochs 10
"""

import argparse
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import sys
import os

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
    data = torch.load(args.data_path)
    
    # For full-graph training (transductive), we often treat the whole graph as one 'batch' instance
    # DataLoader with batch_size=1
    loader = DataLoader([data], batch_size=args.batch_size)
    
    # Init Model
    if args.model == 'cell':
        # data.x shape: [num_nodes, num_features]
        in_channels = data.num_features
        hidden_channels = 64
        out_channels = 3 # 0,1,2 classes
        
        backbone = CellGNN(in_channels, hidden_channels, out_channels)
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
        log_every_n_steps=1
    )
    
    print("Starting training...")
    trainer.fit(module, loader)
    
    # Save checkpoint
    # trainer.save_checkpoint(f"checkpoints/{args.model}_model.ckpt")

if __name__ == "__main__":
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    main()
