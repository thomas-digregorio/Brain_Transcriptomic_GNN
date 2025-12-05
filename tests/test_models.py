import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.gnn_models import CellGNN, GeneGNN

def test_cell_gnn():
    print("Testing CellGNN...")
    # Dummy data
    num_nodes = 100
    in_channels = 50
    hidden_channels = 16
    out_channels = 2 # Binary class
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 200)) # Random edges
    
    model = CellGNN(in_channels, hidden_channels, out_channels)
    out = model(x, edge_index)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (num_nodes, out_channels)
    print("CellGNN test passed.")

def test_gene_gnn():
    print("Testing GeneGNN...")
    num_nodes = 100
    in_channels = 50
    hidden_channels = 16
    out_channels = 8 # Embedding dim
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    model = GeneGNN(in_channels, hidden_channels, out_channels)
    out = model(x, edge_index)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (num_nodes, out_channels)
    print("GeneGNN test passed.")

if __name__ == "__main__":
    test_cell_gnn()
    test_gene_gnn()
