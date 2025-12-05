import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class CellGNN(torch.nn.Module):
    """
    Node classification model for Cells.
    Predicts diagnosis (AD/Control) based on cell expression + neighbor similarity.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # On the second layer, we output `out_channels` classes
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # x: [num_cells, num_features]
        # edge_index: [2, num_edges]
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GeneGNN(torch.nn.Module):
    """
    Graph Autoencoder or Embedding model for Genes.
    Learns latent representations of genes based on the co-expression network.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Returns embeddings (not logits), unless we add a classification head
        return x
