import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from typing import Optional

class GNNLightningModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3, task: str = 'classification'):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = learning_rate
        self.task = task
        
        # Determine num_classes safely
        num_classes = getattr(model, 'out_channels', 3)
        if not isinstance(num_classes, int):
             num_classes = getattr(model, 'out_features', 3)

        # Metrics
        if task == 'classification':
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        # PyG DataBatch object
        # x, edge_index, batch.batch (if needed for pooling), y
        
        out = self(batch.x, batch.edge_index)
        
        if self.task == 'classification':
            # Semi-supervised setting often uses masking, but here we assume full supervision on batch
            # If batch.y is available
            loss = F.nll_loss(out, batch.y)
            self.train_acc(out, batch.y)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_nodes)
            self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_nodes)
            return loss
            
        elif self.task == 'embedding':
             # Unsupervised Link Prediction Loss (GAE style)
             # 1. Positive Edges: Use batch.edge_index
             pos_edge_index = batch.edge_index
             
             # 2. Negative Edges: Sample edges that don't exist
             # We sample same number as positive edges
             from torch_geometric.utils import negative_sampling
             neg_edge_index = negative_sampling(
                 edge_index=pos_edge_index,
                 num_nodes=batch.num_nodes,
                 num_neg_samples=pos_edge_index.size(1),
                 method='sparse'
             )
             
             # 3. Decoder: Dot product similarity
             # Gather node embeddings for edges
             # out: [num_nodes, embed_dim]
             
             # Positive scores
             pos_src, pos_dst = pos_edge_index
             pos_out = (out[pos_src] * out[pos_dst]).sum(dim=-1) # Dot product
             
             # Negative scores
             neg_src, neg_dst = neg_edge_index
             neg_out = (out[neg_src] * out[neg_dst]).sum(dim=-1)
             
             # 4. Loss: Binary Cross Entropy with Logits
             # Positives should be 1, Negatives should be 0
             pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
             neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
             
             loss = pos_loss + neg_loss
             
             self.log('train_loss', loss, batch_size=batch.num_nodes)
             return loss

    def validation_step(self, batch, batch_idx):
        if self.task == 'classification':
            out = self(batch.x, batch.edge_index)
            loss = F.nll_loss(out, batch.y)
            self.val_acc(out, batch.y)
            self.log('val_loss', loss, batch_size=batch.num_nodes)
            self.log('val_acc', self.val_acc, batch_size=batch.num_nodes)

    def test_step(self, batch, batch_idx):
         if self.task == 'classification':
            out = self(batch.x, batch.edge_index)
            loss = F.nll_loss(out, batch.y)
            self.test_acc(out, batch.y)
            self.log('test_loss', loss, batch_size=batch.num_nodes)
            self.log('test_acc', self.test_acc, batch_size=batch.num_nodes)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
