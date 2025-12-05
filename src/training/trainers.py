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
        
        # Metrics
        if task == 'classification':
            self.train_acc = Accuracy(task="multiclass", num_classes=3)
            self.val_acc = Accuracy(task="multiclass", num_classes=3)
            self.test_acc = Accuracy(task="multiclass", num_classes=3)

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
             # Unsupervised / Link Prediction / Reconstruction loss
             # Simplified: just return 0 for demo or implement a reconstruction loss
             # e.g. Reconstruct adjacency
             return 0.0

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
