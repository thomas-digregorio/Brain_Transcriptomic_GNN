"""
FastAPI Model Serving
---------------------
Serves the trained GNN model to predict diagnosis from cell expression profiles.

Run:
    uvicorn deploy.fastapi_app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.gnn_models import CellGNN

app = FastAPI(title="Brain GNN API", description="Predict AD pathology from single-cell expression")

# Global model variable
model = None

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

class PredictionRequest(BaseModel):
    expression: list[float]
    # In a real GNN, we also need neighbors. 
    # For this demo API, we might assume the input vector is already an aggregated feature 
    # or we handle single-node prediction (node classification without graph updates for new nodes is tricky).
    # Simplified: We treat the input as the node feature 'x' and assume 0 edges (transductive to inductive gap).

@app.on_event("startup")
def load_model():
    global model
    # Load model architecture
    # Assuming standard input dims from our dataset (e.g., 50 PCA components)
    # This should be config-driven in a real app
    input_dim = 50 
    model = CellGNN(in_channels=input_dim, hidden_channels=64, out_channels=3)
    
    # Load weights if checkpoint exists
    # checkpoint_path = "checkpoints/cell_model.ckpt"
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    print("Model loaded.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input list to tensor
        x = torch.tensor([request.expression], dtype=torch.float)
        
        # For a single node with no edges, edge_index is empty
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        with torch.no_grad():
            # Logits
            out = model(x, edge_index)
            probs = out.exp()
            pred_idx = probs.argmax(dim=1).item()
            
        classes = {0: 'MCI', 1: 'Control', 2: 'AD'}
        
        return {
            "diagnosis": classes.get(pred_idx, "Unknown"),
            "probabilities": probs.tolist()[0]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
