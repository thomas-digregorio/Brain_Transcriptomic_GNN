# brain-gnn-transcriptomics: GNNs on Single-Cell Human Brain Data for AD/CNS Insights

![Project Banner](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## 🧠 Project Overview
**One-liner:** A scalable Graph Neural Network (GNN) pipeline to analyze large-scale single-cell transcriptomic data from postmortem human brains (ROSMAP) to predict Alzheimer’s pathology and identify disease-associated gene modules.

This project demonstrates an end-to-end "AI for Drug Discovery" workflow. It processes raw single-cell RNA-seq (scRNA-seq) data, constructs biological graphs (Cell-Cell and Gene-Gene), and trains GNNs to uncover molecular drivers of neurodegeneration.

### Key Features
- **Scalable Data Ingestion**: Handling large-scale CNS transcriptomic datasets (ROSMAP / Allen Brain Atlas).
- **biologically-Informed GNNs**:
  - **Cell-Level**: Predicting AD diagnosis and cellular resilience using GraphSAGE/GAT on cell-similarity graphs.
  - **Gene-Level**: Identifying target modules using gene co-expression and PPI networks.
- **Production-Grade**: Distributed training (PyTorch Lightning + DDP), Reproducible environments (Docker), and Cloud-ready deployment (FastAPI).

---

## 📂 Repository Structure
```text
brain-gnn-transcriptomics/
├── config/             # Configuration files (Hydra/YAML)
├── data/               # Data storage (Raw and Processed)
├── deploy/             # Deployment configs (Docker, K8s, FastAPI)
├── notebooks/          # Jupyter notebooks for EDA and insights
├── scripts/            # Executable scripts for data ingest and training
├── src/                # Source code modules
│   ├── data/           # Data loading and transformation
│   ├── models/         # PyTorch Geometric GNN architectures
│   ├── training/       # Training loops and Lightning modules
│   └── eval/           # Evaluation metrics and explainability tools
├── tests/              # Unit and integration tests
└── environment.yml     # Conda environment definition
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (optional but recommended for GNN training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-gnn-transcriptomics.git
   cd brain-gnn-transcriptomics
   ```

2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate brain-gnn
   # Ensure deploy dependencies are installed if using pip fallback
   python -m pip install fastapi uvicorn
   ```

### 🔬 Usage Pipeline

#### 1. Data Ingestion
Download ROSMAP data (or generate synthetic fallback if no credentials):
```bash
python scripts/download_rosmap.py
```

#### 2. Preprocessing
Filter, normalize, and select Highly Variable Genes (HVGs):
```bash
python scripts/preprocess_scRNA.py
```

#### 3. Graph Construction
Build Cell-Cell (KNN) and Gene-Gene (Co-expression) graphs:
```bash
python scripts/build_graphs.py
```

#### 4. Training (Distributed)
Train the GNN model using PyTorch Lightning:
```bash
python scripts/train.py --model cell --data_path data/processed/graphs/cell_graph.pt --epochs 10
```

#### 5. Deployment (API)
Serve the trained model via FastAPI:
```bash
# Start the server (autoreload enabled)
python -m uvicorn deploy.fastapi_app:app --reload
```
API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📊 Scientific Context
Alzheimer's Disease (AD) is characterized by complex cellular interactions and gene expression changes. Traditional flat-file analysis misses the **relational structure** of biological systems. By modeling cells and genes as graphs, we leverage **Graph Neural Networks** to learn:
1. **Manifolds of Disease**: How healthy cells transition to a disease state in latent space.
2. **Gene Regulatory Networks**: Which gene modules drive this transition.

**Primary Dataset**: [ROSMAP (Religious Orders Study and Memory and Aging Project)](https://radc.rush.edu/) - Single-cell transcriptomics of the aging human prefrontal cortex.

---

## 🛠️ Tech Stack
- **Deep Learning**: PyTorch, PyTorch Geometric, PyTorch Lightning
- **Bioinformatics**: Scanpy, Anndata
- **Data Engineering**: Pandas, NumPy, H5py
- **Deployment**: Docker, FastAPI
