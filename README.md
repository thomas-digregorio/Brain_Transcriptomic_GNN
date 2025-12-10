# brain-gnn-transcriptomics: GNNs on Single-Cell Human Brain Data for AD/CNS Insights


## Project Overview
**One-liner:** A **GPU-accelerated**, scalable Graph Neural Network (GNN) pipeline designed to analyze **1.2 million+ single cells** from the **Seattle Alzheimer’s Disease Brain Cell Atlas (SEA-AD)**.

This project overcomes the memory limitations of standard single-cell workflows by replacing CPU-bound operations with **custom PyTorch-based implementations**. It processes raw scRNA-seq data, constructs biological graphs, and trains GNNs to uncover molecular drivers of neurodegeneration—all on consumer-grade hardware (3090/4090) or cloud GPUs.

### Key Innovations & Scalability
- **Fully GPU-Accelerated Preprocessing**:
  - **Custom PyTorch HVG Selection**: Replaces slow CPU-based highly variable gene selection with parallelized GPU kernel operations.
  - **GPU PCA (Covariance Method)**: Implements out-of-core PCA using a streaming $XX^T$ covariance method in PyTorch (Float64 precision), enabling dimensionality reduction on datasets larger than RAM (32 GB SEA-AD data).
- **Memory-Efficient "Backed" Mode**: Uses `AnnData` in `backed='r+'` mode with chunked HDF5 reads/writes, ensuring memory usage stays flat regardless of dataset size.
- **Scalable Graph Construction**:
  - **Cell-Cell Graphs**: Efficient KNN graph building on PCA embeddings.
  - **Gene-Gene Graphs**: Co-expression networks derived from massive correlation matrices.
- **Production-Grade GNNs**:
  - **Deep Graph Infomax (DGI)** for unsupervised representation learning.
  - **GraphSAGE / GAT** with PyTorch Lightning + DDP for distributed training.

---

## Repository Structure
```text
brain-gnn-transcriptomics/
├── config/             # Configuration files
├── data/               # Data storage (Raw and Processed)
├── deploy/             # Deployment configs (Docker, K8s, FastAPI)
├── notebooks/          # Jupyter notebooks for Scalable EDA & Analysis
│   ├── 05_sea_ad_full_analysis.ipynb # End-to-End Analysis on 1.2M Cells
│   ├── 06_sea_ad_gene_analysis.ipynb # Gene Module Discovery
│   └── 07_sea_ad_scalable_eda.ipynb  # GPU-backed Visualization & QC
├── scripts/            # Executable scripts for pipeline
│   ├── preprocess_scRNA.py # GPU-based Filtering, Normalization, HVG, PCA
│   ├── build_graphs.py     # Graph construction
│   ├── train.py            # PyTorch Lightning Training Loop
│   └── download_sea_ad.py  # Data Ingestion
├── src/                # Source code modules
│   ├── data/           # Data loading and transformation
│   ├── models/         # GNN Architectures (GCN, GAT, GraphSAGE)
│   └── training/       # Training Logic
└── environment.yml     # Conda environment definition
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- **CUDA-enabled GPU** (Required for acceleration steps)
- 32GB+ RAM (Recommended for OS caching)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/thomas-digregorio/Brain_Transcriptomic_GNN.git
   cd Brain_Transcriptomic_GNN
   ```

2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate brain-gnn
   # Ensure PyTorch is installed with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Usage Pipeline

#### 1. Data Ingestion
Download the SEA-AD dataset (or subset):
```bash
python scripts/download_sea_ad.py
```

#### 2. GPU-Accelerated Preprocessing
Feature selection and dimensionality reduction on the full 1.2M cell dataset.
*Note: This script uses custom PyTorch kernels for HVG and PCA to avoid OOM errors.*
```bash
python scripts/preprocess_scRNA.py
```

#### 3. Graph Construction
Builds sparse adjacency matrices for cells and genes:
```bash
python scripts/build_graphs.py
```

#### 4. Training (Distributed)
Train the GNN model using PyTorch Lightning:
```bash
# Trains for 5000 epochs to ensure convergence
python scripts/train.py --model cell --data_path data/processed/graphs/cell_graph.pt --epochs 5000
```

#### 5. Analysis & Visualization
Use the provided notebooks in `notebooks/` to explore the results. `07_sea_ad_scalable_eda.ipynb` is specifically optimized to visualize millions of cells without crashing your browser.

---

## Scientific Context
Alzheimer's Disease (AD) is characterized by complex cellular interactions and gene expression changes. Traditional flat-file analysis misses the **relational structure** of biological systems. By modeling cells and genes as graphs, we leverage **Graph Neural Networks** to learn:
1. **Manifolds of Disease**: How healthy cells transition to a disease state in latent space.
2. **Gene Regulatory Networks**: Which gene modules drive this transition.

**Primary Dataset**: [SEA-AD (Seattle Alzheimer’s Disease Brain Cell Atlas)](https://sea-ad.org/) - Single-cell transcriptomics of the aging human cortex.

---

## Tech Stack

### Core Frameworks
- **[PyTorch](https://pytorch.org/)**: The primary deep learning framework used for all tensor operations and custom GPU kernels.
- **[PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/)**: Used for implementing GNN layers (GCN, GraphSAGE, GAT) and handling sparse graph data structures efficiently.
- **[PyTorch Lightning](https://www.lightning.ai/)**: Abstracts the training loop, enabling seamless switching between CPU/GPU and simplifying distributed training (DDP).

### Bioinformatics & Data Handling
- **[Scanpy](https://scanpy.readthedocs.io/)**: The standard toolkit for single-cell analysis (preprocessing, clustering, visualization).
- **[AnnData](https://anndata.readthedocs.io/)**: The underlying data structure for annotated data matrices, heavily utilized in `backed='r+'` mode for out-of-core access.
- **[H5py](https://www.h5py.org/)**: Vital for low-level, chunked read/write operations on HDF5 files, allowing processing of datasets larger than RAM.

### Acceleration & Math
- **CUDA**: Essential for accelerating tensor computations and custom kernels (HVG selection, PCA).
- **NumPy & Pandas**: Backbone for CPu-based data manipulation and metadata handling before GPU transfer.

### Deployment & Infrastructure
- **[FastAPI](https://fastapi.tiangolo.com/)**: High-performance web framework for serving trained GNN models as REST APIs.
- **[Docker](https://www.docker.com/)**: Ensures reproducibility by containerizing the entire environment, including CUDA dependencies.
