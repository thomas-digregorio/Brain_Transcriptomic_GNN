import h5py
import scanpy as sc
import logging

logging.basicConfig(level=logging.INFO)

file_path = "data/processed/sea_ad_proc.h5ad"

print(f"Inspecting {file_path}...")

# 1. Check with h5py (Physical Storage)
try:
    with h5py.File(file_path, 'r') as f:
        if 'var' in f:
            print("Group 'var' exists.")
            print("Keys in 'var':", list(f['var'].keys()))
            if 'highly_variable' in f['var']:
                print("SUCCESS: 'highly_variable' dataset found in file.")
            else:
                print("FAILURE: 'highly_variable' NOT found in 'var' group.")
        else:
            print("FAILURE: Group 'var' missing.")
            
        if 'uns' in f and 'pca' in f['uns']:
             print("PCA found in uns.")
except Exception as e:
    print(f"h5py error: {e}")

# 2. Check with Scanpy (Logical View)
try:
    adata = sc.read_h5ad(file_path, backed='r')
    print("Scanpy loaded.")
    print("adata.var columns:", adata.var.columns.tolist())
    if 'highly_variable' in adata.var.columns:
        print("Scanpy sees 'highly_variable'.")
    else:
        print("Scanpy DOES NOT see 'highly_variable'.")
except Exception as e:
    print(f"Scanpy error: {e}")
