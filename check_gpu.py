import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys

def check_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA Available: YES")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test a small calculation
        try:
            x = torch.rand(5000, 5000).cuda()
            y = torch.rand(5000, 5000).cuda()
            z = torch.matmul(x, y)
            print("Success! Performed 5000x5000 matrix multiplication on GPU.")
        except Exception as e:
            print(f"Failed to run calculation on GPU: {e}")
            
    else:
        print("CUDA Available: NO")
        print("This environment cannot use the GPU.")

if __name__ == "__main__":
    check_gpu()
