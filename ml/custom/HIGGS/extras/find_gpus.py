import os
import torch
os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def list_gpus():
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Torch CUDA available:", torch.cuda.is_available())
    print("Torch CUDA device count:", torch.cuda.device_count())
    print("Torch version:", torch.__version__)
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    print(f"CUDA version: {torch.version.cuda}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if __name__ == "__main__":
    list_gpus()
    
print(torch.version.cuda)       # Should show CUDA version if CUDA-enabled
print(torch.backends.cudnn.enabled)  # Should be True if CUDA-enabled
