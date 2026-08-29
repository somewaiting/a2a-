
import torch

# Check if GPU or MPS is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA GPU is enabled: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS GPU is enabled.")
else:
    raise OSError(
        "No GPU or MPS device found. Please check your environment and ensure GPU or MPS support is configured."
    )