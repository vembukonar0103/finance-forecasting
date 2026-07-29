import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return "mps"

    return "cpu"

if __name__ == "__main__":
    print(f"Detected device: {get_device()}")
