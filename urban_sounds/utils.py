import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():  # for Apple Silicon Macs
        return "mps"
    else:
        return "cpu"
