import os
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: str, epoch: int, accuracy: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "accuracy": accuracy,
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
