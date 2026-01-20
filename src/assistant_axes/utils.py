from pathlib import Path

import torch


def save_activations(activations: dict[int, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in activations.items()}, path)


def load_activations(path: Path) -> dict[int, torch.Tensor]:
    return torch.load(path, weights_only=True)
