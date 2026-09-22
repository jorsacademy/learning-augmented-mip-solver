from __future__ import annotations

from pathlib import Path

import torch

from .model import PresolveSelector


def save_checkpoint(model: PresolveSelector, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first_layer = model.net[0]
    hidden_dim = int(first_layer.out_features)
    torch.save({"hidden_dim": hidden_dim, "model_state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path) -> PresolveSelector:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = PresolveSelector(hidden_dim=int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
