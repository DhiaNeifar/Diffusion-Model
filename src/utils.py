import os
import json
import random
from typing import Any, Dict, Optional

import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def print_device_info(device: torch.device) -> None:
    print("device:", device)
    print("torch:", torch.__version__)
    print("mps built:", torch.backends.mps.is_built())
    print("mps available:", torch.backends.mps.is_available())

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for name in os.listdir(ckpt_dir):
        if name.endswith(".pt"):
            candidates.append(os.path.join(ckpt_dir, name))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int, meta: Dict[str, Any]) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "meta": meta,
    }
    torch.save(payload, path)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
