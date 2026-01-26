from __future__ import annotations



import json

import os

import random

from dataclasses import dataclass

from pathlib import Path

from typing import Any, dict



import numpy as np

import torch

import yaml





def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)





def ensure_dir(path: str | Path) -> None:

    Path(path).parent.mkdir(parents=True, exist_ok=True)





def read_yaml(path: str | Path) -> dict[str, Any]:

    with open(path, "r", encoding="utf-8") as f:

        return yaml.safe_load(f)





def write_json(path: str | Path, obj: dict[str, Any]) -> None:

    ensure_dir(path)

    with open(path, "w", encoding="utf-8") as f:

        json.dump(obj, f, indent=2)





def resolve_device(device_cfg: str) -> torch.device:

    if device_cfg == "cpu":

        return torch.device("cpu")

    if device_cfg == "cuda":

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # auto

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")





@dataclass

class Standardizer:

    mean: np.ndarray

    std: np.ndarray



    def transform(self, x: np.ndarray) -> np.ndarray:

        return (x - self.mean) / (self.std + 1e-8)



    def inverse_transform(self, x: np.ndarray) -> np.ndarray:

        return x * (self.std + 1e-8) + self.mean


