from __future__ import annotations



from dataclasses import dataclass

from typing import Any, Dict, Tuple



import numpy as np

import torch



from ts_forecast.model import LSTMForecast

from ts_forecast.utils import Standardizer





@dataclass

class LoadedModel:

    model: LSTMForecast

    device: torch.device

    standardizer_x: Standardizer | None

    standardizer_y: Standardizer | None

    input_size: int

    horizon: int





def load_model(path: str, device: torch.device) -> LoadedModel:

    ckpt = torch.load(path, map_location="cpu")

    input_size = int(ckpt["input_size"])

    horizon = int(ckpt["horizon"])



    cfg = ckpt["config"]

    train_cfg = cfg["train"]



    model = LSTMForecast(

        input_size=input_size,

        hidden_size=int(train_cfg["hidden_size"]),

        num_layers=int(train_cfg["num_layers"]),

        dropout=float(train_cfg["dropout"]),

        horizon=horizon,

    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()



    sx = None

    sy = None

    if ckpt.get("standardizer_x") is not None:

        sx = Standardizer(

            mean=np.array(ckpt["standardizer_x"]["mean"], dtype=np.float32),

            std=np.array(ckpt["standardizer_x"]["std"], dtype=np.float32),

        )

    if ckpt.get("standardizer_y") is not None:

        sy = Standardizer(

            mean=np.array(ckpt["standardizer_y"]["mean"], dtype=np.float32),

            std=np.array(ckpt["standardizer_y"]["std"], dtype=np.float32),

        )



    return LoadedModel(model=model, device=device, standardizer_x=sx, standardizer_y=sy,

                      input_size=input_size, horizon=horizon)





@torch.no_grad()

def predict_next(loaded: LoadedModel, series: np.ndarray) -> np.ndarray:

    """

    series: [seq_len, D] float32

    returns: [horizon] prediction (in original scale if standardizer_y exists)

    """

    x = series.astype(np.float32)



    if loaded.standardizer_x is not None:

        x = loaded.standardizer_x.transform(x)



    xb = torch.from_numpy(x).unsqueeze(0).to(loaded.device)  # [1, seq_len, D]

    pred = loaded.model(xb).cpu().numpy()[0]  # [horizon]



    if loaded.standardizer_y is not None:

        pred = loaded.standardizer_y.inverse_transform(pred)



    return pred


