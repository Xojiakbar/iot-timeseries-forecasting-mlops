from __future__ import annotations



import argparse

from dataclasses import asdict

from pathlib import Path

from typing import Dict, Any



import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset



from ts_forecast.data import load_dataset

from ts_forecast.model import LSTMForecast

from ts_forecast.utils import read_yaml, resolve_device, set_seed, ensure_dir, write_json





def train_one_epoch(model, loader, optimizer, criterion, device) -> float:

    model.train()

    total_loss = 0.0

    n = 0

    for xb, yb in loader:

        xb = xb.to(device)

        yb = yb.to(device)



        optimizer.zero_grad(set_to_none=True)

        pred = model(xb)

        loss = criterion(pred, yb)

        loss.backward()

        optimizer.step()



        total_loss += loss.item() * xb.size(0)

        n += xb.size(0)

    return total_loss / max(n, 1)





@torch.no_grad()

def eval_epoch(model, loader, criterion, device) -> float:

    model.eval()

    total_loss = 0.0

    n = 0

    for xb, yb in loader:

        xb = xb.to(device)

        yb = yb.to(device)

        pred = model(xb)

        loss = criterion(pred, yb)

        total_loss += loss.item() * xb.size(0)

        n += xb.size(0)

    return total_loss / max(n, 1)





def main(config_path: str) -> None:

    cfg = read_yaml(config_path)

    data_cfg = cfg["data"]

    train_cfg = cfg["train"]



    set_seed(int(data_cfg.get("seed", 42)))

    device = resolve_device(train_cfg.get("device", "auto"))



    bundle = load_dataset(

        csv_path=data_cfg["csv_path"],

        dataset_name=data_cfg.get("dataset_name", "ett_small"),

        target_col=data_cfg["target_col"],

        feature_cols=data_cfg.get("feature_cols", []),

        seq_len=int(data_cfg["seq_len"]),

        horizon=int(data_cfg["horizon"]),

        train_ratio=float(data_cfg["train_ratio"]),

        val_ratio=float(data_cfg["val_ratio"]),

        standardize=bool(data_cfg.get("standardize", True)),

        seed=int(data_cfg.get("seed", 42)),

    )



    x_train = torch.from_numpy(bundle.x_train)

    y_train = torch.from_numpy(bundle.y_train)

    x_val = torch.from_numpy(bundle.x_val)

    y_val = torch.from_numpy(bundle.y_val)



    train_loader = DataLoader(

        TensorDataset(x_train, y_train),

        batch_size=int(train_cfg["batch_size"]),

        shuffle=True,

        drop_last=False,

    )

    val_loader = DataLoader(

        TensorDataset(x_val, y_val),

        batch_size=int(train_cfg["batch_size"]),

        shuffle=False,

        drop_last=False,

    )



    input_size = bundle.x_train.shape[-1]

    horizon = bundle.y_train.shape[-1]



    model = LSTMForecast(

        input_size=input_size,

        hidden_size=int(train_cfg["hidden_size"]),

        num_layers=int(train_cfg["num_layers"]),

        dropout=float(train_cfg["dropout"]),

        horizon=horizon,

    ).to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))

    criterion = nn.MSELoss()



    best_val = float("inf")

    history = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):

        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        va_loss = eval_epoch(model, val_loader, criterion, device)



        history.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": va_loss})

        print(f"Epoch {epoch:02d} | train_mse={tr_loss:.6f} | val_mse={va_loss:.6f}")



        if va_loss < best_val:

            best_val = va_loss

            save_path = Path(train_cfg["save_path"])

            ensure_dir(save_path)

            torch.save(

                {

                    "model_state_dict": model.state_dict(),

                    "config": cfg,

                    "input_size": input_size,

                    "horizon": horizon,

                    "feature_names": bundle.feature_names,

                    "target_name": bundle.target_name,

                    "standardizer_x": None if bundle.standardizer_x is None else asdict(bundle.standardizer_x),

                    "standardizer_y": None if bundle.standardizer_y is None else asdict(bundle.standardizer_y),

                },

                save_path,

            )



    # Save training history as report (optional but nice)

    write_json("reports/train_history.json", {"history": history, "best_val_mse": best_val})

    print(f"Saved best model to: {train_cfg['save_path']}")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/default.yaml")

    args = parser.parse_args()

    main(args.config)


