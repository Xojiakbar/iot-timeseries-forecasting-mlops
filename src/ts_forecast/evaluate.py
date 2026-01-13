from __future__ import annotations



import argparse

from dataclasses import dataclass

from pathlib import Path



import numpy as np

import torch



from ts_forecast.data import load_dataset

from ts_forecast.model import LSTMForecast

from ts_forecast.utils import read_yaml, resolve_device, write_json





def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))





def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return float(np.mean(np.abs(y_true - y_pred)))





@torch.no_grad()

def main(config_path: str) -> None:

    cfg = read_yaml(config_path)

    data_cfg = cfg["data"]

    train_cfg = cfg["train"]

    eval_cfg = cfg["eval"]



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



    ckpt = torch.load(train_cfg["save_path"], map_location="cpu")

    input_size = int(ckpt["input_size"])

    horizon = int(ckpt["horizon"])



    model = LSTMForecast(

        input_size=input_size,

        hidden_size=int(train_cfg["hidden_size"]),

        num_layers=int(train_cfg["num_layers"]),

        dropout=float(train_cfg["dropout"]),

        horizon=horizon,

    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()



    x_test = torch.from_numpy(bundle.x_test).to(device)

    y_test = bundle.y_test



    y_pred = model(x_test).detach().cpu().numpy()



    # If standardized, invert for human-friendly metrics (optional)

    if bundle.standardizer_y is not None:

        y_test_real = bundle.standardizer_y.inverse_transform(y_test)

        y_pred_real = bundle.standardizer_y.inverse_transform(y_pred)

    else:

        y_test_real = y_test

        y_pred_real = y_pred



    metrics = {

        "rmse": rmse(y_test_real, y_pred_real),

        "mae": mae(y_test_real, y_pred_real),

        "n_test_windows": int(y_test.shape[0]),

        "horizon": int(y_test.shape[1]),

        "target": bundle.target_name,

        "dataset": data_cfg.get("dataset_name", "ett_small"),

    }



    write_json(eval_cfg["metrics_path"], metrics)

    print("Saved metrics:", metrics)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/default.yaml")

    args = parser.parse_args()

    main(args.config)


