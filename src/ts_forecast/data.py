from __future__ import annotations



import io

from dataclasses import dataclass

from pathlib import Path

from typing import Tuple



import numpy as np

import pandas as pd



from ts_forecast.utils import Standardizer, ensure_dir, set_seed





ETT_SMALL_URL = (

    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

)





def _try_download_ett(csv_path: Path) -> bool:

    """

    Try to download ETT-small (ETTh1.csv) without extra dependencies.

    If it fails (no internet), return False.

    """

    try:

        import urllib.request



        ensure_dir(csv_path)

        with urllib.request.urlopen(ETT_SMALL_URL, timeout=15) as resp:

            content = resp.read()

        df = pd.read_csv(io.BytesIO(content))

        df.to_csv(csv_path, index=False)

        return True

    except Exception:

        return False





def _make_synthetic(csv_path: Path, n_rows: int = 20000, seed: int = 42) -> None:

    """

    Generate a synthetic IoT-like multivariate time-series dataset with a target column 'OT'.

    """

    rng = np.random.default_rng(seed)

    t = np.arange(n_rows)



    # Signals

    daily = np.sin(2 * np.pi * t / 96.0)

    weekly = np.sin(2 * np.pi * t / (96.0 * 7))

    noise = rng.normal(0, 0.2, size=n_rows)



    # Features

    f1 = daily + 0.1 * rng.normal(size=n_rows)

    f2 = weekly + 0.1 * rng.normal(size=n_rows)

    f3 = 0.5 * daily + 0.2 * weekly + rng.normal(0, 0.15, size=n_rows)



    # Target 'OT' (some mixture + noise)

    ot = 0.7 * f1 + 0.2 * f2 + 0.1 * f3 + noise



    df = pd.DataFrame(

        {

            "date": pd.date_range("2021-01-01", periods=n_rows, freq="15min"),

            "F1": f1,

            "F2": f2,

            "F3": f3,

            "OT": ot,

        }

    )

    ensure_dir(csv_path)

    df.to_csv(csv_path, index=False)





@dataclass

class DataBundle:

    x_train: np.ndarray

    y_train: np.ndarray

    x_val: np.ndarray

    y_val: np.ndarray

    x_test: np.ndarray

    y_test: np.ndarray

    standardizer_x: Standardizer | None

    standardizer_y: Standardizer | None

    feature_names: list[str]

    target_name: str





def build_windows(

    values: np.ndarray, seq_len: int, horizon: int, target_idx: int

) -> Tuple[np.ndarray, np.ndarray]:

    """

    values: [T, D]

    Returns:

      X: [N, seq_len, D]

      y: [N, horizon]  (target only)

    """

    T, D = values.shape

    N = T - seq_len - horizon + 1

    if N <= 0:

        raise ValueError("Not enough data to create windows. Reduce seq_len/horizon.")



    X = np.zeros((N, seq_len, D), dtype=np.float32)

    y = np.zeros((N, horizon), dtype=np.float32)

    for i in range(N):

        X[i] = values[i : i + seq_len]

        y[i] = values[i + seq_len : i + seq_len + horizon, target_idx]

    return X, y





def load_dataset(

    csv_path: str,

    dataset_name: str,

    target_col: str,

    feature_cols: list[str] | None,

    seq_len: int,

    horizon: int,

    train_ratio: float,

    val_ratio: float,

    standardize: bool,

    seed: int,

) -> DataBundle:

    set_seed(seed)

    path = Path(csv_path)



    if not path.exists():

        ok = False

        if dataset_name.lower() in {"ett_small", "ett", "etth1"}:

            ok = _try_download_ett(path)

        if not ok:

            _make_synthetic(path, seed=seed)



    df = pd.read_csv(path)



    # Handle date column if exists

    if "date" in df.columns:

        # Keep date for potential future use, but not as numeric feature

        df = df.copy()



    if target_col not in df.columns:

        raise ValueError(f"target_col='{target_col}' not found in CSV columns: {df.columns}")



    # Choose feature columns

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if feature_cols is None or len(feature_cols) == 0:

        feature_cols = [c for c in numeric_cols if c != target_col]

    else:

        for c in feature_cols:

            if c not in df.columns:

                raise ValueError(f"feature_col '{c}' not found in CSV columns.")



    used_cols = feature_cols + [target_col]

    used_df = df[used_cols].copy()



    values = used_df.to_numpy(dtype=np.float32)

    target_idx = used_cols.index(target_col)



    X, y = build_windows(values, seq_len=seq_len, horizon=horizon, target_idx=target_idx)



    # Split into train/val/test by time order (not random) to be realistic

    N = X.shape[0]

    n_train = int(N * train_ratio)

    n_val = int(N * val_ratio)

    n_test = N - n_train - n_val



    x_train = X[:n_train]

    y_train = y[:n_train]

    x_val = X[n_train : n_train + n_val]

    y_val = y[n_train : n_train + n_val]

    x_test = X[n_train + n_val :]

    y_test = y[n_train + n_val :]



    sx = sy = None

    if standardize:

        # Standardize features+target jointly for X (all dims), and target for y

        # X: standardize each dimension using train statistics across all timesteps

        flat_train = x_train.reshape(-1, x_train.shape[-1])

        mean_x = flat_train.mean(axis=0)

        std_x = flat_train.std(axis=0)

        sx = Standardizer(mean=mean_x, std=std_x)



        x_train = sx.transform(x_train)

        x_val = sx.transform(x_val)

        x_test = sx.transform(x_test)



        # y is target only; standardize using train target stats

        mean_y = y_train.mean(axis=0)  # per-horizon mean

        std_y = y_train.std(axis=0)

        sy = Standardizer(mean=mean_y, std=std_y)



        y_train = sy.transform(y_train)

        y_val = sy.transform(y_val)

        y_test = sy.transform(y_test)



    return DataBundle(

        x_train=x_train,

        y_train=y_train,

        x_val=x_val,

        y_val=y_val,

        x_test=x_test,

        y_test=y_test,

        standardizer_x=sx,

        standardizer_y=sy,

        feature_names=feature_cols,

        target_name=target_col,

    )


