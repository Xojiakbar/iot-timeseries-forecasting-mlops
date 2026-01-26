import numpy as np
import torch

from ts_forecast.model import LSTMForecast
from ts_forecast.predict import predict_next, LoadedModel


def test_predict_next_output_shape():
    seq_len = 16
    D = 3
    horizon = 4

    model = LSTMForecast(input_size=D, hidden_size=8, num_layers=1, dropout=0.0, horizon=horizon)
    device = torch.device("cpu")
    model.eval()

    loaded = LoadedModel(
        model=model,
        device=device,
        standardizer_x=None,
        standardizer_y=None,
        input_size=D,
        horizon=horizon,
    )

    series = np.random.randn(seq_len, D).astype(np.float32)
    pred = predict_next(loaded, series)
    assert pred.shape == (horizon,)
