import numpy as np



from ts_forecast.data import build_windows





def test_build_windows_shapes():

    T, D = 200, 4

    seq_len = 20

    horizon = 5

    target_idx = 3

    values = np.random.randn(T, D).astype(np.float32)



    X, y = build_windows(values, seq_len=seq_len, horizon=horizon, target_idx=target_idx)

    assert X.ndim == 3

    assert y.ndim == 2

    assert X.shape[1] == seq_len

    assert X.shape[2] == D

    assert y.shape[1] == horizon

    assert X.shape[0] == y.shape[0]


