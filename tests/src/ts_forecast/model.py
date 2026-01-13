from __future__ import annotations



import torch

import torch.nn as nn





class LSTMForecast(nn.Module):

    """

    Many-to-many: takes seq_len timesteps with D features, outputs horizon steps for target.

    """



    def __init__(

        self,

        input_size: int,

        hidden_size: int,

        num_layers: int,

        dropout: float,

        horizon: int,

    ) -> None:

        super().__init__()

        self.horizon = horizon

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            dropout=dropout if num_layers > 1 else 0.0,

            batch_first=True,

        )

        self.head = nn.Sequential(

            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, horizon),

        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: [B, seq_len, D]

        out, _ = self.lstm(x)             # [B, seq_len, H]

        last = out[:, -1, :]              # [B, H]

        y = self.head(last)               # [B, horizon]

        return y


