import torch
import torch.nn as nn


class SOCLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_layers=1,
        dropout=0.2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(

            nn.Linear(hidden_size, 16),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(16, 1)

        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]

        output = self.fc(final_hidden)

        return output