"""Neural architectures used in the ETTh1 forecasting comparison."""
import math
import torch
from torch import nn

class BaselineLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, layers=2, horizon=24, dropout=.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden_size, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

class AdditiveAttention(nn.Module):
    def __init__(self, width, attention_width=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width, attention_width), nn.Tanh(),
                                   nn.Linear(attention_width, 1))
    def forward(self, sequence):
        weights = torch.softmax(self.score(sequence), dim=1)
        return torch.sum(weights * sequence, dim=1)

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, layers=2, horizon=24, dropout=.3):
        super().__init__()
        width = hidden_size * 2
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.attention = AdditiveAttention(width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, horizon))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.attention(out))

class PositionalEncoding(nn.Module):
    def __init__(self, width, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, width)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerForecaster(nn.Module):
    def __init__(self, input_size=7, width=128, heads=4, layers=2, horizon=24, dropout=.2):
        super().__init__()
        self.projection = nn.Linear(input_size, width)
        self.position = PositionalEncoding(width)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=256,
                    dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, horizon))
    def forward(self, x):
        x = self.position(self.projection(x))
        return self.head(self.encoder(x)[:, -1])
