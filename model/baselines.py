from __future__ import annotations
import torch
import torch.nn as nn

class PersistenceBaseline(nn.Module):

    def __init__(self, *, window: int, num_base_features: int=5, stored_channel: int=4, capacity: float=160.0, feature_scale: float=100.0):
        super().__init__()
        self.window = window
        self.num_base_features = num_base_features
        self.stored_channel = stored_channel
        self.rescale = feature_scale / capacity

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None) -> torch.Tensor:
        last_block_offset = (self.window - 1) * self.num_base_features
        stored_last = x[:, last_block_offset + self.stored_channel]
        return stored_last * self.rescale

class MLPBaseline(nn.Module):

    def __init__(self, *, in_features: int, hidden: int=64, dropout: float=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class GRUBaseline(nn.Module):

    def __init__(self, *, window: int, num_base_features: int=5, hidden: int=32, dropout: float=0.1):
        super().__init__()
        self.window = window
        self.num_features = num_base_features
        self.gru = nn.GRU(input_size=num_base_features, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None) -> torch.Tensor:
        N = x.shape[0]
        seq = x.view(N, self.window, self.num_features)
        out, _ = self.gru(seq)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
