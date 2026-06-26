from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def _shift_edge_index_for_window(edge_index: torch.Tensor, num_nodes: int, window: int) -> torch.Tensor:
    repeats = []
    for w in range(window):
        repeats.append(edge_index + w * num_nodes)
    return torch.cat(repeats, dim=1)

class STGNN(nn.Module):

    def __init__(self, *, window: int, num_base_features: int=5, gcn_hidden: int=32, gru_hidden: int=32, dropout: float=0.1):
        super().__init__()
        self.window = window
        self.num_base_features = num_base_features
        self.dropout = dropout
        self.gcn1 = GCNConv(num_base_features, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(gru_hidden, gru_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(gru_hidden, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None=None) -> torch.Tensor:
        N = x.shape[0]
        W = self.window
        F_in = self.num_base_features
        x_seq = x.view(N, W, F_in).permute(1, 0, 2).reshape(W * N, F_in)
        big_edge = _shift_edge_index_for_window(edge_index, N, W)
        big_w = edge_weight.repeat(W) if edge_weight is not None else None
        h = self.gcn1(x_seq, big_edge, big_w)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn2(h, big_edge, big_w)
        h = F.relu(h)
        h = h.view(W, N, -1).permute(1, 0, 2).contiguous()
        gru_out, _ = self.gru(h)
        last_h = gru_out[:, -1, :]
        return self.head(last_h).squeeze(-1)
