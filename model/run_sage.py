from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.dataset import build_split, signal_size
from model.stgnn import _shift_edge_index_for_window
from model.train import _signal_to_tensors, train_model, eval_trained, TrainConfig
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'

class STGNNSage(nn.Module):

    def __init__(self, *, window, num_base_features=5, hidden=32, dropout=0.1):
        super().__init__()
        self.window = window
        self.num_base_features = num_base_features
        self.dropout = dropout
        self.sage1 = SAGEConv(num_base_features, hidden, aggr='max')
        self.sage2 = SAGEConv(hidden, hidden, aggr='max')
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_weight=None):
        N = x.shape[0]
        W = self.window
        Fb = self.num_base_features
        x_seq = x.view(N, W, Fb).permute(1, 0, 2).reshape(W * N, Fb)
        big_edge = _shift_edge_index_for_window(edge_index, N, W)
        h = F.relu(self.sage1(x_seq, big_edge))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.sage2(h, big_edge))
        h = h.view(W, N, -1).permute(1, 0, 2).contiguous()
        gru_out, _ = self.gru(h)
        return self.head(gru_out[:, -1, :]).squeeze(-1)
MESHES = {'4x4': dict(mesh_id='mesh_4x4', val_runs=('mesh_4x4_uniform_ir050', 'mesh_4x4_neighbor_ir050'), test_runs=('mesh_4x4_uniform_ir060', 'mesh_4x4_transpose_ir060', 'mesh_4x4_hotspot_ir060'), excl=('mesh_4x4_uniform_smoke',)), '8x8': dict(mesh_id='mesh_8x8', val_runs=('mesh_8x8_uniform_ir020', 'mesh_8x8_neighbor_ir020'), test_runs=('mesh_8x8_uniform_ir040', 'mesh_8x8_transpose_ir040', 'mesh_8x8_hotspot_ir040'), excl=())}

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 42, 123])
    p.add_argument('--meshes', nargs='+', default=['4x4', '8x8'])
    args = p.parse_args(argv)
    device = torch.device('cpu')
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(window=5, epochs=15, lr=0.005, weight_decay=0.0005, hidden=32, seed=42)
    rows = []
    for mk in args.meshes:
        spec = MESHES[mk]
        print(f'\n===== SAGE-max on {mk} =====')
        split = build_split(window=cfg.window, mesh_id=spec['mesh_id'], val_runs=spec['val_runs'], test_runs=spec['test_runs'], exclude_runs=spec['excl'])
        tr = _signal_to_tensors(split.train, device)
        va = _signal_to_tensors(split.val, device)
        te = _signal_to_tensors(split.test, device)
        print(f'  nodes={split.num_nodes} train={signal_size(split.train)} val={signal_size(split.val)} test={signal_size(split.test)}')
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = STGNNSage(window=cfg.window, hidden=cfg.hidden, dropout=0.1)
            c = TrainConfig(window=cfg.window, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay, hidden=cfg.hidden, seed=seed)
            train_model(f'SAGEmax-{mk}-s{seed}', m, tr, va, cfg=c, device=device)
            rmse, mae, _, _ = eval_trained(m, te, target_scale=c.target_scale)
            rows.append({'mesh': mk, 'seed': seed, 'test_rmse': rmse, 'test_mae': mae})
            print(f'  {mk} seed={seed}: RMSE={rmse:.5f} MAE={mae:.5f}')
    out = METRICS_DIR / 'sage_max.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['mesh', 'seed', 'test_rmse', 'test_mae'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nsaved {out}')
    for mk in args.meshes:
        r = [x['test_rmse'] for x in rows if x['mesh'] == mk]
        a = [x['test_mae'] for x in rows if x['mesh'] == mk]
        print(f'SAGE-max {mk}: RMSE {np.mean(r):.5f} ± {np.std(r):.5f}  MAE {np.mean(a):.5f} ± {np.std(a):.5f}  (n={len(r)})')
if __name__ == '__main__':
    main()
