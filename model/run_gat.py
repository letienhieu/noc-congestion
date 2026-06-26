from __future__ import annotations
import argparse
import csv
import sys
import types
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from neo4j import GraphDatabase
except ImportError:
    _m = types.ModuleType('neo4j')
    _m.GraphDatabase = None
    sys.modules['neo4j'] = _m
try:
    from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
except ImportError:
    _p = types.ModuleType('torch_geometric_temporal')
    _s = types.ModuleType('torch_geometric_temporal.signal')

    class DynamicGraphTemporalSignal:

        def __init__(self, **kw):
            self.__dict__.update(kw)
    _s.DynamicGraphTemporalSignal = DynamicGraphTemporalSignal
    _p.signal = _s
    sys.modules['torch_geometric_temporal'] = _p
    sys.modules['torch_geometric_temporal.signal'] = _s
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from model.stgnn import _shift_edge_index_for_window
from model.train import train_model, eval_trained, TrainConfig
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
NUM_FEATURES = 5

class STGNNGat(nn.Module):

    def __init__(self, *, window, num_base_features=5, hidden=32, heads=4, dropout=0.1):
        super().__init__()
        self.window = window
        self.num_base_features = num_base_features
        self.dropout = dropout
        self.gat1 = GATConv(num_base_features, hidden, heads=heads, concat=False)
        self.gat2 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_weight=None):
        N = x.shape[0]
        W = self.window
        Fb = self.num_base_features
        x_seq = x.view(N, W, Fb).permute(1, 0, 2).reshape(W * N, Fb)
        big_edge = _shift_edge_index_for_window(edge_index, N, W)
        h = F.elu(self.gat1(x_seq, big_edge))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.gat2(h, big_edge))
        h = h.view(W, N, -1).permute(1, 0, 2).contiguous()
        gru_out, _ = self.gru(h)
        return self.head(gru_out[:, -1, :]).squeeze(-1)
MESHES = {'4x4': dict(mesh_id='mesh_4x4', val_runs=('mesh_4x4_uniform_ir050', 'mesh_4x4_neighbor_ir050'), test_runs=('mesh_4x4_uniform_ir060', 'mesh_4x4_transpose_ir060', 'mesh_4x4_hotspot_ir060'), excl=('mesh_4x4_uniform_smoke',)), '8x8': dict(mesh_id='mesh_8x8', val_runs=('mesh_8x8_uniform_ir020', 'mesh_8x8_neighbor_ir020'), test_runs=('mesh_8x8_uniform_ir040', 'mesh_8x8_transpose_ir040', 'mesh_8x8_hotspot_ir040'), excl=())}

def _read_run_csv(run_id: str):
    path = DATA_DIR / run_id / 'router_timeseries.csv'
    rows = []
    with path.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    num_nodes = max((int(r['router_id']) for r in rows)) + 1
    n_samples = max((int(r['sample_idx']) for r in rows)) + 1
    feats = np.zeros((n_samples, num_nodes, NUM_FEATURES), dtype=np.float32)
    targ = np.zeros((n_samples, num_nodes), dtype=np.float32)
    coords = {}
    for r in rows:
        t = int(r['sample_idx'])
        rid = int(r['router_id'])
        feats[t, rid, 0] = float(r['injected']) / 100.0
        feats[t, rid, 1] = float(r['ejected']) / 100.0
        feats[t, rid, 2] = float(r['received_total']) / 100.0
        feats[t, rid, 3] = float(r['sent_total']) / 100.0
        feats[t, rid, 4] = float(r['stored_total']) / 100.0
        targ[t, rid] = float(r['buffer_occupancy_norm'])
        coords[rid] = (int(r['x']), int(r['y']))
    return (feats, targ, coords)

def _edge_index_from_coords(coords: dict) -> np.ndarray:
    pos2id = {v: k for k, v in coords.items()}
    pairs = []
    for rid, (x, y) in coords.items():
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = pos2id.get((x + dx, y + dy))
            if nb is not None:
                pairs.append((rid, nb))
    pairs.sort()
    return np.asarray(pairs, dtype=np.int64).T

def _windowed_items(feats, targ, edge_index_t, window, device):
    items = []
    T, N, Fb = feats.shape
    for t in range(window - 1, T - 1):
        block = feats[t - window + 1:t + 1]
        x = block.transpose(1, 0, 2).reshape(N, window * Fb)
        y = targ[t + 1]
        items.append((torch.from_numpy(x).to(device), edge_index_t, None, torch.from_numpy(y.copy()).to(device)))
    return items

def build_items_from_csv(spec, window, device):
    run_ids = sorted((p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith(spec['mesh_id'] + '_')))
    run_ids = [r for r in run_ids if r not in set(spec['excl'])]
    val_set, test_set = (set(spec['val_runs']), set(spec['test_runs']))
    splits = {'train': [], 'val': [], 'test': []}
    edge_index_t = None
    num_nodes = None
    for rid in run_ids:
        feats, targ, coords = _read_run_csv(rid)
        if edge_index_t is None:
            ei = _edge_index_from_coords(coords)
            edge_index_t = torch.from_numpy(ei).to(device)
            num_nodes = len(coords)
        key = 'val' if rid in val_set else 'test' if rid in test_set else 'train'
        splits[key] += _windowed_items(feats, targ, edge_index_t, window, device)
    assert splits['train'] and splits['val'] and splits['test'], 'split rỗng!'
    return (splits, num_nodes)

def build_items_neo4j(spec, window, device):
    from model.dataset import build_split
    from model.train import _signal_to_tensors
    split = build_split(window=window, mesh_id=spec['mesh_id'], val_runs=spec['val_runs'], test_runs=spec['test_runs'], exclude_runs=spec['excl'])
    return ({'train': _signal_to_tensors(split.train, device), 'val': _signal_to_tensors(split.val, device), 'test': _signal_to_tensors(split.test, device)}, split.num_nodes)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 42, 123])
    p.add_argument('--meshes', nargs='+', default=['4x4', '8x8'])
    p.add_argument('--source', choices=['csv', 'neo4j'], default='csv')
    p.add_argument('--heads', type=int, default=4)
    args = p.parse_args(argv)
    device = torch.device('cpu')
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for mk in args.meshes:
        spec = MESHES[mk]
        print(f'\n===== GAT on {mk} (source={args.source}) =====', flush=True)
        if args.source == 'csv':
            splits, num_nodes = build_items_from_csv(spec, 5, device)
        else:
            splits, num_nodes = build_items_neo4j(spec, 5, device)
        tr, va, te = (splits['train'], splits['val'], splits['test'])
        print(f'  nodes={num_nodes} train={len(tr)} val={len(va)} test={len(te)}', flush=True)
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = STGNNGat(window=5, hidden=32, heads=args.heads, dropout=0.1)
            c = TrainConfig(window=5, epochs=15, lr=0.005, weight_decay=0.0005, hidden=32, seed=seed)
            train_model(f'GAT-{mk}-s{seed}', m, tr, va, cfg=c, device=device)
            rmse, mae, _, _ = eval_trained(m, te, target_scale=c.target_scale)
            rows.append({'mesh': mk, 'seed': seed, 'test_rmse': rmse, 'test_mae': mae})
            print(f'  {mk} seed={seed}: RMSE={rmse:.5f} MAE={mae:.5f}', flush=True)
    out = METRICS_DIR / 'gat.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['mesh', 'seed', 'test_rmse', 'test_mae'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nsaved {out}')
    for mk in args.meshes:
        r = [x['test_rmse'] for x in rows if x['mesh'] == mk]
        a = [x['test_mae'] for x in rows if x['mesh'] == mk]
        if r:
            print(f'GAT {mk}: RMSE {np.mean(r):.5f} ± {np.std(r):.5f}  MAE {np.mean(a):.5f} ± {np.std(a):.5f}  (n={len(r)})')
if __name__ == '__main__':
    main()
