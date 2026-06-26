from __future__ import annotations
import csv
import sys
import time
from pathlib import Path
import numpy as np
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.baselines import GRUBaseline, MLPBaseline, PersistenceBaseline
from model.dataset import build_split
from model.stgnn import STGNN
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'
WINDOW = 5
HIDDEN = 32
TARGET_SCALE = 100.0
N_WARMUP = 50
N_SAMPLES = 500

def time_model(model, snapshots) -> dict:
    model.eval()
    times_ms = []
    with torch.no_grad():
        for i, snap in enumerate(snapshots):
            if i >= N_WARMUP:
                break
            _ = model(snap[0], snap[1], snap[2])
    with torch.no_grad():
        for i, snap in enumerate(snapshots):
            if i >= N_SAMPLES:
                break
            t0 = time.perf_counter()
            _ = model(snap[0], snap[1], snap[2])
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    arr = np.array(times_ms)
    return {'samples': len(arr), 'mean_ms': float(arr.mean()), 'std_ms': float(arr.std()), 'p50_ms': float(np.percentile(arr, 50)), 'p95_ms': float(np.percentile(arr, 95)), 'p99_ms': float(np.percentile(arr, 99))}

def main():
    print(f'[latency] device = CPU (Apple Silicon arm64)')
    print(f'[latency] N_warmup = {N_WARMUP}, N_samples = {N_SAMPLES}')
    print(f'[latency] building dataset (window={WINDOW})...')
    split = build_split(window=WINDOW)
    snapshots = [(s.x, s.edge_index, s.edge_attr, s.y) for s in split.test]
    print(f'[latency] test snapshots available: {len(snapshots)}')
    pers = PersistenceBaseline(window=WINDOW)
    mlp = MLPBaseline(in_features=split.num_features, hidden=HIDDEN, dropout=0.2)
    gru = GRUBaseline(window=WINDOW, hidden=HIDDEN, dropout=0.15)
    stgnn = STGNN(window=WINDOW, gcn_hidden=HIDDEN, gru_hidden=HIDDEN, dropout=0.1)
    for name, fn in [('mlp', 'mlp.pt'), ('gru', 'gru.pt'), ('stgnn', 'stgnn.pt')]:
        ck = MODELS_DIR / fn
        if ck.is_file():
            try:
                if name == 'mlp':
                    mlp.load_state_dict(torch.load(ck, map_location='cpu'))
                elif name == 'gru':
                    gru.load_state_dict(torch.load(ck, map_location='cpu'))
                elif name == 'stgnn':
                    stgnn.load_state_dict(torch.load(ck, map_location='cpu'))
                print(f'  loaded checkpoint {ck.name}')
            except Exception as e:
                print(f'  warn: failed to load {fn}: {e}')
    models = [('Persistence', pers, 0), ('MLP', mlp, sum((p.numel() for p in mlp.parameters()))), ('GRU', gru, sum((p.numel() for p in gru.parameters()))), ('STGNN', stgnn, sum((p.numel() for p in stgnn.parameters())))]
    rows = []
    for name, model, params in models:
        print(f'\n[latency] benchmarking {name} ({params:,} params)...')
        stats = time_model(model, snapshots)
        rows.append({'model': name, 'params': params, **stats})
        print(f"  mean = {stats['mean_ms']:.3f} ms ± {stats['std_ms']:.3f}, p50 = {stats['p50_ms']:.3f}, p95 = {stats['p95_ms']:.3f}, p99 = {stats['p99_ms']:.3f}")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / 'latency.csv'
    fields = ['model', 'params', 'samples', 'mean_ms', 'std_ms', 'p50_ms', 'p95_ms', 'p99_ms']
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n[latency] saved {out_csv}')
    print('\n=== Summary table ===')
    print(f"{'Model':<14}{'Params':<10}{'Mean ms':<11}{'p50':<9}{'p95':<9}{'p99':<9}")
    for r in rows:
        print(f"{r['model']:<14}{r['params']:<10}{r['mean_ms']:<11.3f}{r['p50_ms']:<9.3f}{r['p95_ms']:<9.3f}{r['p99_ms']:<9.3f}")
if __name__ == '__main__':
    main()
