from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.baselines import GRUBaseline, MLPBaseline, PersistenceBaseline
from model.dataset import build_split, signal_size
from model.stgnn import STGNN
from model.train import _signal_to_tensors, train_model, eval_persistence, eval_trained, TrainConfig
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
MESHES = {'4x4': dict(mesh_id='mesh_4x4', val_runs=('mesh_4x4_uniform_ir050', 'mesh_4x4_neighbor_ir050'), test_runs=('mesh_4x4_uniform_ir060', 'mesh_4x4_transpose_ir060', 'mesh_4x4_hotspot_ir060'), exclude_runs=('mesh_4x4_uniform_smoke',)), '8x8': dict(mesh_id='mesh_8x8', val_runs=('mesh_8x8_uniform_ir020', 'mesh_8x8_neighbor_ir020'), test_runs=('mesh_8x8_uniform_ir040', 'mesh_8x8_transpose_ir040', 'mesh_8x8_hotspot_ir040'), exclude_runs=())}

def run_mesh(mesh_key: str, seeds: list[int], window: int, hidden: int, epochs: int, lr: float, wd: float, device) -> list[dict]:
    spec = MESHES[mesh_key]
    print(f'\n========== MESH {mesh_key} (W={window} hidden={hidden}) ==========')
    split = build_split(window=window, mesh_id=spec['mesh_id'], val_runs=spec['val_runs'], test_runs=spec['test_runs'], exclude_runs=spec['exclude_runs'])
    print(f'[{mesh_key}] nodes={split.num_nodes} feat={split.num_features} train={signal_size(split.train)} val={signal_size(split.val)} test={signal_size(split.test)}')
    train_data = _signal_to_tensors(split.train, device)
    val_data = _signal_to_tensors(split.val, device)
    test_data = _signal_to_tensors(split.test, device)
    rows: list[dict] = []
    pers = PersistenceBaseline(window=window).to(device)
    p_rmse, p_mae, _, _ = eval_persistence(pers, test_data)
    rows.append({'mesh': mesh_key, 'seed': 'n/a', 'model': 'Persistence', 'test_rmse': p_rmse, 'test_mae': p_mae})
    print(f'[{mesh_key}] Persistence test RMSE={p_rmse:.5f} MAE={p_mae:.5f}')
    for seed in seeds:
        cfg = TrainConfig(window=window, epochs=epochs, lr=lr, weight_decay=wd, hidden=hidden, seed=seed)
        for name, build in (('MLP', lambda: MLPBaseline(in_features=split.num_features, hidden=hidden, dropout=0.2)), ('GRU', lambda: GRUBaseline(window=window, hidden=hidden, dropout=0.15)), ('STGNN', lambda: STGNN(window=window, gcn_hidden=hidden, gru_hidden=hidden, dropout=0.1))):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build()
            train_model(f'{name}-{mesh_key}-s{seed}', model, train_data, val_data, cfg=cfg, device=device)
            rmse, mae, _, _ = eval_trained(model, test_data, target_scale=cfg.target_scale)
            rows.append({'mesh': mesh_key, 'seed': seed, 'model': name, 'test_rmse': rmse, 'test_mae': mae})
            print(f'[{mesh_key}] seed={seed} {name}: test RMSE={rmse:.5f} MAE={mae:.5f}')
    out = METRICS_DIR / f'multiseed_{mesh_key}.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['mesh', 'seed', 'model', 'test_rmse', 'test_mae'])
        w.writeheader()
        w.writerows(rows)
    print(f'[{mesh_key}] saved {out}')
    return rows

def summarize(rows: list[dict], mesh_key: str):
    print(f'\n----- SUMMARY {mesh_key}: test RMSE mean ± std over seeds -----')
    for model in ('Persistence', 'MLP', 'GRU', 'STGNN'):
        vals = [r['test_rmse'] for r in rows if r['model'] == model and r['seed'] != 'n/a']
        maes = [r['test_mae'] for r in rows if r['model'] == model and r['seed'] != 'n/a']
        if model == 'Persistence':
            v = [r['test_rmse'] for r in rows if r['model'] == 'Persistence']
            m = [r['test_mae'] for r in rows if r['model'] == 'Persistence']
            print(f'  {model:<12} RMSE {v[0]:.5f} (det.)        MAE {m[0]:.5f}')
            continue
        print(f'  {model:<12} RMSE {np.mean(vals):.5f} ± {np.std(vals):.5f}   MAE {np.mean(maes):.5f} ± {np.std(maes):.5f}   (n={len(vals)})')

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 42, 123])
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--meshes', nargs='+', default=['4x4', '8x8'])
    args = p.parse_args(argv)
    device = torch.device('cpu')
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[multiseed] seeds={args.seeds} meshes={args.meshes}')
    all_summary = {}
    for mk in args.meshes:
        rows = run_mesh(mk, args.seeds, args.window, args.hidden, args.epochs, args.lr, args.weight_decay, device)
        all_summary[mk] = rows
    for mk in args.meshes:
        summarize(all_summary[mk], mk)
if __name__ == '__main__':
    main()
