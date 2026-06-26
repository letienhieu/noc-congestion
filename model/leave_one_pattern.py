from __future__ import annotations
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
PATTERNS = ['uniform', 'transpose', 'neighbor', 'hotspot']
RATES = ['ir010', 'ir020', 'ir030', 'ir040', 'ir050', 'ir060', 'ir070']

def runs_of(pattern):
    return [f'mesh_4x4_{pattern}_{r}' for r in RATES]

def main():
    device = torch.device('cpu')
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(window=5, epochs=15, lr=0.005, weight_decay=0.0005, hidden=32, seed=42)
    rows = []
    for held in PATTERNS:
        others = [p for p in PATTERNS if p != held]
        test_runs = tuple(runs_of(held))
        val_runs = (f'mesh_4x4_{others[0]}_ir030', f'mesh_4x4_{others[1]}_ir030')
        print(f'\n===== HELD-OUT pattern = {held} (test={len(test_runs)} runs) =====')
        split = build_split(window=cfg.window, mesh_id='mesh_4x4', val_runs=val_runs, test_runs=test_runs, exclude_runs=('mesh_4x4_uniform_smoke',))
        print(f'  train={signal_size(split.train)} val={signal_size(split.val)} test={signal_size(split.test)}')
        train_data = _signal_to_tensors(split.train, device)
        val_data = _signal_to_tensors(split.val, device)
        test_data = _signal_to_tensors(split.test, device)
        pers = PersistenceBaseline(window=cfg.window).to(device)
        p_rmse, p_mae, _, _ = eval_persistence(pers, test_data)
        rows.append({'held_out': held, 'model': 'Persistence', 'test_rmse': p_rmse, 'test_mae': p_mae})
        for name, build in (('MLP', lambda: MLPBaseline(in_features=split.num_features, hidden=cfg.hidden, dropout=0.2)), ('GRU', lambda: GRUBaseline(window=cfg.window, hidden=cfg.hidden, dropout=0.15)), ('STGNN', lambda: STGNN(window=cfg.window, gcn_hidden=cfg.hidden, gru_hidden=cfg.hidden, dropout=0.1))):
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            model = build()
            train_model(f'{name}-LOPO-{held}', model, train_data, val_data, cfg=cfg, device=device)
            rmse, mae, _, _ = eval_trained(model, test_data, target_scale=cfg.target_scale)
            rows.append({'held_out': held, 'model': name, 'test_rmse': rmse, 'test_mae': mae})
            print(f'  {held}/{name}: test RMSE={rmse:.5f} MAE={mae:.5f}')
    out = METRICS_DIR / 'lopo_4x4.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['held_out', 'model', 'test_rmse', 'test_mae'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nsaved {out}')
    print(f"\n{'held_out':<11}{'Pers':<10}{'MLP':<10}{'GRU':<10}{'STGNN':<10}")
    for held in PATTERNS:
        d = {r['model']: r['test_rmse'] for r in rows if r['held_out'] == held}
        print(f"{held:<11}{d['Persistence']:<10.5f}{d['MLP']:<10.5f}{d['GRU']:<10.5f}{d['STGNN']:<10.5f}")
if __name__ == '__main__':
    main()
