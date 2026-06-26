from __future__ import annotations
import csv
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.baselines import GRUBaseline, MLPBaseline, PersistenceBaseline
from model.dataset import build_split, signal_size
from model.stgnn import STGNN
from model.train import _signal_to_tensors, train_model, eval_persistence, eval_trained, TrainConfig
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'

def main():
    cfg = TrainConfig(window=5, epochs=15, lr=0.005, weight_decay=0.0005, hidden=32, seed=42)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cpu')
    print('[train-8x8] device =', device)
    print('[train-8x8] building 8x8 dataset...')
    val_runs = ('mesh_8x8_uniform_ir020', 'mesh_8x8_neighbor_ir020')
    test_runs = ('mesh_8x8_uniform_ir040', 'mesh_8x8_transpose_ir040', 'mesh_8x8_hotspot_ir040')
    split = build_split(window=cfg.window, mesh_id='mesh_8x8', val_runs=val_runs, test_runs=test_runs, exclude_runs=())
    print(f'[train-8x8] num_nodes={split.num_nodes} num_features={split.num_features} train={signal_size(split.train)} val={signal_size(split.val)} test={signal_size(split.test)}')
    train_data = _signal_to_tensors(split.train, device)
    val_data = _signal_to_tensors(split.val, device)
    test_data = _signal_to_tensors(split.test, device)
    pers = PersistenceBaseline(window=cfg.window).to(device)
    torch.manual_seed(cfg.seed)
    mlp = MLPBaseline(in_features=split.num_features, hidden=cfg.hidden, dropout=0.2)
    torch.manual_seed(cfg.seed)
    gru = GRUBaseline(window=cfg.window, hidden=cfg.hidden, dropout=0.15)
    torch.manual_seed(cfg.seed)
    stgnn = STGNN(window=cfg.window, gcn_hidden=cfg.hidden, gru_hidden=cfg.hidden, dropout=0.1)
    train_model('MLP-8x8', mlp, train_data, val_data, cfg=cfg, device=device)
    train_model('GRU-8x8', gru, train_data, val_data, cfg=cfg, device=device)
    train_model('STGNN-8x8', stgnn, train_data, val_data, cfg=cfg, device=device)
    print('\n[train-8x8] === FINAL TEST EVAL (mesh 8x8) ===')
    rows = []
    for split_name, data in (('train', train_data), ('val', val_data), ('test', test_data)):
        rmse, mae, _, _ = eval_persistence(pers, data)
        rows.append({'model': 'Persistence', 'split': split_name, 'rmse': rmse, 'mae': mae})
        rmse_m, mae_m, _, _ = eval_trained(mlp, data, target_scale=cfg.target_scale)
        rmse_g, mae_g, _, _ = eval_trained(gru, data, target_scale=cfg.target_scale)
        rmse_s, mae_s, _, _ = eval_trained(stgnn, data, target_scale=cfg.target_scale)
        rows.append({'model': 'MLP', 'split': split_name, 'rmse': rmse_m, 'mae': mae_m})
        rows.append({'model': 'GRU', 'split': split_name, 'rmse': rmse_g, 'mae': mae_g})
        rows.append({'model': 'STGNN', 'split': split_name, 'rmse': rmse_s, 'mae': mae_s})
    print(f"\n{'model':<14}{'split':<10}{'rmse':<12}{'mae':<12}")
    for row in rows:
        print(f"{row['model']:<14}{row['split']:<10}{row['rmse']:<12.5f}{row['mae']:<12.5f}")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / 'comparison_8x8.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'split', 'rmse', 'mae'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n[train-8x8] saved {out_csv}')
    torch.save(mlp.state_dict(), MODELS_DIR / 'mlp_8x8.pt')
    torch.save(gru.state_dict(), MODELS_DIR / 'gru_8x8.pt')
    torch.save(stgnn.state_dict(), MODELS_DIR / 'stgnn_8x8.pt')
    pers_t = rows[2]['rmse']
    mlp_t = rows[7]['rmse']
    gru_t = rows[8]['rmse']
    stgnn_t = rows[9]['rmse']
    print(f'\n[train-8x8] test RMSE: Pers={pers_t:.5f}, MLP={mlp_t:.5f}, GRU={gru_t:.5f}, STGNN={stgnn_t:.5f}')
    if stgnn_t < gru_t:
        print(f'[train-8x8] STGNN BEAT GRU on 8x8! (Δ = {gru_t - stgnn_t:.5f})')
    else:
        print(f'[train-8x8] STGNN vẫn thua GRU ({stgnn_t - gru_t:+.5f})')
if __name__ == '__main__':
    main()
