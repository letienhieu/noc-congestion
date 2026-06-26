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
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    cfg = TrainConfig(window=args.window, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, hidden=args.hidden, seed=args.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cpu')
    print(f'[8x8-tuned] window={cfg.window} hidden={cfg.hidden} epochs={cfg.epochs} lr={cfg.lr} wd={cfg.weight_decay} seed={cfg.seed}')
    print('[8x8-tuned] building 8x8 dataset...')
    val_runs = ('mesh_8x8_uniform_ir020', 'mesh_8x8_neighbor_ir020')
    test_runs = ('mesh_8x8_uniform_ir040', 'mesh_8x8_transpose_ir040', 'mesh_8x8_hotspot_ir040')
    split = build_split(window=cfg.window, mesh_id='mesh_8x8', val_runs=val_runs, test_runs=test_runs, exclude_runs=())
    print(f'[8x8-tuned] num_nodes={split.num_nodes} num_features={split.num_features} train={signal_size(split.train)} val={signal_size(split.val)} test={signal_size(split.test)}')
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
    n_params = {'MLP': sum((x.numel() for x in mlp.parameters())), 'GRU': sum((x.numel() for x in gru.parameters())), 'STGNN': sum((x.numel() for x in stgnn.parameters()))}
    train_model('MLP-8x8t', mlp, train_data, val_data, cfg=cfg, device=device)
    train_model('GRU-8x8t', gru, train_data, val_data, cfg=cfg, device=device)
    train_model('STGNN-8x8t', stgnn, train_data, val_data, cfg=cfg, device=device)
    print('\n[8x8-tuned] === FINAL TEST EVAL (mesh 8x8, tuned) ===')
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
    print(f'\n[8x8-tuned] params: {n_params}')
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / 'comparison_8x8_tuned.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'split', 'rmse', 'mae'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'[8x8-tuned] saved {out_csv}')
    pers_t = rows[-4]['rmse']
    mlp_t = rows[-3]['rmse']
    gru_t = rows[-2]['rmse']
    stgnn_t = rows[-1]['rmse']
    print(f'\n[8x8-tuned] TEST RMSE: Pers={pers_t:.5f}, MLP={mlp_t:.5f}, GRU={gru_t:.5f}, STGNN={stgnn_t:.5f}')
    if stgnn_t < min(pers_t, mlp_t, gru_t):
        print(f'[8x8-tuned] ST-GNN THẮNG TẤT CẢ ở 8x8 tuned! (Δ vs GRU = {gru_t - stgnn_t:+.5f}, {100 * (gru_t - stgnn_t) / gru_t:+.1f}%)')
    else:
        print(f'[8x8-tuned] ST-GNN CHƯA thắng. Δ vs GRU = {stgnn_t - gru_t:+.5f}')
if __name__ == '__main__':
    main()
