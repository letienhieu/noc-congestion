from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.baselines import GRUBaseline, MLPBaseline, PersistenceBaseline
from model.dataset import GraphSplit, build_split, signal_size
from model.stgnn import STGNN
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'

@dataclass
class TrainConfig:
    window: int
    epochs: int
    lr: float
    weight_decay: float
    hidden: int
    seed: int
    target_scale: float = 100.0
    grad_clip: float = 1.0

def _signal_to_tensors(signal, device: torch.device):
    items = []
    for snapshot in signal:
        items.append((snapshot.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None, snapshot.y.to(device)))
    return items

def rmse_mae(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    diff = preds - targets
    rmse = torch.sqrt((diff * diff).mean()).item()
    mae = diff.abs().mean().item()
    return (rmse, mae)

def _epoch(model: nn.Module, data: list, *, optimizer: torch.optim.Optimizer | None, loss_fn: nn.Module, target_scale: float=1.0, grad_clip: float | None=None) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    all_preds, all_targs = ([], [])
    for x, edge_index, edge_weight, y in data:
        if is_train:
            optimizer.zero_grad()
        preds_scaled = model(x, edge_index, edge_weight)
        y_scaled = y * target_scale
        loss = loss_fn(preds_scaled, y_scaled)
        if is_train:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(loss.item())
        all_preds.append((preds_scaled / target_scale).detach())
        all_targs.append(y.detach())
    preds_all = torch.cat(all_preds)
    targs_all = torch.cat(all_targs)
    rmse, mae = rmse_mae(preds_all, targs_all)
    return (float(np.mean(losses)), rmse, mae)

def train_model(name: str, model: nn.Module, train_data: list, val_data: list, *, cfg: TrainConfig, device: torch.device) -> dict[str, list[float]]:
    print(f'\n[train] === {name} === params = {sum((p.numel() for p in model.parameters())):,}')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    history = {'train_loss': [], 'val_rmse': [], 'val_mae': []}
    best_val = float('inf')
    best_state = None
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_rmse, tr_mae = _epoch(model, train_data, optimizer=optimizer, loss_fn=loss_fn, target_scale=cfg.target_scale, grad_clip=cfg.grad_clip)
        scheduler.step()
        with torch.no_grad():
            _, val_rmse, val_mae = _epoch(model, val_data, optimizer=None, loss_fn=loss_fn, target_scale=cfg.target_scale)
        history['train_loss'].append(tr_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == cfg.epochs:
            print(f'  ep {ep:3d} | train_loss {tr_loss:.5f} | train_rmse {tr_rmse:.5f} | val_rmse {val_rmse:.5f} | val_mae {val_mae:.5f}')
    print(f'[train] {name} done in {time.time() - t0:.1f}s, best_val_rmse = {best_val:.5f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return history

def eval_persistence(pers: PersistenceBaseline, data: list) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    preds_all, targs_all = ([], [])
    pers.eval()
    with torch.no_grad():
        for x, edge_index, edge_weight, y in data:
            preds = pers(x, edge_index, edge_weight)
            preds_all.append(preds)
            targs_all.append(y)
    preds_t = torch.cat(preds_all)
    targs_t = torch.cat(targs_all)
    rmse, mae = rmse_mae(preds_t, targs_t)
    return (rmse, mae, preds_t, targs_t)

def eval_trained(model: nn.Module, data: list, *, target_scale: float=1.0) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    preds_all, targs_all = ([], [])
    with torch.no_grad():
        for x, edge_index, edge_weight, y in data:
            preds = model(x, edge_index, edge_weight) / target_scale
            preds_all.append(preds)
            targs_all.append(y)
    preds_t = torch.cat(preds_all)
    targs_t = torch.cat(targs_all)
    rmse, mae = rmse_mae(preds_t, targs_t)
    return (rmse, mae, preds_t, targs_t)

def main(argv: list[str] | None=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    cfg = TrainConfig(window=args.window, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, hidden=args.hidden, seed=args.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cpu')
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[train] device = {device}')
    print(f'[train] building dataset (window={cfg.window})...')
    split: GraphSplit = build_split(window=cfg.window)
    print(f'[train] num_nodes={split.num_nodes} num_features={split.num_features} train={signal_size(split.train)} val={signal_size(split.val)} test={signal_size(split.test)}')
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
    histories = {'MLP': train_model('MLP', mlp, train_data, val_data, cfg=cfg, device=device), 'GRU': train_model('GRU', gru, train_data, val_data, cfg=cfg, device=device), 'STGNN': train_model('STGNN', stgnn, train_data, val_data, cfg=cfg, device=device)}
    print('\n[train] === FINAL TEST EVAL ===')
    rows = []
    pers_results = {}
    for split_name, data in (('train', train_data), ('val', val_data), ('test', test_data)):
        rmse, mae, _, _ = eval_persistence(pers, data)
        rows.append({'model': 'Persistence', 'split': split_name, 'rmse': rmse, 'mae': mae})
        pers_results[split_name] = (rmse, mae)
    mlp_results, gru_results, stgnn_results = ({}, {}, {})
    for split_name, data in (('train', train_data), ('val', val_data), ('test', test_data)):
        rmse_m, mae_m, _, _ = eval_trained(mlp, data, target_scale=cfg.target_scale)
        rmse_g, mae_g, _, _ = eval_trained(gru, data, target_scale=cfg.target_scale)
        rmse_s, mae_s, _, _ = eval_trained(stgnn, data, target_scale=cfg.target_scale)
        rows.append({'model': 'MLP', 'split': split_name, 'rmse': rmse_m, 'mae': mae_m})
        rows.append({'model': 'GRU', 'split': split_name, 'rmse': rmse_g, 'mae': mae_g})
        rows.append({'model': 'STGNN', 'split': split_name, 'rmse': rmse_s, 'mae': mae_s})
        mlp_results[split_name] = (rmse_m, mae_m)
        gru_results[split_name] = (rmse_g, mae_g)
        stgnn_results[split_name] = (rmse_s, mae_s)
    print(f"\n{'model':<14}{'split':<10}{'rmse':<12}{'mae':<12}")
    for row in rows:
        print(f"{row['model']:<14}{row['split']:<10}{row['rmse']:<12.5f}{row['mae']:<12.5f}")
    out_csv = METRICS_DIR / 'comparison.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'split', 'rmse', 'mae'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n[train] saved {out_csv}')
    (METRICS_DIR / 'history.json').write_text(json.dumps(histories, indent=2))
    (METRICS_DIR / 'config.json').write_text(json.dumps({'window': cfg.window, 'epochs': cfg.epochs, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay, 'hidden': cfg.hidden, 'seed': cfg.seed}, indent=2))
    torch.save(mlp.state_dict(), MODELS_DIR / 'mlp.pt')
    torch.save(gru.state_dict(), MODELS_DIR / 'gru.pt')
    torch.save(stgnn.state_dict(), MODELS_DIR / 'stgnn.pt')
    print(f'[train] saved checkpoints in {MODELS_DIR}')
    pers_test_rmse = pers_results['test'][0]
    mlp_test_rmse = mlp_results['test'][0]
    gru_test_rmse = gru_results['test'][0]
    stgnn_test_rmse = stgnn_results['test'][0]
    print(f'\n[train] test RMSE: persistence={pers_test_rmse:.5f}, MLP={mlp_test_rmse:.5f}, GRU={gru_test_rmse:.5f}, STGNN={stgnn_test_rmse:.5f}')
    if stgnn_test_rmse < pers_test_rmse and stgnn_test_rmse < mlp_test_rmse and (stgnn_test_rmse < gru_test_rmse):
        print('[train] ST-GNN thắng tất cả baseline trên test RMSE.')
    else:
        print('[train] ST-GNN CHƯA thắng baseline - cần điều chỉnh hyperparams.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
