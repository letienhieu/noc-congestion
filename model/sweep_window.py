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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
TARGET_SCALE = 100.0
EPOCHS = 15
LR = 0.005
WD = 0.0005
HIDDEN = 32
SEED = 42

def _signal_to_list(signal):
    return [(s.x, s.edge_index, s.edge_attr, s.y) for s in signal]

def _epoch_run(model, data, opt, loss_fn):
    model.train(opt is not None)
    losses, ps, ts = ([], [], [])
    for x, ei, ew, y in data:
        if opt:
            opt.zero_grad()
        p = model(x, ei, ew)
        l = loss_fn(p, y * TARGET_SCALE)
        if opt:
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        losses.append(l.item())
        ps.append((p / TARGET_SCALE).detach())
        ts.append(y.detach())
    pt = torch.cat(ps)
    tt = torch.cat(ts)
    rmse = torch.sqrt(((pt - tt) ** 2).mean()).item()
    mae = (pt - tt).abs().mean().item()
    return (float(np.mean(losses)), rmse, mae)

def train_one_window(window: int) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f'\n[sweep W={window}] building dataset...')
    split = build_split(window=window)
    tr = _signal_to_list(split.train)
    va = _signal_to_list(split.val)
    te = _signal_to_list(split.test)
    print(f'[sweep W={window}] train={len(tr)} val={len(va)} test={len(te)}')
    torch.manual_seed(SEED)
    model = STGNN(window=window, gcn_hidden=HIDDEN, gru_hidden=HIDDEN, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    best_val = float('inf')
    best_state = None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        _, _, _ = _epoch_run(model, tr, opt, loss_fn)
        sch.step()
        with torch.no_grad():
            _, vr, _ = _epoch_run(model, va, None, loss_fn)
        if vr < best_val:
            best_val = vr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f'[sweep W={window}] done {time.time() - t0:.0f}s best_val_rmse={best_val:.5f}')
    model.load_state_dict(best_state)
    model.eval()
    out = {}
    with torch.no_grad():
        for name, data in [('train', tr), ('val', va), ('test', te)]:
            _, rmse, mae = _epoch_run(model, data, None, loss_fn)
            out[name] = (rmse, mae)
    return out

def main():
    windows = [3, 5, 7, 10]
    rows = []
    for W in windows:
        results = train_one_window(W)
        for split_name, (rmse, mae) in results.items():
            rows.append({'window': W, 'model': 'STGNN', 'split': split_name, 'rmse': rmse, 'mae': mae})
            print(f'  W={W} {split_name:5s}: RMSE={rmse:.5f}  MAE={mae:.5f}')
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / 'window_sweep.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['window', 'model', 'split', 'rmse', 'mae'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n[sweep] saved {out_csv}')
    print('\n=== Window sweep - test set ===')
    print(f"{'W':<6}{'RMSE':<12}{'MAE':<12}")
    for r in rows:
        if r['split'] == 'test':
            print(f"{r['window']:<6}{r['rmse']:<12.5f}{r['mae']:<12.5f}")
if __name__ == '__main__':
    main()
