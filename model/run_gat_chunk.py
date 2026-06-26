from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from model.run_gat import STGNNGat, MESHES, build_items_from_csv
from model.train import TrainConfig, eval_trained, rmse_mae
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
CKPT_DIR = PROJECT_ROOT / 'results' / 'gat_ckpt'
PROGRESS = METRICS_DIR / 'gat_progress.json'
OUT_CSV = METRICS_DIR / 'gat.csv'
SEEDS = [0, 1, 2, 42, 123]
JOBS = [('4x4', s) for s in SEEDS] + [('8x8', s) for s in SEEDS]
EPOCHS = 15

def _epoch_pass(model, data, *, optimizer, loss_fn, target_scale, grad_clip):
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
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
    rmse, mae = rmse_mae(torch.cat(all_preds), torch.cat(all_targs))
    return (float(np.mean(losses)), rmse, mae)

def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {'done': [], 'results': []}

def save_progress(pg):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(pg, indent=1))
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['mesh', 'seed', 'test_rmse', 'test_mae'])
        w.writeheader()
        w.writerows(pg['results'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=30.0)
    args = ap.parse_args()
    t_start = time.time()
    device = torch.device('cpu')
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pg = load_progress()
    done = {tuple(d) for d in pg['done']}
    job = next(((m, s) for m, s in JOBS if (m, s) not in done), None)
    if job is None:
        print('ALL DONE')
        for mk in ('4x4', '8x8'):
            r = [x['test_rmse'] for x in pg['results'] if x['mesh'] == mk]
            a = [x['test_mae'] for x in pg['results'] if x['mesh'] == mk]
            if r:
                print(f'GAT {mk}: RMSE {np.mean(r):.5f} ± {np.std(r):.5f}  MAE {np.mean(a):.5f} ± {np.std(a):.5f}  (n={len(r)})')
        return
    mesh, seed = job
    cfg = TrainConfig(window=5, epochs=EPOCHS, lr=0.005, weight_decay=0.0005, hidden=32, seed=seed)
    splits, num_nodes = build_items_from_csv(MESHES[mesh], cfg.window, device)
    tr, va, te = (splits['train'], splits['val'], splits['test'])
    ck_path = CKPT_DIR / f'gat_{mesh}_{seed}.pt'
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    if ck_path.exists():
        ck = torch.load(ck_path, weights_only=False)
        model = STGNNGat(window=cfg.window, hidden=cfg.hidden, heads=4, dropout=0.1)
        model.load_state_dict(ck['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        optimizer.load_state_dict(ck['optim'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        scheduler.load_state_dict(ck['sched'])
        torch.set_rng_state(ck['rng'])
        ep0 = ck['epoch']
        best_val = ck['best_val']
        best_state = ck['best_state']
        print(f'[resume] GAT-{mesh}-s{seed} from epoch {ep0}')
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = STGNNGat(window=cfg.window, hidden=cfg.hidden, heads=4, dropout=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        ep0 = 0
        best_val = float('inf')
        best_state = None
        print(f'[start ] GAT-{mesh}-s{seed} nodes={num_nodes} train={len(tr)} val={len(va)} test={len(te)} params={sum((p.numel() for p in model.parameters())):,}')
    ep = ep0
    while ep < EPOCHS:
        ep += 1
        tr_loss, tr_rmse, _ = _epoch_pass(model, tr, optimizer=optimizer, loss_fn=loss_fn, target_scale=cfg.target_scale, grad_clip=cfg.grad_clip)
        scheduler.step()
        with torch.no_grad():
            _, val_rmse, val_mae = _epoch_pass(model, va, optimizer=None, loss_fn=loss_fn, target_scale=cfg.target_scale, grad_clip=None)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'  ep {ep:3d} | train_loss {tr_loss:.5f} | train_rmse {tr_rmse:.5f} | val_rmse {val_rmse:.5f} | val_mae {val_mae:.5f}', flush=True)
        if ep < EPOCHS and time.time() - t_start > args.budget:
            break
    if ep >= EPOCHS:
        if best_state is not None:
            model.load_state_dict(best_state)
        rmse, mae, _, _ = eval_trained(model, te, target_scale=cfg.target_scale)
        pg['results'].append({'mesh': mesh, 'seed': seed, 'test_rmse': rmse, 'test_mae': mae})
        pg['done'].append([mesh, seed])
        save_progress(pg)
        ck_path.unlink(missing_ok=True)
        print(f"[DONE ] GAT-{mesh}-s{seed}: test RMSE={rmse:.5f} MAE={mae:.5f} ({len(pg['done'])}/{len(JOBS)} jobs)")
    else:
        torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'sched': scheduler.state_dict(), 'rng': torch.get_rng_state(), 'epoch': ep, 'best_val': best_val, 'best_state': best_state}, ck_path)
        print(f'[ckpt ] GAT-{mesh}-s{seed} saved at epoch {ep}')
if __name__ == '__main__':
    main()
