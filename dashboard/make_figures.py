from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard.app import fetch_run_dataframe, list_runs, load_models, predict_run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'
WINDOW = 5
NUM_FEATURES = 5
HIDDEN = 32
TARGET_SCALE = 100.0
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 140, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def fig_training_loss():
    history_path = METRICS_DIR / 'history.json'
    if not history_path.is_file():
        print('[fig] history.json không có - bỏ qua training_loss.')
        return
    hist = json.loads(history_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in hist.items():
        epochs = np.arange(1, len(h['train_loss']) + 1)
        axes[0].plot(epochs, h['train_loss'], label=name)
        axes[1].plot(epochs, h['val_rmse'], label=name)
    axes[0].set_title('Training loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss (Huber)')
    axes[0].legend()
    axes[1].set_title('Validation RMSE')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()
    fig.tight_layout()
    out = FIGURES_DIR / 'training_loss.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] saved {out}')

def fig_comparison_bar():
    import csv as _csv, statistics as _st
    f4, f8 = (METRICS_DIR / 'multiseed_4x4.csv', METRICS_DIR / 'multiseed_8x8.csv')
    if not (f4.is_file() and f8.is_file()):
        print('[fig] multiseed_4x4/8x8.csv missing - skip comparison_bar.')
        return

    def _stats(path, mesh):
        rows = list(_csv.DictReader(open(path)))
        out = {}
        for mdl in ('Persistence', 'MLP', 'STGNN', 'GRU'):
            allv = [float(r['test_rmse']) for r in rows if r['model'] == mdl and r['mesh'] == mesh]
            seeded = [float(r['test_rmse']) for r in rows if r['model'] == mdl and r['mesh'] == mesh and (r['seed'] != 'n/a')]
            out[mdl] = (sum(allv) / len(allv), _st.pstdev(seeded) if len(seeded) > 1 else 0.0)
        return out
    s4, s8 = (_stats(f4, '4x4'), _stats(f8, '8x8'))
    models = [('Persistence', '#888888'), ('MLP', '#1f77b4'), ('ST-GNN', '#d62728'), ('GRU (per-node)', '#2ca02c')]
    keymap = {'Persistence': 'Persistence', 'MLP': 'MLP', 'ST-GNN': 'STGNN', 'GRU (per-node)': 'GRU'}
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(2)
    w = 0.2
    for i, (label, color) in enumerate(models):
        k = keymap[label]
        ax.bar(x + (i - 1.5) * w, [s4[k][0], s8[k][0]], w, yerr=[s4[k][1], s8[k][1]], capsize=3, label=label, color=color, error_kw=dict(ecolor='black', lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(['Mesh 4×4', 'Mesh 8×8'])
    ax.set_ylabel('Test RMSE')
    ax.set_title('Test RMSE (mean ± std, 5 seeds)')
    ax.set_ylim(0, 0.016)
    ax.legend(ncol=2, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / 'comparison_bar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] saved {out}')

def fig_heatmaps(run_id: str, sample_idx: int, models: dict):
    df = fetch_run_dataframe(run_id)
    sub = df[df['t'] == sample_idx]
    z_actual = np.full((4, 4), np.nan)
    for _, row in sub.iterrows():
        z_actual[int(row['y']), int(row['x'])] = row['occ']
    vmax = max(0.05, float(df['occ'].max()))

    def _draw(z, title, out_name):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(z, cmap='viridis', vmin=0, vmax=vmax, origin='lower')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        for y in range(4):
            for x in range(4):
                v = z[y, x]
                if np.isnan(v):
                    continue
                ax.text(x, y, f'{v:.3f}', ha='center', va='center', color='black' if v > vmax * 0.5 else 'white', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        out = FIGURES_DIR / out_name
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f'[fig] saved {out}')
    _draw(z_actual, f'Actual - {run_id} @ t={sample_idx}', f'heatmap_actual_{run_id}_t{sample_idx:03d}.png')
    if not any((m is not None for n, m in models.items() if n != 'Persistence')):
        return
    try:
        preds = predict_run(models, run_id, window=WINDOW, target_scale=TARGET_SCALE)
    except Exception as exc:
        print(f'[fig] predict_run lỗi cho {run_id}: {exc}')
        return
    for name, arr in preds.items():
        if sample_idx >= arr.shape[1]:
            continue
        col = arr[:, sample_idx]
        if np.all(np.isnan(col)):
            continue
        z = np.full((4, 4), np.nan)
        for rid, v in enumerate(col):
            y, x = (rid // 4, rid % 4)
            z[y, x] = v
        _draw(z, f'{name} pred - {run_id} @ t={sample_idx}', f'heatmap_pred_{name.lower()}_{run_id}_t{sample_idx:03d}.png')

def fig_preds_vs_actual(run_id: str, router_id: int, models: dict):
    df = fetch_run_dataframe(run_id)
    sub = df[df['router_id'] == router_id].sort_values('t')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub['t'], sub['occ'], color='#222', lw=2, label='Actual')
    if any((m is not None for n, m in models.items() if n != 'Persistence')):
        try:
            preds = predict_run(models, run_id, window=WINDOW, target_scale=TARGET_SCALE)
            for name, arr in preds.items():
                col = arr[router_id]
                style = {'Persistence': ('--', '#888'), 'MLP': ('--', '#1f77b4'), 'STGNN': ('--', '#d62728')}[name]
                ax.plot(range(len(col)), col, linestyle=style[0], color=style[1], lw=1.5, label=f'{name} pred')
        except Exception as exc:
            print(f'[fig] preds_vs_actual lỗi: {exc}')
    ax.set_title(f'Buffer occupancy - Router {router_id} - {run_id}')
    ax.set_xlabel('sample idx (t)')
    ax.set_ylabel('occupancy')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f'preds_vs_actual_{run_id}_router{router_id}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] saved {out}')

def fig_per_run_rmse(models: dict):
    runs = [r for r in list_runs() if r != 'mesh_4x4_uniform_smoke']
    if not any((m is not None for n, m in models.items() if n != 'Persistence')):
        print('[fig] không có model trained - bỏ qua per_run_rmse.')
        return
    records = []
    for run_id in runs:
        try:
            preds = predict_run(models, run_id, window=WINDOW, target_scale=TARGET_SCALE)
        except Exception as exc:
            print(f'[fig] per_run_rmse: bỏ qua {run_id} ({exc})')
            continue
        df = fetch_run_dataframe(run_id)
        T = int(df['t'].max()) + 1
        actual = np.full((16, T), np.nan)
        for _, row in df.iterrows():
            actual[int(row['router_id']), int(row['t'])] = row['occ']
        for name, arr in preds.items():
            mask = ~np.isnan(arr) & ~np.isnan(actual[:, :arr.shape[1]])
            diff = (arr - actual[:, :arr.shape[1]])[mask]
            if len(diff) == 0:
                continue
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            records.append({'run': run_id, 'model': name, 'rmse': rmse})
    if not records:
        return
    res = pd.DataFrame(records).pivot(index='run', columns='model', values='rmse')
    fig, ax = plt.subplots(figsize=(12, 5))
    res.plot(kind='bar', ax=ax, color={'Persistence': '#888', 'MLP': '#1f77b4', 'STGNN': '#d62728'})
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE theo từng run - Persistence vs MLP vs ST-GNN')
    ax.tick_params(axis='x', rotation=60, labelsize=9)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    out = FIGURES_DIR / 'per_run_rmse.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] saved {out}')

def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models(WINDOW, in_features=WINDOW * NUM_FEATURES, hidden=HIDDEN)
    fig_training_loss()
    fig_comparison_bar()
    focus_run = 'mesh_4x4_hotspot_ir050'
    focus_sample = 30
    fig_heatmaps(focus_run, focus_sample, models)
    fig_preds_vs_actual(focus_run, router_id=5, models=models)
    fig_heatmaps('mesh_4x4_transpose_ir040', 40, models)
    fig_preds_vs_actual('mesh_4x4_uniform_ir040', router_id=6, models=models)
    fig_per_run_rmse(models)
    print(f'\n[fig] tất cả figure đã lưu trong {FIGURES_DIR}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
