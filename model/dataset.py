from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch
from neo4j import GraphDatabase
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7688')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'neo4jpass')
NUM_FEATURES = 5
DEFAULT_VAL_RUNS = ('mesh_4x4_uniform_ir050', 'mesh_4x4_neighbor_ir050')
DEFAULT_TEST_RUNS = ('mesh_4x4_uniform_ir060', 'mesh_4x4_transpose_ir060', 'mesh_4x4_hotspot_ir060')

@dataclass
class RunData:
    run_id: str
    features: np.ndarray
    target: np.ndarray

@dataclass
class GraphSplit:
    train: DynamicGraphTemporalSignal
    val: DynamicGraphTemporalSignal
    test: DynamicGraphTemporalSignal
    edge_index: np.ndarray
    num_nodes: int
    num_features: int
    feature_names: list[str]
    window: int

def _fetch_edge_index(session, mesh_id: str) -> np.ndarray:
    result = session.run('\n        MATCH (a:Router {mesh_id: $mesh_id})-[:LINK]->(b:Router {mesh_id: $mesh_id})\n        RETURN a.id AS src, b.id AS dst ORDER BY src, dst\n        ', mesh_id=mesh_id)
    pairs = [(r['src'], r['dst']) for r in result]
    if not pairs:
        raise ValueError(f'Không tìm thấy LINK nào cho mesh_id={mesh_id}')
    arr = np.asarray(pairs, dtype=np.int64).T
    return arr

def _fetch_run(session, run_id: str, num_nodes: int) -> RunData:
    result = session.run('\n        MATCH (rs:RouterState {run_id: $run_id})\n        RETURN rs.sample_idx AS t,\n               rs.router_id AS rid,\n               rs.injected AS inj,\n               rs.ejected AS ej,\n               rs.received_total AS rec,\n               rs.sent_total AS snt,\n               rs.stored_total AS stored,\n               rs.buffer_occupancy_norm AS occ\n        ORDER BY t, rid\n        ', run_id=run_id)
    records = list(result)
    if not records:
        raise ValueError(f'Không có RouterState nào cho run_id={run_id}')
    n_samples = max((r['t'] for r in records)) + 1
    features = np.zeros((n_samples, num_nodes, NUM_FEATURES), dtype=np.float32)
    target = np.zeros((n_samples, num_nodes), dtype=np.float32)
    for r in records:
        t = r['t']
        rid = r['rid']
        features[t, rid, 0] = r['inj'] / 100.0
        features[t, rid, 1] = r['ej'] / 100.0
        features[t, rid, 2] = r['rec'] / 100.0
        features[t, rid, 3] = r['snt'] / 100.0
        features[t, rid, 4] = r['stored'] / 100.0
        target[t, rid] = r['occ']
    return RunData(run_id=run_id, features=features, target=target)

def _list_run_ids(session, mesh_id: str | None=None) -> list[str]:
    if mesh_id is None:
        result = session.run('MATCH (s:Snapshot) RETURN DISTINCT s.run_id AS rid ORDER BY rid')
        return [r['rid'] for r in result]
    result = session.run('MATCH (s:Snapshot) WHERE s.run_id STARTS WITH $prefix RETURN DISTINCT s.run_id AS rid ORDER BY rid', prefix=mesh_id + '_')
    return [r['rid'] for r in result]

def _make_windowed_snapshots(runs: list[RunData], edge_index: np.ndarray, window: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    snapshots_feat: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    edges_per_snapshot: list[np.ndarray] = []
    weights_per_snapshot: list[np.ndarray] = []
    edge_w = np.ones(edge_index.shape[1], dtype=np.float32)
    for run in runs:
        T, N, F = run.features.shape
        for t in range(window - 1, T - 1):
            window_block = run.features[t - window + 1:t + 1]
            x = window_block.transpose(1, 0, 2).reshape(N, window * F)
            y = run.target[t + 1]
            snapshots_feat.append(x.astype(np.float32))
            targets.append(y.astype(np.float32))
            edges_per_snapshot.append(edge_index)
            weights_per_snapshot.append(edge_w)
    return (snapshots_feat, edges_per_snapshot, weights_per_snapshot, targets)

def _to_signal(feats: list[np.ndarray], edges: list[np.ndarray], weights: list[np.ndarray], targets: list[np.ndarray]) -> DynamicGraphTemporalSignal:
    return DynamicGraphTemporalSignal(edge_indices=edges, edge_weights=weights, features=feats, targets=targets)
import csv as _csv
from pathlib import Path as _Path
_PROCESSED_DIR = _Path(__file__).resolve().parent.parent / 'data' / 'processed'

def _csv_run_dirs(mesh_id: str) -> list[str]:
    runs = []
    for d in sorted(_PROCESSED_DIR.glob(f'{mesh_id}_*')):
        if (d / 'router_timeseries.csv').is_file():
            runs.append(d.name)
    if 'burst' not in mesh_id:
        runs = [r for r in runs if 'burst' not in r]
    return runs

def _read_run_csv(run_id: str, num_nodes: int | None=None):
    path = _PROCESSED_DIR / run_id / 'router_timeseries.csv'
    with open(path, newline='') as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f'empty CSV for run_id={run_id}')
    n = max((int(r['router_id']) for r in rows)) + 1
    if num_nodes is None:
        num_nodes = n
    T = max((int(r['sample_idx']) for r in rows)) + 1
    feats = np.zeros((T, num_nodes, NUM_FEATURES), dtype=np.float32)
    targ = np.zeros((T, num_nodes), dtype=np.float32)
    xy: dict[int, tuple[int, int]] = {}
    for r in rows:
        t = int(r['sample_idx'])
        rid = int(r['router_id'])
        feats[t, rid, 0] = float(r['injected']) / 100.0
        feats[t, rid, 1] = float(r['ejected']) / 100.0
        feats[t, rid, 2] = float(r['received_total']) / 100.0
        feats[t, rid, 3] = float(r['sent_total']) / 100.0
        feats[t, rid, 4] = float(r['stored_total']) / 100.0
        targ[t, rid] = float(r['buffer_occupancy_norm'])
        xy[rid] = (int(r['x']), int(r['y']))
    return (RunData(run_id=run_id, features=feats, target=targ), xy, num_nodes)

def _edge_index_from_xy(xy: dict[int, tuple[int, int]]) -> np.ndarray:
    id_by_pos = {pos: rid for rid, pos in xy.items()}
    edges = []
    for rid, (x, y) in xy.items():
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = id_by_pos.get((x + dx, y + dy))
            if nb is not None:
                edges.append((rid, nb))
    return np.asarray(sorted(edges), dtype=np.int64).T

def _build_split_csv(window, mesh_id, val_runs, test_runs, exclude_runs) -> GraphSplit:
    val_set, test_set, excl = (set(val_runs), set(test_runs), set(exclude_runs))
    all_runs = [r for r in _csv_run_dirs(mesh_id) if r not in excl]
    if not all_runs:
        raise ValueError(f'No CSV runs for mesh_id={mesh_id} under {_PROCESSED_DIR}')
    _, xy0, num_nodes = _read_run_csv(all_runs[0])
    edge_index = _edge_index_from_xy(xy0)
    train_ids, val_ids, test_ids = ([], [], [])
    for rid in all_runs:
        (val_ids if rid in val_set else test_ids if rid in test_set else train_ids).append(rid)
    for nm, ids in (('train', train_ids), ('val', val_ids), ('test', test_ids)):
        if not ids:
            raise ValueError(f'CSV split: no runs for {nm} (mesh_id={mesh_id})')
    tr = [_read_run_csv(r, num_nodes)[0] for r in train_ids]
    va = [_read_run_csv(r, num_nodes)[0] for r in val_ids]
    te = [_read_run_csv(r, num_nodes)[0] for r in test_ids]
    train_signal = _to_signal(*_make_windowed_snapshots(tr, edge_index, window))
    val_signal = _to_signal(*_make_windowed_snapshots(va, edge_index, window))
    test_signal = _to_signal(*_make_windowed_snapshots(te, edge_index, window))
    feature_names = []
    for w in range(window):
        for name in ['injected', 'ejected', 'received', 'sent', 'stored']:
            feature_names.append(f'{name}_t{w - window + 1}')
    return GraphSplit(train=train_signal, val=val_signal, test=test_signal, edge_index=edge_index, num_nodes=num_nodes, num_features=NUM_FEATURES * window, feature_names=feature_names, window=window)

def build_split(*, window: int=5, mesh_id: str='mesh_4x4', val_runs: Iterable[str]=DEFAULT_VAL_RUNS, test_runs: Iterable[str]=DEFAULT_TEST_RUNS, exclude_runs: Iterable[str]=('mesh_4x4_uniform_smoke',)) -> GraphSplit:
    if os.environ.get('NOC_DATA_BACKEND', '').lower() == 'csv':
        return _build_split_csv(window, mesh_id, val_runs, test_runs, exclude_runs)
    val_set = set(val_runs)
    test_set = set(test_runs)
    excl_set = set(exclude_runs)
    if val_set & test_set:
        raise ValueError('val_runs và test_runs trùng nhau.')
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            edge_index = _fetch_edge_index(session, mesh_id)
            num_nodes = int(edge_index.max()) + 1
            all_runs = _list_run_ids(session, mesh_id=mesh_id)
            train_ids, val_ids, test_ids = ([], [], [])
            for rid in all_runs:
                if rid in excl_set:
                    continue
                if rid in val_set:
                    val_ids.append(rid)
                elif rid in test_set:
                    test_ids.append(rid)
                else:
                    train_ids.append(rid)
            if not train_ids:
                raise ValueError('Không có run nào dành cho train.')
            if not val_ids:
                raise ValueError('Không có run nào dành cho val.')
            if not test_ids:
                raise ValueError('Không có run nào dành cho test.')
            train_runs = [_fetch_run(session, rid, num_nodes) for rid in train_ids]
            val_runs_data = [_fetch_run(session, rid, num_nodes) for rid in val_ids]
            test_runs_data = [_fetch_run(session, rid, num_nodes) for rid in test_ids]
    finally:
        driver.close()
    train_signal = _to_signal(*_make_windowed_snapshots(train_runs, edge_index, window))
    val_signal = _to_signal(*_make_windowed_snapshots(val_runs_data, edge_index, window))
    test_signal = _to_signal(*_make_windowed_snapshots(test_runs_data, edge_index, window))
    feature_names = []
    base_names = ['injected', 'ejected', 'received', 'sent', 'stored']
    for w in range(window):
        for name in base_names:
            feature_names.append(f'{name}_t{w - window + 1}')
    return GraphSplit(train=train_signal, val=val_signal, test=test_signal, edge_index=edge_index, num_nodes=num_nodes, num_features=NUM_FEATURES * window, feature_names=feature_names, window=window)

def signal_size(signal: DynamicGraphTemporalSignal) -> int:
    return signal.snapshot_count
if __name__ == '__main__':
    split = build_split(window=5)
    print(f'[dataset] num_nodes={split.num_nodes} num_features={split.num_features} window={split.window}')
    print(f'[dataset] edges shape = {split.edge_index.shape}')
    print(f'[dataset] train snapshots = {signal_size(split.train)}')
    print(f'[dataset] val snapshots   = {signal_size(split.val)}')
    print(f'[dataset] test snapshots  = {signal_size(split.test)}')
    g = next(iter(split.train))
    print(f'[dataset] sample snapshot: x={g.x.shape}, y={g.y.shape}, edge_index={g.edge_index.shape}')
