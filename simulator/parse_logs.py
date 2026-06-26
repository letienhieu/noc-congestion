from __future__ import annotations
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class MeshSpec:
    k: int
    n: int
    num_vcs: int
    vc_buf_size: int

    @property
    def num_nodes(self) -> int:
        return self.k ** self.n

    @property
    def num_ports(self) -> int:
        return 2 * self.n + 1

    @property
    def buffer_capacity_per_router(self) -> int:
        return self.num_ports * self.num_vcs * self.vc_buf_size

def _read_int_matrix(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open('r', newline='') as f:
        for raw_row in csv.reader(f):
            row = [int(v) for v in raw_row if v.strip() != '']
            rows.append(row)
    return rows

def _validate_dims(name: str, mat: list[list[int]], expected_cols: int) -> None:
    if not mat:
        raise ValueError(f'{name}: file rỗng.')
    bad = [i for i, r in enumerate(mat) if len(r) != expected_cols]
    if bad:
        raise ValueError(f'{name}: số cột không đồng nhất. Mong đợi {expected_cols}, row vi phạm đầu tiên: idx={bad[0]} len={len(mat[bad[0]])}')

def _sum_per_router(row: list[int], num_routers: int, ports_per_router: int) -> list[int]:
    assert len(row) == num_routers * ports_per_router
    out: list[int] = []
    for r in range(num_routers):
        start = r * ports_per_router
        out.append(sum(row[start:start + ports_per_router]))
    return out

def parse_run(*, raw_dir: Path, mesh: MeshSpec) -> list[dict[str, float | int]]:
    n_nodes = mesh.num_nodes
    n_ports = mesh.num_ports
    injected = _read_int_matrix(raw_dir / 'smoke_injected.csv')
    ejected = _read_int_matrix(raw_dir / 'smoke_ejected.csv')
    received = _read_int_matrix(raw_dir / 'smoke_received.csv')
    sent = _read_int_matrix(raw_dir / 'smoke_sent.csv')
    stored = _read_int_matrix(raw_dir / 'smoke_stored.csv')
    _validate_dims('injected', injected, n_nodes)
    _validate_dims('ejected', ejected, n_nodes)
    _validate_dims('received', received, n_nodes * n_ports)
    _validate_dims('sent', sent, n_nodes * n_ports)
    _validate_dims('stored', stored, n_nodes + n_nodes * n_ports)
    T = len(injected)
    for nm, m in [('ejected', ejected), ('received', received), ('sent', sent), ('stored', stored)]:
        if len(m) != T:
            raise ValueError(f'{nm}: số sample {len(m)} != injected {T}')
    cap = mesh.buffer_capacity_per_router
    long_rows: list[dict[str, float | int]] = []
    for t in range(T):
        stored_per_port = stored[t][n_nodes:]
        received_per_router = _sum_per_router(received[t], n_nodes, n_ports)
        sent_per_router = _sum_per_router(sent[t], n_nodes, n_ports)
        stored_per_router = _sum_per_router(stored_per_port, n_nodes, n_ports)
        for r in range(n_nodes):
            x = r % mesh.k
            y = r // mesh.k
            stored_val = stored_per_router[r]
            occ_norm = stored_val / cap if cap > 0 else 0.0
            long_rows.append({'sample_idx': t, 'router_id': r, 'x': x, 'y': y, 'injected': injected[t][r], 'ejected': ejected[t][r], 'received_total': received_per_router[r], 'sent_total': sent_per_router[r], 'stored_total': stored_val, 'buffer_occupancy_norm': round(occ_norm, 6)})
    return long_rows

def write_long_csv(rows: list[dict[str, float | int]], out_path: Path) -> None:
    if not rows:
        raise ValueError('Không có dòng nào để ghi.')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main(argv: list[str] | None=None) -> int:
    p = argparse.ArgumentParser(description='Parse BookSim TRACK_FLOWS CSV -> long format')
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--k', type=int, required=True, help='số router mỗi chiều')
    p.add_argument('--n', type=int, required=True, help='số chiều của mesh')
    p.add_argument('--num-vcs', type=int, required=True)
    p.add_argument('--vc-buf-size', type=int, required=True)
    args = p.parse_args(argv)
    mesh = MeshSpec(k=args.k, n=args.n, num_vcs=args.num_vcs, vc_buf_size=args.vc_buf_size)
    try:
        rows = parse_run(raw_dir=args.raw_dir.resolve(), mesh=mesh)
    except FileNotFoundError as exc:
        print(f'[parse_logs] LỖI: thiếu file CSV - {exc}', file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f'[parse_logs] LỖI dữ liệu: {exc}', file=sys.stderr)
        return 1
    out_csv = args.out_dir.resolve() / 'router_timeseries.csv'
    write_long_csv(rows, out_csv)
    n_samples = max((r['sample_idx'] for r in rows)) + 1
    print(f'[parse_logs] OK - {len(rows):,} dòng = {n_samples} sample × {mesh.num_nodes} router')
    print(f'[parse_logs] capacity_per_router = {mesh.buffer_capacity_per_router} flit (P={mesh.num_ports} × num_vcs={mesh.num_vcs} × vc_buf_size={mesh.vc_buf_size})')
    print(f'[parse_logs] output: {out_csv}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
