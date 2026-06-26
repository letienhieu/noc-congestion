from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable
from neo4j import Driver, GraphDatabase
DEFAULT_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7688')
DEFAULT_USER = os.environ.get('NEO4J_USER', 'neo4j')
DEFAULT_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'neo4jpass')
INGEST_BATCH_SIZE = 1000
SCHEMA_STATEMENTS: tuple[str, ...] = ('\n    CREATE CONSTRAINT router_unique IF NOT EXISTS\n        FOR (r:Router) REQUIRE (r.mesh_id, r.id) IS UNIQUE\n    ', '\n    CREATE CONSTRAINT snapshot_unique IF NOT EXISTS\n        FOR (s:Snapshot) REQUIRE (s.run_id, s.sample_idx) IS UNIQUE\n    ', '\n    CREATE INDEX snapshot_sample_idx IF NOT EXISTS\n        FOR (s:Snapshot) ON (s.sample_idx)\n    ')

def init_schema(driver: Driver) -> None:
    with driver.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            session.run(stmt)
    print('[ingest] schema OK - đã tạo constraints + indexes.')

def _mesh_neighbors_2d(k: int) -> Iterable[tuple[int, int, str]]:
    for y in range(k):
        for x in range(k):
            src = y * k + x
            if x + 1 < k:
                dst = y * k + (x + 1)
                yield (src, dst, 'E')
                yield (dst, src, 'W')
            if y + 1 < k:
                dst = (y + 1) * k + x
                yield (src, dst, 'S')
                yield (dst, src, 'N')

def ingest_topology(driver: Driver, *, mesh_id: str, k: int, n: int) -> None:
    if n != 2:
        raise NotImplementedError(f'Hiện chỉ hỗ trợ mesh 2D, được yêu cầu n={n}')
    nodes = [{'id': y * k + x, 'x': x, 'y': y, 'mesh_id': mesh_id} for y in range(k) for x in range(k)]
    edges = [{'src': s, 'dst': d, 'direction': dr, 'mesh_id': mesh_id} for s, d, dr in _mesh_neighbors_2d(k)]
    with driver.session() as session:
        session.run('\n            UNWIND $rows AS row\n            MERGE (r:Router {mesh_id: row.mesh_id, id: row.id})\n            SET r.x = row.x, r.y = row.y\n            ', rows=nodes)
        session.run('\n            UNWIND $rows AS row\n            MATCH (a:Router {mesh_id: row.mesh_id, id: row.src})\n            MATCH (b:Router {mesh_id: row.mesh_id, id: row.dst})\n            MERGE (a)-[l:LINK {direction: row.direction}]->(b)\n            ', rows=edges)
    print(f'[ingest] topology OK - {len(nodes)} Router + {len(edges)} LINK (mesh_id={mesh_id}, k={k}x{k}).')

def _load_csv(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({'sample_idx': int(r['sample_idx']), 'router_id': int(r['router_id']), 'injected': int(r['injected']), 'ejected': int(r['ejected']), 'received_total': int(r['received_total']), 'sent_total': int(r['sent_total']), 'stored_total': int(r['stored_total']), 'buffer_occupancy_norm': float(r['buffer_occupancy_norm'])})
    return rows

def _batch(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]

def ingest_run(driver: Driver, *, csv_path: Path, run_id: str, mesh_id: str, sample_period: int) -> None:
    rows = _load_csv(csv_path)
    if not rows:
        print(f'[ingest] CSV {csv_path} rỗng - bỏ qua.', file=sys.stderr)
        return
    sample_indices = sorted({r['sample_idx'] for r in rows})
    snapshots = [{'run_id': run_id, 'sample_idx': s, 'cycle_start': s * sample_period, 'cycle_end': (s + 1) * sample_period} for s in sample_indices]
    with driver.session() as session:
        session.run('\n            UNWIND $rows AS row\n            MERGE (s:Snapshot {run_id: row.run_id, sample_idx: row.sample_idx})\n            SET s.cycle_start = row.cycle_start, s.cycle_end = row.cycle_end\n            ', rows=snapshots)
        next_pairs = [{'run_id': run_id, 'a': sample_indices[i], 'b': sample_indices[i + 1]} for i in range(len(sample_indices) - 1)]
        if next_pairs:
            session.run('\n                UNWIND $rows AS row\n                MATCH (a:Snapshot {run_id: row.run_id, sample_idx: row.a})\n                MATCH (b:Snapshot {run_id: row.run_id, sample_idx: row.b})\n                MERGE (a)-[:NEXT]->(b)\n                ', rows=next_pairs)
        total = 0
        for chunk in _batch(rows, INGEST_BATCH_SIZE):
            session.run('\n                UNWIND $rows AS row\n                MATCH (s:Snapshot {run_id: $run_id, sample_idx: row.sample_idx})\n                MATCH (r:Router {mesh_id: $mesh_id, id: row.router_id})\n                MERGE (rs:RouterState {\n                    run_id: $run_id,\n                    sample_idx: row.sample_idx,\n                    router_id: row.router_id\n                })\n                SET rs.injected = row.injected,\n                    rs.ejected = row.ejected,\n                    rs.received_total = row.received_total,\n                    rs.sent_total = row.sent_total,\n                    rs.stored_total = row.stored_total,\n                    rs.buffer_occupancy_norm = row.buffer_occupancy_norm\n                MERGE (rs)-[:OBSERVED_AT]->(r)\n                MERGE (rs)-[:IN_SNAPSHOT]->(s)\n                ', rows=chunk, run_id=run_id, mesh_id=mesh_id)
            total += len(chunk)
    print(f'[ingest] run OK - {len(snapshots)} Snapshot, {total} RouterState (run_id={run_id}).')

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Ingest NoC dataset into Neo4j')
    parser.add_argument('--uri', default=DEFAULT_URI)
    parser.add_argument('--user', default=DEFAULT_USER)
    parser.add_argument('--password', default=DEFAULT_PASSWORD)
    parser.add_argument('--init-schema', action='store_true', help='Tạo constraints/indexes; có thể combine với ingest')
    parser.add_argument('--csv', type=Path, help='router_timeseries.csv của 1 run (bắt buộc nếu ingest)')
    parser.add_argument('--run-id', help='Tên run, ví dụ mesh_4x4_uniform_smoke')
    parser.add_argument('--mesh-id', default='mesh_4x4')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--sample-period', type=int, default=100, help='cycle count mỗi sample period (chỉ để gắn cycle_start/end)')
    parser.add_argument('--skip-topology', action='store_true', help='Đã ingest topology rồi, không lặp lại')
    args = parser.parse_args(argv)
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        if args.init_schema:
            init_schema(driver)
        if args.csv is not None:
            if not args.run_id:
                print('[ingest] LỖI: thiếu --run-id khi ingest dữ liệu.', file=sys.stderr)
                return 2
            if not args.csv.is_file():
                print(f'[ingest] LỖI: không thấy CSV {args.csv}', file=sys.stderr)
                return 2
            if not args.skip_topology:
                ingest_topology(driver, mesh_id=args.mesh_id, k=args.k, n=args.n)
            ingest_run(driver, csv_path=args.csv.resolve(), run_id=args.run_id, mesh_id=args.mesh_id, sample_period=args.sample_period)
        elif not args.init_schema:
            parser.print_help()
            return 2
    finally:
        driver.close()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
