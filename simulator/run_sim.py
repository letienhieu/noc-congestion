from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
EXPECTED_CSV_KEYS = ('injected_flits_out', 'ejected_flits_out', 'received_flits_out', 'sent_flits_out', 'stored_flits_out', 'outstanding_credits_out')
DEFAULT_BOOKSIM = Path(__file__).resolve().parent / 'booksim2' / 'src' / 'booksim'

def parse_config_outputs(cfg_path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for raw_line in cfg_path.read_text().splitlines():
        line = raw_line.split('//', 1)[0].strip()
        if not line or '=' not in line:
            continue
        line = line.rstrip(';').strip()
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if key in EXPECTED_CSV_KEYS and value:
            outputs[key] = value
    return outputs

def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def run_booksim(*, booksim_bin: Path, cfg_path: Path, work_dir: Path) -> subprocess.CompletedProcess[str]:
    cmd = [str(booksim_bin), str(cfg_path)]
    proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, check=False)
    return proc

def main(argv: Iterable[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Run BookSim2 + collect TRACK_FLOWS CSV.')
    parser.add_argument('--config', required=True, type=Path, help='File cfg đầu vào')
    parser.add_argument('--out-dir', required=True, type=Path, help='Thư mục chứa CSV + log')
    parser.add_argument('--booksim', type=Path, default=DEFAULT_BOOKSIM, help=f'Đường dẫn binary booksim (mặc định: {DEFAULT_BOOKSIM})')
    parser.add_argument('--keep-existing', action='store_true', help='Không xoá out-dir nếu đã tồn tại (mặc định: xoá rồi tạo lại)')
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg_path: Path = args.config.resolve()
    out_dir: Path = args.out_dir.resolve()
    booksim_bin: Path = args.booksim.resolve()
    if not cfg_path.is_file():
        print(f'[run_sim] LỖI: không thấy config {cfg_path}', file=sys.stderr)
        return 2
    if not booksim_bin.is_file():
        print(f'[run_sim] LỖI: không thấy booksim binary {booksim_bin}', file=sys.stderr)
        return 2
    outputs = parse_config_outputs(cfg_path)
    if not outputs:
        print('[run_sim] CẢNH BÁO: config không khai báo *_out nào - BookSim sẽ chạy nhưng không sinh CSV per-router.', file=sys.stderr)
    if args.keep_existing:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        ensure_clean_dir(out_dir)
    print(f'[run_sim] config = {cfg_path}')
    print(f'[run_sim] out_dir = {out_dir}')
    print(f'[run_sim] binary = {booksim_bin}')
    print(f'[run_sim] mong đợi CSV: {sorted(outputs.values())}')
    proc = run_booksim(booksim_bin=booksim_bin, cfg_path=cfg_path, work_dir=out_dir)
    log_path = out_dir / 'booksim_run.log'
    log_path.write_text(proc.stdout + '\n--- STDERR ---\n' + proc.stderr)
    missing = [v for v in outputs.values() if not (out_dir / v).is_file()]
    if missing:
        print(f'[run_sim] LỖI: thiếu CSV {missing}. Return code={proc.returncode}. Xem {log_path}', file=sys.stderr)
        return 1
    print(f'[run_sim] OK - return code={proc.returncode}, đã sinh {len(outputs)} CSV trong {out_dir}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
