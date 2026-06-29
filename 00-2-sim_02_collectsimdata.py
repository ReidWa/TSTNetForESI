"""Aggregate training simulation parts into a single collection directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Merge simulation parts into a single dataset folder')
	parser.add_argument('--root', default=r'E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\simulationV3\trainset',
						help='训练集根目录（包含各个 part 子目录）')
	parser.add_argument('--pattern', default='time_jitter_0.5ms_part', help='匹配训练 part 的名称前缀')
	parser.add_argument('--subject', default='sub-01', help='被试 ID')
	parser.add_argument('--dest', default='collection_all', help='汇总后的目录名')
	parser.add_argument('--run-digits', type=int, default=5, help='run 目录编号补零宽度')
	parser.add_argument('--dry-run', action='store_true', help='仅打印将执行的操作，不实际复制')
	return parser.parse_args()


def list_parts(root: Path, pattern: str) -> List[Path]:
	return sorted([p for p in root.iterdir() if p.is_dir() and pattern in p.name])


def copy_runs(parts: Iterable[Path], subject: str, dest_runs: Path, run_digits: int, dry_run: bool) -> List[pd.DataFrame]:
	if dest_runs.exists() and not dry_run:
		shutil.rmtree(dest_runs)
	dest_runs.mkdir(parents=True, exist_ok=True)

	global_idx = 0
	csv_frames: List[pd.DataFrame] = []

	for part in parts:
		part_root = part / subject
		runs_dir = part_root / 'runs_data'
		if not runs_dir.exists():
			print(f'[Warn] {runs_dir} 不存在，跳过该 part')
			continue

		csv_path = part_root / 'simulation_info.csv'
		if csv_path.exists():
			df = pd.read_csv(csv_path)
			df.insert(0, 'run_index', range(global_idx, global_idx + len(df)))
			csv_frames.append(df)

		for run_dir in sorted(runs_dir.iterdir()):
			if not run_dir.is_dir():
				continue
			new_name = f'run-{global_idx:0{run_digits}d}'
			dest_run = dest_runs / new_name
			print(f'[Run] {run_dir.name} -> {new_name}')
			if not dry_run:
				shutil.copytree(run_dir, dest_run)
			global_idx += 1

	print(f'[Run] 总计复制 {global_idx} 个 run 目录')
	return csv_frames


def save_combined_csv(frames: List[pd.DataFrame], dest_subject_root: Path, dry_run: bool) -> None:
	if not frames:
		print('[Info] 未找到任何 simulation_info.csv，跳过合并')
		return
	combined = pd.concat(frames, ignore_index=True)
	csv_dest = dest_subject_root / 'simulation_info.csv'
	print(f'[CSV] 写入合并后的 metadata 到 {csv_dest}')
	if not dry_run:
		combined.to_csv(csv_dest, index=False)


def main() -> None:
	args = parse_args()
	root = Path(args.root)
	parts = list_parts(root, args.pattern)

	if not parts:
		raise RuntimeError(f'在 {root} 下未找到匹配 {args.pattern} 的子目录')

	dest_subject_root = root / args.dest / args.subject
	dest_runs = dest_subject_root / 'runs_data'

	print(f'[Init] 共发现 {len(parts)} 个 part: {[p.name for p in parts]}')

	csv_frames = copy_runs(parts, args.subject, dest_runs, args.run_digits, args.dry_run)
	save_combined_csv(csv_frames, dest_subject_root, args.dry_run)
	print('[Done] 汇总完成。')


if __name__ == '__main__':
	main()
