import argparse
import math
import os.path as op
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List

import mne
import numpy as np

from data.dataPreprocessor import Simulate
from utils.fx_bids_local_mi import load_bids
from utils.utils_train import load_forward_fixed


# settings 各字段含义：
# random_type           : 参数随机方式（'range': 在范围内随机取值; 'choice': 从给定列表中随机选取）。
# method                : 源采样策略（standard/其他自定义）。
# num_firing_sources    : 每次激活的源数目（单值或区间，区间内随机采样整数；choice模式下为候选列表）。
# extents               : 源扩散直径范围(mm)，影响 region_growing 阶数或 spherical 半径。
# amplitudes            : 源幅值范围(nAm)，内部会转换为 A·m 以匹配前向矩阵。
# shapes                : 单源的空间形状（gaussian/flat/mixed）。
# duration_of_stimulate : 脉冲宽度(ms)，影响 mono/bipulse 波形。
# time_jitter           : 时间抖动；对方波是 ms 的起始偏移，对正弦 pulse 是相位(弧度)。
# signal_time_range     : 信号时间窗(s)，即模拟 EEG 的 tmin,tmax。
# eeg_snr               : 传感器域目标 SNR(dB)，用于 add_noise 的缩放。
# eeg_snir              : 生理/背景噪声的目标 SNIR(dB)，用于 mix_signal_and_noise 的粉红噪声权重。
# beta                  : 源随机波形的谱指数β，仅 source_time_course='random' 时生效。
# beta_noise            : 传感器噪声的谱指数β_noise，用于生成 1/f^β 噪声。
# source_spread         : 源扩散方式（region_growing/spherical/mixed）。
# source_number_weighting: 是否对源数量做 1/n 加权，倾向少量源点。
# source_time_course    : 源时程类型（pulse/bipulse/mono_pulse/random）。
# random_seed           : 随机种子，保证可复现。
# sample_frequency      : 采样率(Hz)，None 表示沿用输入 Epochs。
# Pre_EA                : 是否额外保存 ESA 对齐后的 EEG。
# Pre_White             : 是否额外保存通道白化后的 EEG。
# save_figs             : 是否保存每次仿真的可视化结果。


GLOBAL_CONTEXT: Dict[str, Any] = {}


def _init_worker(dir_bids: str, task: str, subj: str, run: str) -> None:
    """Load heavy objects once per worker process."""
    global GLOBAL_CONTEXT
    mne.set_log_level('ERROR')  # 抑制 MNE 日志
    fwd = load_forward_fixed(op.join(dir_bids, 'derivatives', 'sourcemodelling', subj, 'fwd', f'{subj}_fwd.fif'))
    epo = load_bids(dir_bids, subj, task, run)
    GLOBAL_CONTEXT = {
        'fwd': fwd,
        'info': epo.info,
        'dir_bids': dir_bids,
        'subj': subj,
    }


def _simulate_job(job: Dict[str, Any]) -> str:
    """Worker entry: 执行单个模拟任务。"""
    settings = deepcopy(job['settings'])
    simulate = Simulate(
        fwd=GLOBAL_CONTEXT['fwd'],
        info=GLOBAL_CONTEXT['info'],
        save_path=job['save_path'],
        settings=settings,
        subject_id=GLOBAL_CONTEXT['subj'],
        save_tag=job['save_tag'],
    )
    simulate.simulate(job['num_sources'], job['num_trials'])
    del simulate
    return job['name']


def clone_settings(base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    data = deepcopy(base)
    data.update(overrides)
    return data


def build_training_jobs(train_settings: Dict[str, Any], save_path: str, base_tag: str,
                        total_sources: int, chunk_size: int, trials: int) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    chunk_size = max(1, chunk_size)
    total_chunks = math.ceil(total_sources / chunk_size)
    base_seed = train_settings.get('random_seed', 42)
    seed_rng = np.random.RandomState(base_seed)
    job_seeds = seed_rng.randint(0, 2**31, size=total_chunks)
    for idx in range(total_chunks):
        remaining = total_sources - idx * chunk_size
        current_sources = min(chunk_size, remaining)
        job_settings = clone_settings(train_settings, random_seed=int(job_seeds[idx]))
        jobs.append({
            'name': f'train_part_{idx + 1}',
            'save_tag': f'{base_tag}_part{idx + 1:02d}',
            'save_path': save_path,
            'settings': job_settings,
            'num_sources': current_sources,
            'num_trials': trials,
        })
    return jobs


def build_test_jobs(base_settings: Dict[str, Any], save_path: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    base_seed = base_settings.get('random_seed', 42)
    n_test_jobs = len([1, 2, 3, 4]) + len([-5, 0, 5, 10]) + len([0, 20, 40, 60])
    seed_rng = np.random.RandomState(base_seed + 1)
    job_seeds = seed_rng.randint(0, 2**31, size=n_test_jobs)
    job_idx = 0

    # 激活源个数 sweep
    for num_src in [1, 2, 3, 4]:
        jobs.append({
            'name': f'firingsource_{num_src}',
            'save_tag': f'firingsource_{num_src}',
            'save_path': save_path,
            'settings': clone_settings(base_settings, num_firing_sources=num_src, random_seed=int(job_seeds[job_idx])),
            'num_sources': 100,
            'num_trials': 5,
        })
        job_idx += 1

    # SNR sweep
    for snr in [-5, 0, 5, 10]:
        tag = 'minus5dB' if snr == -5 else f'{snr}dB'
        jobs.append({
            'name': f'snr_{tag}',
            'save_tag': f'snr_{tag}',
            'save_path': save_path,
            'settings': clone_settings(base_settings, eeg_snr=snr, num_firing_sources=1, random_seed=int(job_seeds[job_idx])),
            'num_sources': 100,
            'num_trials': 5,
        })
        job_idx += 1

    # extents sweep
    for extent in [0, 10, 20, 30]:
        jobs.append({
            'name': f'extents_{extent}mm',
            'save_tag': f'extents_{extent}mm',
            'save_path': save_path,
            'settings': clone_settings(base_settings, extents=extent, eeg_snr=10, random_seed=int(job_seeds[job_idx])),
            'num_sources': 100,
            'num_trials': 5,
        })
        job_idx += 1

    return jobs


def execute_jobs(jobs: List[Dict[str, Any]], max_workers: int,
                 init_args: Dict[str, str]) -> None:
    if not jobs:
        print('[Info] 没有需要执行的任务。')
        return

    if max_workers <= 1:
        print('[Runner] 单进程顺序执行任务。')
        _init_worker(**init_args)
        for job in jobs:
            print(f"[Runner] 开始任务 {job['name']} ({job['save_tag']}) ...")
            _simulate_job(job)
        return

    print(f"[Runner] 使用 {max_workers} 个进程并行执行 {len(jobs)} 个任务。")
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_init_worker,
                             initargs=(init_args['dir_bids'], init_args['task'], init_args['subj'], init_args['run'])) as executor:
        future_to_job = {executor.submit(_simulate_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            future.result()
            print(f"[Runner] 任务 {job['name']} 完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='批量 EEG 仿真脚本')
    parser.add_argument('--dir-bids', default=r'E:\2_ESIdiff\Dataset\localize-mi-data')
    parser.add_argument('--task', default='seegstim')
    parser.add_argument('--subj', default='sub-01')
    parser.add_argument('--run', default='run-01')
    parser.add_argument('--train-total', type=int, default=20000, help='训练集需要模拟的源数量总和')
    parser.add_argument('--train-chunk-size', type=int, default=2000, help='训练集每批次模拟的源数量')
    parser.add_argument('--train-trials', type=int, default=20, help='训练集中每个源生成的 trial 数')
    parser.add_argument('--max-workers', type=int, default=12, help='并行进程数 (Windows 需 >=1)')
    parser.add_argument('--disable-figs', action='store_true', help='关闭所有绘图以提升速度')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dir_bids = args.dir_bids
    task = args.task
    subj = args.subj
    run = args.run

    print('[Init] 准备任务描述...')

    train_settings = {
        'random_type': 'range',
        'random_method_S_fireNum': 'standard',
        'random_method_extents': 'reciprocal',
        'num_firing_sources': (1, 5), 
        'extents': (0, 36),
        'amplitudes': 10,
        'shapes': 'gaussian',
        'duration_of_stimulate': 0.5,
        'time_jitter': 0.5,
        'signal_time_range': (-0.001, 0.002),
        'eeg_snr': (-10, 11),
        'eeg_snir': 5,
        'beta_noise': 1,
        'source_spread': 'region_growing',
        'source_time_course': 'bipulse',
        'random_seed': 42,
        'sample_frequency': None,
        'Pre_EA': True,
        'Pre_White': True,
        'fro_normalize_source': True,
        'save_figs': not args.disable_figs,
    }

    default_test_settings = {
        'random_type': 'range',
        'random_method_S_fireNum': 'standard',
        'random_method_extents': 'uniform',
        'num_firing_sources': 1,
        'extents': 10,
        'amplitudes': 10,
        'eeg_snr': 5,
        'eeg_snir': 5,
        'shapes': 'gaussian',
        'duration_of_stimulate': 0.5,
        'time_jitter': 0.5,
        'signal_time_range': (-0.001, 0.002),
        'beta_noise': 1,
        'source_spread': 'region_growing',
        'source_time_course': 'bipulse',
        'random_seed': 42,
        'sample_frequency': None,
        'Pre_EA': True,
        'Pre_White': True,
        'fro_normalize_source': True,
        'save_figs': not args.disable_figs,
    }

    train_jobs = build_training_jobs(
        train_settings=train_settings,
        save_path=op.join(dir_bids, 'derivatives', 'simulationV3', 'trainset'),
        base_tag='time_jitter_0.5ms',
        total_sources=args.train_total,
        chunk_size=args.train_chunk_size,
        trials=args.train_trials,
    )

    test_jobs = build_test_jobs(
        base_settings=default_test_settings,
        save_path=op.join(dir_bids, 'derivatives', 'simulationV3', 'testset'),
    )

    all_jobs = train_jobs + test_jobs
    print(f"[Init] 共定义 {len(all_jobs)} 个模拟任务。")

    init_args = {'dir_bids': dir_bids, 'task': task, 'subj': subj, 'run': run}
    execute_jobs(all_jobs, args.max_workers, init_args)
    print('[Done] 所有模拟任务完成。')


if __name__ == '__main__':
    main()