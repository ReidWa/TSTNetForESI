"""为每个 run 生成子网络训练标签 (noise_ratio, sparsity)。

遍历 collection_all 下所有 run 文件夹，读取 sim_source.npz 和 sim_run_eegs.npz，
计算每个 EEG trial 对应的两个物理参数标签，保存为 labels.npz；
运行结束后汇总全数据集的数值分布（摘要 + 可选直方图）。

用法:
    python 00-2-sim_03_generate_labels.py

运行前请在下方 ``CONFIG`` 中修改路径与开关。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# =====================================================================
# 运行配置（在此修改，不使用命令行参数）
# =====================================================================


@dataclass
class LabelGenConfig:
    """标签生成与分布统计的配置。"""

    # collection_all 根目录，其下为 <subject>/runs_data/run-*/
    root: Path = field(
        default_factory=lambda: Path(
            r'E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\simulationV3\trainset\collection_all'
        )
    )
    subject: str = 'sub-01'

    # 是否覆盖已存在的 labels.npz
    overwrite: bool = True

    # True：不重新生成，只从已有 labels.npz 汇总分布统计
    stats_only: bool = False

    # 统计输出目录；为 None 时使用 <root>/<subject>/stats_labels
    stats_dir: Optional[Path] = None

    # 是否在 stats 目录保存直方图 PNG（需安装 matplotlib）
    save_plots: bool = True


# 脚本入口使用的配置实例（修改此处即可）
CONFIG = LabelGenConfig()


# =====================================================================
# 核心计算函数
# =====================================================================

def compute_sparsity(source: np.ndarray) -> float:
    """基于 Hoyer 公式计算空间稀疏度。

    Hoyer(x) = (sqrt(N) - ||x||_1 / ||x||_2) / (sqrt(N) - 1)

    先沿时间维求能量得到空间能量分布 e_i = ||s_i(t)||_2，
    再对该向量计算 Hoyer sparsity。

    Args:
        source: (N, T) 源信号矩阵

    Returns:
        sparsity: float, 值域 [0, 1]
            1 → 极稀疏（单节点激活）
            0 → 完全均匀分布
    """
    energy = np.linalg.norm(np.asarray(source, dtype=np.float64), axis=1)
    n = int(energy.shape[0])
    if n <= 1:
        return 1.0
    l1 = np.abs(energy).sum()
    l2 = np.linalg.norm(energy)
    if l2 < 1e-30:
        return 1.0
    sqrt_n = np.sqrt(n)
    hoyer = float((sqrt_n - l1 / l2) / (sqrt_n - 1.0))
    return 2*hoyer-1.0


def compute_noise_ratio(target_snr_db: float, target_snir_db: float) -> float:
    """根据 SNR 和 SNIR (dB) 计算 Fro 归一化后噪声能量占比。

    混合公式: noisy = clean_unit + sp * pink_unit + ss * sensor_unit
    其中三个 unit 分量的 ||·||_F = 1，且统计独立（交叉项≈0）。

    归一化后: noise_ratio = (sp² + ss²) / (1 + sp² + ss²)

    Args:
        target_snr_db:  传感器噪声 SNR (dB)
        target_snir_db: 粉红噪声 SNIR (dB)

    Returns:
        noise_ratio: float, 值域 [0, 1)
    """
    sp = 10.0 ** (-float(target_snir_db) / 20.0)
    ss = 10.0 ** (-float(target_snr_db) / 20.0)
    return float((sp ** 2 + ss ** 2) / (1.0 + sp ** 2 + ss ** 2))


def _load_snr_snir_arrays(eeg_npz: Any) -> Tuple[np.ndarray, np.ndarray]:
    """从 sim_run_eegs.npz 读取每 trial 的 SNR / SNIR，兼容旧键名。"""
    keys = set(eeg_npz.files)
    if 'target_snr' in keys:
        snr = np.asarray(eeg_npz['target_snr'], dtype=np.float64).ravel()
    elif 'eeg_snrs' in keys:
        snr = np.asarray(eeg_npz['eeg_snrs'], dtype=np.float64).ravel()
    else:
        raise KeyError(
            "sim_run_eegs.npz 中未找到 'target_snr' 或 'eeg_snrs'，无法计算 noise_ratio。"
        )

    if 'target_snir' in keys:
        snir = np.asarray(eeg_npz['target_snir'], dtype=np.float64).ravel()
    elif 'eeg_snirs' in keys:
        snir = np.asarray(eeg_npz['eeg_snirs'], dtype=np.float64).ravel()
    else:
        raise KeyError(
            "sim_run_eegs.npz 中未找到 'target_snir' 或 'eeg_snirs'，无法计算 noise_ratio。"
        )

    if snr.shape != snir.shape:
        raise ValueError(
            f'SNR 与 SNIR 长度不一致: {snr.shape} vs {snir.shape}'
        )
    return snr, snir


def _load_source_matrix(src_npz: Any) -> np.ndarray:
    """读取 sim_source.npz 中的 (N, T) 源矩阵。"""
    if 'data' not in src_npz.files:
        raise KeyError("sim_source.npz 中缺少 'data' 数组。")
    src = np.asarray(src_npz['data'], dtype=np.float64)
    if src.ndim == 3:
        # (n_trials, N, T) — 每 trial 单独算稀疏度
        return src
    if src.ndim != 2:
        raise ValueError(f"期望源形状为 (N,T) 或 (trial,N,T)，得到 {src.shape}")
    return src


# =====================================================================
# 分布统计
# =====================================================================

def describe_array(name: str, x: np.ndarray) -> Dict[str, Any]:
    """计算一维数组的描述性统计（用于打印与 JSON）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {'name': name, 'count': 0}
    qs = (5, 25, 50, 75, 95)
    pct = {f'p{q}': float(np.percentile(x, q)) for q in qs}
    return {
        'name': name,
        'count': int(x.size),
        'min': float(np.min(x)),
        'max': float(np.max(x)),
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        **pct,
    }


def save_distribution_report(
    noise_ratio: np.ndarray,
    sparsity: np.ndarray,
    out_dir: Path,
    make_plots: bool,
) -> None:
    """保存 JSON 摘要；可选生成双直方图。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = {
        'noise_ratio': describe_array('noise_ratio', noise_ratio),
        'sparsity': describe_array('sparsity', sparsity),
        'note': (
            'noise_ratio 与 sparsity 均按「每个 EEG trial 一条」统计；'
            '同一 run 内 sparsity 在 trial 间重复。'
        ),
    }
    json_path = out_dir / 'labels_distribution_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'[Save] {json_path}')

    # 人类可读文本
    txt_path = out_dir / 'labels_distribution_summary.txt'
    lines = [
        '=== labels 数值分布（按 EEG trial）===',
        '',
        '--- noise_ratio ---',
        _format_stats_block(rep['noise_ratio']),
        '',
        '--- sparsity ---',
        _format_stats_block(rep['sparsity']),
        '',
    ]
    txt_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[Save] {txt_path}')

    if not make_plots:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[Warn] 未安装 matplotlib，跳过直方图。')
        return

    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    nr = np.asarray(noise_ratio, dtype=np.float64).ravel()
    sp = np.asarray(sparsity, dtype=np.float64).ravel()

    axes[0].hist(nr, bins=64, color='#4C72B0', edgecolor='white', alpha=0.9)
    axes[0].set_title('noise_ratio 分布')
    axes[0].set_xlabel('noise_ratio')
    axes[0].set_ylabel('trial 计数')

    axes[1].hist(sp, bins=64, color='#55A868', edgecolor='white', alpha=0.9)
    axes[1].set_title('sparsity 分布（每 trial 一条，同 run 重复）')
    axes[1].set_xlabel('sparsity (Hoyer)')
    axes[1].set_ylabel('trial 计数')

    fig.suptitle(f'标签分布 (N={nr.size:,} trials)', fontsize=12)
    fig.tight_layout()
    png_path = out_dir / 'labels_distribution_hist.png'
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'[Save] {png_path}')


def _format_stats_block(d: Dict[str, Any]) -> str:
    if d.get('count', 0) == 0:
        return '  (无数据)'
    lines = [
        f"  count = {d['count']}",
        f"  min / max = {d['min']:.6f} / {d['max']:.6f}",
        f"  mean ± std = {d['mean']:.6f} ± {d['std']:.6f}",
        f"  p5 / p25 / p50 / p75 / p95 = "
        f"{d['p5']:.6f}, {d['p25']:.6f}, {d['p50']:.6f}, {d['p75']:.6f}, {d['p95']:.6f}",
    ]
    return '\n'.join(lines)


def collect_stats_from_existing_labels(
    run_dirs: List[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """从已生成的 labels.npz 汇总全数据集（用于 --stats-only）。"""
    noise_list: List[np.ndarray] = []
    sp_list: List[np.ndarray] = []
    for run_dir in tqdm(run_dirs, desc='Loading labels.npz'):
        lp = run_dir / 'labels.npz'
        if not lp.exists():
            continue
        z = np.load(lp, allow_pickle=True)
        if 'noise_ratio' not in z.files or 'sparsity' not in z.files:
            z.close()
            continue
        nr = np.asarray(z['noise_ratio'], dtype=np.float64).ravel()
        sp = np.asarray(z['sparsity'], dtype=np.float64).ravel()
        z.close()
        n = min(nr.size, sp.size)
        if n == 0:
            continue
        noise_list.append(nr[:n])
        sp_list.append(sp[:n])
    if not noise_list:
        return np.array([]), np.array([])
    return np.concatenate(noise_list), np.concatenate(sp_list)


# =====================================================================
# 主流程
# =====================================================================

def process_run(
    run_dir: Path,
    overwrite: bool,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """处理单个 run 文件夹，生成 labels.npz。

    Returns:
        (成功写入, noise_ratio 向量或 None, 与 trial 对齐的 sparsity 向量或 None)
    """
    label_path = run_dir / 'labels.npz'
    if label_path.exists() and not overwrite:
        return False, None, None

    try:
        src_data = np.load(run_dir / 'sim_source.npz', allow_pickle=True)
        source = _load_source_matrix(src_data)
        src_data.close()

        eeg_data = np.load(run_dir / 'sim_run_eegs.npz', allow_pickle=True)
        target_snr, target_snir = _load_snr_snir_arrays(eeg_data)
        eeg_data.close()
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f'[Skip] {run_dir.name}: {e}')
        return False, None, None

    n_trials = int(target_snr.shape[0])

    if source.ndim == 3:
        if source.shape[0] != n_trials:
            print(
                f'[Warn] {run_dir.name}: 源 trial 数 {source.shape[0]} != EEG trial 数 {n_trials}，按最小长度截断'
            )
        n_use = min(source.shape[0], n_trials)
        sparsities = np.array(
            [compute_sparsity(source[t]) for t in range(n_use)],
            dtype=np.float32,
        )
        noise_ratios = np.array(
            [
                compute_noise_ratio(target_snr[t], target_snir[t])
                for t in range(n_use)
            ],
            dtype=np.float32,
        )
    else:
        sp = float(compute_sparsity(source))
        sparsities = np.full(n_trials, sp, dtype=np.float32)
        noise_ratios = np.array(
            [
                compute_noise_ratio(target_snr[t], target_snir[t])
                for t in range(n_trials)
            ],
            dtype=np.float32,
        )

    np.savez(
        label_path,
        noise_ratio=noise_ratios,
        sparsity=sparsities,
    )
    return True, noise_ratios, sparsities


def main(cfg: LabelGenConfig | None = None) -> None:
    """执行标签生成与统计。默认使用模块级 ``CONFIG``；也可传入自定义 ``LabelGenConfig``。"""
    c = cfg if cfg is not None else CONFIG
    root = Path(c.root)
    runs_dir = root / c.subject / 'runs_data'
    if not runs_dir.is_dir():
        raise FileNotFoundError(f'runs_data 目录不存在: {runs_dir}')

    stats_dir = Path(c.stats_dir) if c.stats_dir is not None else root / c.subject / 'stats_labels'
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    print(f'[Init] {runs_dir} 下共 {len(run_dirs)} 个 run')

    if c.stats_only:
        noise_all, sp_all = collect_stats_from_existing_labels(run_dirs)
        if noise_all.size == 0:
            print('[Error] 未找到任何有效的 labels.npz，请先将 CONFIG.stats_only 设为 False 生成标签。')
            return
        print(f'[Info] 汇总 {noise_all.size} 条 EEG trial 的标签统计')
        save_distribution_report(
            noise_all,
            sp_all,
            stats_dir,
            make_plots=c.save_plots,
        )
        print('[Done] 仅统计完成。')
        return

    processed = 0
    skipped = 0
    noise_chunks: List[np.ndarray] = []
    sp_chunks: List[np.ndarray] = []

    for run_dir in tqdm(run_dirs, desc='Generating labels'):
        ok, nr, sp = process_run(run_dir, c.overwrite)
        if ok:
            processed += 1
            if nr is not None and sp is not None:
                noise_chunks.append(np.asarray(nr, dtype=np.float64).ravel())
                sp_chunks.append(np.asarray(sp, dtype=np.float64).ravel())
        else:
            skipped += 1
            # 未覆盖且已存在：仍纳入统计
            if not c.overwrite:
                lp = run_dir / 'labels.npz'
                if lp.exists():
                    z = np.load(lp, allow_pickle=True)
                    if 'noise_ratio' in z.files and 'sparsity' in z.files:
                        noise_chunks.append(
                            np.asarray(z['noise_ratio'], dtype=np.float64).ravel()
                        )
                        sp_chunks.append(
                            np.asarray(z['sparsity'], dtype=np.float64).ravel()
                        )
                    z.close()

    print(
        f'[Done] 本次新写入 {processed} 个 run；'
        f'未写入 {skipped} 个（含已存在未覆盖、或读数据失败）。'
    )

    if noise_chunks:
        noise_all = np.concatenate(noise_chunks)
        sp_all = np.concatenate(sp_chunks)
        print(f'[Info] 全数据集共 {noise_all.size} 条 EEG trial 用于分布统计')
        save_distribution_report(
            noise_all,
            sp_all,
            stats_dir,
            make_plots=c.save_plots,
        )
    else:
        print('[Warn] 没有可用于统计的标签向量（检查数据路径与 overwrite）。')


if __name__ == '__main__':
    main()
