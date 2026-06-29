import numpy as np
from typing import Callable, Sequence, Union


Number = Union[int, float]
Spec = Union[Number, Sequence[Number], Callable[[float, int], Number]]

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _ensure_time_axis(times, total_duration_ms, sampling_rate):
    """Return time axis in seconds and total sample count."""
    if times is None:
        if total_duration_ms is None:
            raise ValueError("total_duration_ms 必须在未提供 times 时指定")
        duration_s = float(total_duration_ms) / 1000.0
        total_samples = int(np.round(duration_s * sampling_rate))
        total_samples = max(total_samples, 1)
        time_axis = np.linspace(0.0, duration_s, total_samples, endpoint=False)
    else:
        time_axis = np.asarray(times, dtype=float)
        total_samples = time_axis.size
    return time_axis, total_samples


def _gaussian_sinusoid(times,
                       frequency_hz,
                       center_time,
                       sigma_s,
                       amplitude,
                       phase=0.0):
    """Compute sin(2πft+φ) · exp(-((t-τ)/ω)^2) scaled by amplitude."""
    if sigma_s <= 0:
        return np.zeros_like(times, dtype=float)
    times = np.asarray(times, dtype=float)
    t_rel = times - center_time
    sinusoid = np.sin(2.0 * np.pi * frequency_hz * t_rel + phase)
    envelope = np.exp(-(t_rel / sigma_s) ** 2)
    return amplitude * sinusoid * envelope


def _sample_onset_seconds(start_time_ms: float,
                          time_jitter_ms: float) -> float:
    """Sample onset time in seconds with symmetric jitter around start_time_ms."""
    onset_s = float(start_time_ms) / 1000.0
    jitter_ms = float(abs(time_jitter_ms)) if time_jitter_ms else 0.0
    if jitter_ms > 0.0:
        onset_s += np.random.uniform(-jitter_ms, jitter_ms) / 1000.0
    return onset_s


def _sigma_from_duration_ms(duration_ms: float) -> float:
    """Interpret duration (ms) as FWHM and convert to Gaussian sigma in seconds."""
    duration_ms = max(float(duration_ms), 1e-6)
    duration_s = duration_ms / 1000.0
    return duration_s * _FWHM_TO_SIGMA



def generate_biphasic_pulse(times=None,
                            amplitude_mA=8.0,
                            contact_distance_mm=2.0,
                            pulse_width_ms=0.5,
                            sampling_rate=8000,
                            total_duration_ms=None,
                            start_time_ms=None,
                            time_jitter=0.0,
                            interphase_gap_ms=0.0,
                            phase=np.pi/2):
    """使用高斯调制正弦生成双相（正/负）脉冲。

    参数说明：
        - time_jitter 以毫秒为单位，在 [-time_jitter, +time_jitter] 内随机偏移“起始”时间；
        - pulse_width_ms 表示单个相位的 FWHM（毫秒），用来确定高斯 envelope 的有效宽度。
    """

    assert (times is not None or total_duration_ms) and start_time_ms is not None, (
        "times 或 total_duration_ms 以及 start_time_ms 必须提供")

    time_axis, total_samples = _ensure_time_axis(times, total_duration_ms, sampling_rate)
    sigma_s = _sigma_from_duration_ms(pulse_width_ms)
    duration_s = max(float(pulse_width_ms), 1e-6) / 1000.0
    gap_s = max(float(interphase_gap_ms), 0.0) / 1000.0

    onset_pos = _sample_onset_seconds(start_time_ms, time_jitter)
    onset_neg = onset_pos + duration_s + gap_s
    pos_center = onset_pos + duration_s / 2.0
    neg_center = onset_neg + duration_s / 2.0
    carrier_frequency_hz = 1.0 / duration_s
    amplitude_scale = amplitude_mA * contact_distance_mm * 1e-6
    pos = _gaussian_sinusoid(time_axis,
                             frequency_hz=carrier_frequency_hz,
                             center_time=pos_center,
                             sigma_s=sigma_s,
                             amplitude=amplitude_scale,
                             phase=phase)
    neg = _gaussian_sinusoid(time_axis,
                             frequency_hz=carrier_frequency_hz,
                             center_time=neg_center,
                             sigma_s=sigma_s,
                             amplitude=-amplitude_scale,
                             phase=phase)
    signal = pos + neg
    if signal.size != total_samples:
        signal = signal.reshape(total_samples)
    return signal.astype(float)


def generate_monophasic_pulse(times=None,
                              amplitude_mA=8.0,
                              contact_distance_mm=2.0,
                              pulse_width_ms=0.5,
                              sampling_rate=8000,
                              total_duration_ms=None,
                              start_time_ms=None,
                              time_jitter=0.0,
                              phase=np.pi/2):
    """使用高斯调制正弦生成单相脉冲。

    time_jitter 以毫秒为单位，控制起始时间（相对于 0 点）的左右偏移；
    pulse_width_ms 控制单个脉冲相位的 FWHM，从而对应脉冲宽度。
    """

    assert (times is not None or total_duration_ms) and start_time_ms is not None, (
        "times 或 total_duration_ms 以及 start_time_ms 必须提供")

    time_axis, total_samples = _ensure_time_axis(times, total_duration_ms, sampling_rate)
    onset_s = _sample_onset_seconds(start_time_ms, time_jitter)
    sigma_s = _sigma_from_duration_ms(pulse_width_ms)
    duration_s = max(float(pulse_width_ms), 1e-6) / 1000.0
    center_time = onset_s + duration_s / 2.0
    carrier_frequency_hz = 1.0 / duration_s
    amplitude_scale = amplitude_mA * contact_distance_mm * 1e-6
    signal = _gaussian_sinusoid(time_axis,
                                frequency_hz=carrier_frequency_hz,
                                center_time=center_time,
                                sigma_s=sigma_s,
                                amplitude=amplitude_scale,
                                phase=phase)
    if signal.size != total_samples:
        signal = signal.reshape(total_samples)
    return signal.astype(float)


def generate_matlab_decay_wave(times=None,
                               amplitude_mA=8.0,
                               contact_distance_mm=2.0,
                               sampling_rate=1000,
                               total_duration_ms=None,
                               start_time_ms=None,
                               time_jitter=0.0,
                               frequency_hz=10.0,
                               decay_tau_ms=150.0,
                               phase=0.0):
    """低频正弦 + 指数衰减的波形（模仿 MATLAB 脚本），起点由 start_time_ms 决定。

    - frequency_hz: 载频，默认 10 Hz 对应 MATLAB f=[8–11] 的一类。
    - decay_tau_ms: 包络时间常数（ms），默认 150 ms，决定衰减快慢。
    - start_time_ms: 脉冲开始时间（ms）；time_jitter 在其附近随机抖动。
    """

    assert (times is not None or total_duration_ms) and start_time_ms is not None, (
        "times 或 total_duration_ms 以及 start_time_ms 必须提供")

    time_axis, total_samples = _ensure_time_axis(times, total_duration_ms, sampling_rate)
    onset_s = _sample_onset_seconds(start_time_ms, time_jitter)
    tau_s = max(float(decay_tau_ms), 1e-6) / 1000.0

    # 只在 onset 之后生效；onset 之前填 0
    t_rel = time_axis - onset_s
    envelope = np.exp(-np.clip(t_rel, a_min=0.0, a_max=None) / tau_s)
    sinusoid = np.sin(2.0 * np.pi * frequency_hz * t_rel + phase)

    amplitude_scale = amplitude_mA * contact_distance_mm * 1e-6
    signal = amplitude_scale * sinusoid * envelope
    signal = np.where(t_rel >= 0.0, signal, 0.0)

    if signal.size != total_samples:
        signal = signal.reshape(total_samples)
    return signal.astype(float)

def build_source_matrix_multi_trials(fwd, stim_info, biphasic, sampling_rate, times = None, t_min = None, t_max=None):
    """
    构建多trial自由方向EEG源信号矩阵（[n_trials, n_dipoles, T]）。
    
    参数：
    - fwd_source_nn: [n_dipoles, 3]，源空间的dipole坐标。
    - stim_info: dict, 使用load_sti_info函数加载的刺激信息。
        可选键：
        - waveform_mode='matlab_decay' 时，使用低频正弦+指数衰减；否则默认高斯包络正/负脉冲。
        - wave_frequency_hz / wave_decay_tau_ms 可覆盖默认 10 Hz / 150 ms。
    - biphasic: bool, 是否生成双相脉冲。localize-mi数据集使用双相脉冲。
    - sampling_rate: float, 采样率（Hz）。
    - t_min: float, 信号开始时间（秒）。
    - t_max: float, 信号结束时间（秒）。

    返回：
    - source_mtx: [n_trials, n_dipoles, T]，多trial源信号矩阵。
    """
    if times is None:
        assert (t_max - t_min)*1000 >= stim_info['durations'][0], "信号时间范围必须大于等于刺激持续时间"
        assert t_min <= 0, "t_min 必须早于或等于刺激点开始时间"
        assert t_max > 0, "t_max 必须大于刺激点开始时间"
    n_trials = len(stim_info['stim_coords'])
    fwd_source_rr = fwd['source_rr']
    n_dipoles = fwd_source_rr.shape[0]
    if times is None:
        T = int(np.round((t_max-t_min) * sampling_rate))
    else:
        T = len(times)

    
    source_mtx = np.zeros((n_trials, n_dipoles, T), dtype=float)
    for trial in range(n_trials):
        dir0 = np.array(stim_info['stim_coords'][trial][0])
        dir1 = np.array(stim_info['stim_coords'][trial][1])
        center = np.array(stim_info['stim_coords_bips'][trial])
        direction = dir0 - dir1
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1,0,0])  # 兜底：万一方向为0
        # 找最近dipole的index
        base_idx = np.argmin(np.linalg.norm(fwd_source_rr - center, axis=1))
        dipole_base = base_idx
        normals = fwd['source_nn']                            # Nx3 法向
        cos_theta = float(np.dot(direction, normals[base_idx]))
        start_ms = 0.0 if times is not None else (0 - t_min) * 1000.0
        if stim_info.get('waveform_mode', 'gaussian') == 'matlab_decay':
            freq_hz = float(stim_info.get('wave_frequency_hz', 10.0))
            tau_ms = float(stim_info.get('wave_decay_tau_ms', 150.0))
            if times is not None:
                pulse = generate_matlab_decay_wave(
                    times=times,
                    amplitude_mA=stim_info['strengths_mA'][trial],
                    contact_distance_mm=stim_info['stim_len_between_contacts'][trial],
                    sampling_rate=sampling_rate,
                    start_time_ms=start_ms,
                    frequency_hz=freq_hz,
                    decay_tau_ms=tau_ms)
            else:
                pulse = generate_matlab_decay_wave(
                    times=None,
                    amplitude_mA=stim_info['strengths_mA'][trial],
                    contact_distance_mm=stim_info['stim_len_between_contacts'][trial],
                    sampling_rate=sampling_rate,
                    total_duration_ms=(t_max - t_min) * 1000,
                    start_time_ms=start_ms,
                    frequency_hz=freq_hz,
                    decay_tau_ms=tau_ms)
        elif biphasic:
            if times is not None:
                pulse = generate_biphasic_pulse(times = times,
                                            amplitude_mA=stim_info['strengths_mA'][trial], 
                                            contact_distance_mm=stim_info['stim_len_between_contacts'][trial], 
                                            pulse_width_ms=stim_info['pulse_width'][trial], 
                                            sampling_rate=sampling_rate, 
                                            start_time_ms=start_ms)
            else: 
                pulse = generate_biphasic_pulse(times = None,
                                                amplitude_mA=stim_info['strengths_mA'][trial], 
                                                contact_distance_mm=stim_info['stim_len_between_contacts'][trial], 
                                                pulse_width_ms=stim_info['pulse_width'][trial], 
                                                sampling_rate=sampling_rate, 
                                                total_duration_ms=(t_max-t_min)*1000,
                                                start_time_ms=start_ms)
        else:
            if times is not None:
                pulse = generate_monophasic_pulse(times = times,
                                            amplitude_mA=stim_info['strengths_mA'][trial], 
                                            contact_distance_mm=stim_info['stim_len_between_contacts'][trial], 
                                            pulse_width_ms=stim_info['pulse_width'][trial], 
                                            sampling_rate=sampling_rate, 
                                            start_time_ms=start_ms)
            else:
                pulse = generate_monophasic_pulse(times = None,
                                                amplitude_mA=stim_info['strengths_mA'][trial], 
                                                contact_distance_mm=stim_info['stim_len_between_contacts'][trial], 
                                                pulse_width_ms=stim_info['pulse_width'][trial], 
                                                sampling_rate=sampling_rate, 
                                                total_duration_ms=(t_max-t_min)*1000,
                                                start_time_ms=start_ms)

        
        source_mtx[trial, dipole_base, :] = cos_theta*pulse
    return source_mtx





if __name__ == "__main__":
    import mne
    import sys
    from pathlib import Path
    nb_dir = Path().resolve()
    parent_dir = nb_dir.parent
    sys.path.insert(0, 'C:/ExperimentCodes/ESIdiff/')
    from utils.fx_bids_local_mi import load_bids, load_sti_info
    import os.path as op
    from utils.utils_train import source_to_sourceEstimate

    subject_dir = r'E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\freesurfer'
    dir_bids = r'E:\2_ESIdiff\Dataset\localize-mi-data' # base directory of the BIDS dataset
    task = 'seegstim' # task name
    subj = 'sub-02' # subject id
    run = 'run-01' # run id
    fwd = mne.read_forward_solution(op.join(dir_bids, 'derivatives', 'sourcemodelling', subj, 'fwd', '%s_fwd.fif' % subj)) # load forward solution
    run_eeg = load_bids(dir_bids, subj, task, run_id=run)
            # run_eeg.crop(t_range[0], t_range[1])  # Crop EEG data to the specified time range
            # Get Ground Truth source from stimulation info
    run_eeg.crop(-0.001, 0.002)  # Crop EEG data to the specified time range
    run_sti = load_sti_info(dir_bids,subj, task, run)
    source_GT = build_source_matrix_multi_trials(fwd, 
                                                run_eeg.times, 
                                                stim_info=run_sti,
                                                biphasic=True, 
                                                sampling_rate=run_eeg.info['sfreq'],    
                                                nbhd_k=1)
    src_GT = source_to_sourceEstimate(source_GT[0,:], fwd, sfreq=run_eeg.info['sfreq'], tmin=-0.001, free_ori=False)

    src_GT.subject = subj  # Set subject name for the source estimate
    
    
    # Visualize the source signal waveform
    import matplotlib.pyplot as plt

    # Create a time array for plotting
    times = np.linspace(-0.001, 0.002, source_GT.shape[2])

    # Find active dipoles (those with non-zero values)
    active_dipoles = np.where(np.any(source_GT[0] != 0, axis=1))[0]

    # Plot the waveforms of active dipoles
    # plt.figure(figsize=(10, 6))
    # for idx in active_dipoles:
    #     plt.plot(times * 1000, source_GT[0, idx], label=f'Dipole {idx}')
    # plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Amplitude (V/m)')
    # plt.title('Source Signal Waveforms')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    mne.viz.set_3d_backend('pyvistaqt')  # 需要安装 PyQt5/PySide2
# ...existing code...
    brain = src_GT.plot(
        initial_time=0.0005,
        hemi="both",
        subjects_dir=subject_dir,
        smoothing_steps=1,
    )
    
    if hasattr(brain, "show"):
        brain.show()  # 阻塞，直到用户关闭窗口
    else:
        input("按回车退出...")  # 兜底防闪退
# ...existing code...
    # 确保使用交互式3D后端，并阻塞直到窗口关闭


