import sys
from pathlib import Path
nb_dir = Path().resolve()
parent_dir = nb_dir.parent
sys.path.insert(0, 'C:/ExperimentCodes/ESIdiff/')
from utils.util_sim import create_n_dim_noise, unpack_fwd, gaussian, repeat_newcol, get_source_diam_from_order
from mne.channels.layout import _find_topomap_coords
import mne
from copy import deepcopy
from tqdm import tqdm
from data.sigGen import  generate_monophasic_pulse, generate_biphasic_pulse
import os
import os.path as op

import numpy as np

from typing import Callable, Sequence, Union, Optional, Dict
from abc import ABC, abstractmethod
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import gc

Number = Union[int, float]
Spec = Union[Number, Sequence[Number], Callable[[float, int], Number]]






DEFAULT_SETTINGS = {
    'random_type': 'range',  # 'range': 在 (min, max) 范围内随机取值; 'choice': 从给定列表中随机选取
    'random_method_S_fireNum': 'standard',
    'random_method_extents': 'uniform',  # 'uniform' | 'reciprocal' | 'weighted'
    'num_firing_sources': (1, 25),
    'extents':  (1, 50),  # in millimeters
    'amplitudes': (1e-3, 10),  # amplitudes come in nAm
    'shapes': 'gaussian', # shape控制源spread的形状，可以选gaussian、flatten或者是混合
    'duration_of_stimulate': 0.5, # 脉冲宽度 ms
    'time_jitter':0.5, # 时间随机偏移的最大范围，在方波里边，单位ms；但是在sin的pulse中是相位单位弧度。
    'signal_time_range': (-0.002, 0.002), # 信号的时间跨度 s
    'eeg_snir': 5,  # 粉红噪声模拟 SNR in dB
    'eeg_snr': (0, 2),
    'beta_noise': 1,
    'source_spread': "mixed",
    'source_time_course': "pulse", # 4种选择 pulse：用的是sin， 双向脉冲 bipulse：用的方波, 单向脉冲 mono_pulse也是方波，和随机random
    'random_seed': 42,
    'sample_frequency': None,
    'Pre_EA': False,
    'Pre_White': False,
    'save_figs': True,
    'fro_normalize_source': True,

}

class SimSource(ABC):
    
    def __init__(self, 
                 # 源模拟参数
                 fwd:mne.Forward,
                 info:mne.Info = None,
                 settings:dict = DEFAULT_SETTINGS,
                 # 保存参数
                 save_path: str = r'E:\2_ESIdiff\Dataset\localize-mi-data\derivatives',
                 subject_id = 'sub-01',
                 save_tag = None):

        self.fwd = deepcopy(fwd)
        info.bads = []
        self.fwd.pick_channels(info['ch_names'])  # 确保前向解通道与 info 匹配
        self.settings = settings
        self.settings['sample_frequency'] = info['sfreq']
        self.info = deepcopy(info)
        self.noise_generator = NoiseGenerator(self.info)
        self.prepare()
        seed = self.settings['random_seed']
        self.rng = np.random.RandomState(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        self.subject = subject_id
        if save_tag is not None:
            self.save_path = op.join(save_path, save_tag, subject_id)
        else:
            self.save_path = op.join(save_path, subject_id)
        self._pathcheck(self.save_path)
    
    def prepare(self):
        if self.settings is None:
            self.settings = DEFAULT_SETTINGS
        self.fwd, self.leadfield, self.source_pos, self.source_tris = unpack_fwd(self.fwd)
        for key in self.settings.keys():
            if not key in DEFAULT_SETTINGS.keys():
                msg = f'key {key} is not part of allowed settings. See DEFAULT_SETTINGS for reference: {DEFAULT_SETTINGS}'
                raise AttributeError(msg)
        for key in DEFAULT_SETTINGS.keys():
            # Check if setting exists and is not None
            if not (key in self.settings.keys() and self.settings[key] is not None):
                self.settings[key] = DEFAULT_SETTINGS[key]
        
        self.neighbors = self.calculate_neighbors()
        self._precompute_diams()
    
    def calculate_neighbors(self):
        adj = mne.spatial_src_adjacency(
            self.fwd["src"], verbose=0
        ).tocsr()
        adj.sort_indices()
        neighbors = np.empty(adj.shape[0], dtype=object)
        for index in range(adj.shape[0]):
            start, stop = adj.indptr[index:index + 2]
            neighbors[index] = adj.indices[start:stop].copy()
        return neighbors
    
    
    @abstractmethod
    def simulate_source(self)->np.ndarray:
        ...
    
    @abstractmethod
    def simulate_run_eeg(self)->np.ndarray:
        ...

    @abstractmethod
    def simulate(self):
        ...


    def source2mneEstimatesource(self, source_data: np.ndarray, sample_rate) -> mne.SourceEstimate:
        """
        将源数据转换为 MNE 的 SourceEstimate 对象
        """
        # 创建一个空的 SourceEstimate 对象
        stc = mne.SourceEstimate(data=source_data, vertices=self.fwd['source_rr'], tmin=self.t_min_s, tstep=1/sample_rate)
        return stc

    

    def save_run_data(self, sim_run_eegs, sim_source, savetag,
                      source_params: Optional[dict] = None,
                      noise_params: Optional[dict] = None,
                      preprocess_eegs: Optional[Dict[str, np.ndarray]] = None):
        save_path = op.join(self.save_path, 'runs_data', savetag)
        self._pathcheck(save_path)
        if len(sim_run_eegs.shape) == 2:
            sim_run_eegs = sim_run_eegs[np.newaxis, :, :]

        source_payload = {'data': sim_source}
        if source_params:
            source_payload.update(source_params)
        np.savez(op.join(save_path, 'sim_source.npz'), **source_payload)

        eeg_payload = {'data': sim_run_eegs}
        if noise_params:
            for k, v in noise_params.items():
                eeg_payload[k] = np.array(v)
        np.savez(op.join(save_path, 'sim_run_eegs.npz'), **eeg_payload)

        if preprocess_eegs:
            for key, value in preprocess_eegs.items():
                fname = f'sim_run_eegs_{key}.npy'
                np.save(op.join(save_path, fname), value)

    def _pathcheck(self, path):
        if not op.exists(path):
            os.makedirs(path)

    def _plot_epochs_figs(self,
                          epochs: mne.Epochs,
                          save_path: str,
                          tag: str,
                          evoked_suffix: str = '_evoked',
                          first_suffix: str = '_firsttrial',
                          save_first_trial: bool = True) -> None:
        """绘制并保存指定 Epochs 的 evoked 以及首个试次曲线。"""
        epochs = epochs.copy()
        epochs.set_eeg_reference('average', projection=True)
        epochs.apply_proj()

        evoked = epochs.average()
        fig_evoked = evoked.plot(show=False)
        fig_evoked.savefig(op.join(save_path, f'{tag}{evoked_suffix}.png'))
        plt.close(fig_evoked)

        if save_first_trial:
            first = epochs[0].average()
            fig_first = first.plot(show=False)
            fig_first.savefig(op.join(save_path, f'{tag}{first_suffix}.png'))
            plt.close(fig_first)

    @staticmethod
    def get_from_range(val, dtype=int):
        ''' If list of two integers/floats is given this method outputs a value in between the two values.
        Otherwise, it returns the value.
        
        Parameters
        ----------
        val : list/tuple/int/float

        Return
        ------
        out : int/float

        '''
        # If input is a function -> call it and return the result
        if callable(val):
            return val()
   

        if dtype==int:
            rng = random.randrange
        elif dtype==float:
            rng = random.uniform
        else:
            msg = f'dtype must be int or float, got {type(dtype)} instead'
            raise AttributeError(msg)

        if isinstance(val, (list, tuple, np.ndarray)):
            if val[0] == val[1]:
                out = dtype(val[0])
            else:
                out = rng(*val)
        else:
            # If input is only a single value anyway, there is no range and it can
            # be returned in the desired dtype.
            out = dtype(val)
        return out

    @staticmethod
    def get_from_choices(val, dtype=int):
        '''从给定的候选值列表中随机选取一个。若输入为单一值则直接返回。

        Parameters
        ----------
        val : list/tuple/np.ndarray/int/float
            候选值列表，或单一值。

        Return
        ------
        out : int/float
        '''
        if callable(val):
            return val()

        if isinstance(val, (list, tuple, np.ndarray)):
            out = dtype(random.choice(val))
        else:
            out = dtype(val)
        return out

    def get_random_param(self, val, dtype=int):
        '''根据 self.settings["random_type"] 分派到对应的随机取值方法。

        - "range" (默认): 从 (min, max) 范围内随机取值 (get_from_range)
        - "choice": 从给定候选列表中随机选取一个 (get_from_choices)
        '''
        random_type = self.settings.get('random_type', 'range')
        if random_type == 'choice':
            return self.get_from_choices(val, dtype=dtype)
        else:
            return self.get_from_range(val, dtype=dtype)

    def compute_distance_matrix_broadcast(self, source_rr):
        """
        使用广播计算源点之间的欧几里得距离矩阵
        """
        diff = source_rr[:, np.newaxis, :] - source_rr[np.newaxis, :, :]  # [n_sources, n_sources, 3]
        distance_matrix = np.linalg.norm(diff, axis=-1)
        return distance_matrix

    def discretize_distance_matrix(self, distance_matrix, sigma = 0.2, k_neighbors=5):
        """
        基于 K-NN 离散化距离矩阵，生成图的边和边权重
        参数：
            -sigma：在把distance转换成边权重的时候使用的高斯核的sigma。单位为mm
                    脑皮层源重建中（欧氏距离，毫米）：
                        5~10mm：局部化很强（小于一个脑回）
                        10~20mm：中等（约一个脑回）
                        20~30mm：范围较广（多个脑回）
        """
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, metric='precomputed')
        nbrs.fit(distance_matrix)
        distances, indices = nbrs.kneighbors(distance_matrix)
        max_dist = distances.max()
        edge_index = []
        edge_attr = []
        for i in range(len(indices)):
            for j in range(1, k_neighbors+1):  # 排除自己
                dist_norm = distances[i, j] / max_dist
                # 高斯核权重: exp(-(d^2)/(2*sigma^2))，距离越小权重越大
                weight = np.exp(-(dist_norm**2) / (2 * (sigma**2)))
                edge_attr.append([weight])
                edge_index.append([i, indices[i, j]])
                
        edge_index = np.array(edge_index).T
        edge_attr = np.array(edge_attr)
        edge_index = np.array(edge_index, dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32)
        return edge_index, edge_attr


class Simulate(SimSource):

    def simulate(self, num_source, num_trials_of_single_source):
        for run_id in tqdm(range(num_source), desc="Simulating sources"):
            sim_source, source_params = self.simulate_source()
            sim_run_eegs, noise_params = self.simulate_run_eeg(
                num_trials=num_trials_of_single_source, sim_source=sim_source)
            savetag = f"run-{run_id:05d}"
            preprocess_eegs = self._preprocess_sim_eegs(sim_run_eegs)
            preprocess_payload = preprocess_eegs if preprocess_eegs else None
            if self.settings.get('save_figs', True):
                self.sim_check_by_evoked_sig_fig(sim_run_eegs, sim_source, savetag, preprocess_payload)
            self.save_run_data(sim_run_eegs, sim_source, savetag,
                               source_params=source_params,
                               noise_params=noise_params,
                               preprocess_eegs=preprocess_payload)
            del sim_source, sim_run_eegs, preprocess_eegs, preprocess_payload
            gc.collect()
        

    def sim_check_by_evoked_sig_fig(self, sim_run_eegs, sim_source,savetag, preprocess_eeg = None):
        # 进行仿真结果的检查和可视化
        clean_eeg = self.project_sources(sim_source)
        info = deepcopy(self.info)

        sim_epos = mne.EpochsArray(sim_run_eegs, info, tmin=self.settings['signal_time_range'][0], verbose=False)
        save_path = op.join(self.save_path,'figs')
        self._pathcheck(save_path)

        # 清洁源投影：仅保存 evoked
        clean_epochs = mne.EpochsArray(clean_eeg[np.newaxis, ...], info,
                                       tmin=self.settings['signal_time_range'][0], verbose=False)
        self._plot_epochs_figs(clean_epochs,
                               save_path,
                               tag=f'{savetag}_clean',
                               save_first_trial=False)

        # 模拟 EEG：保存 evoked 与首个试次
        self._plot_epochs_figs(sim_epos,
                               save_path,
                               tag=f'{savetag}_simEEG',
                               first_suffix='_epoch')

        if preprocess_eeg is not None:
            if isinstance(preprocess_eeg, dict):
                for key, value in preprocess_eeg.items():
                    sim_pre_eeg = mne.EpochsArray(value, info, tmin=self.settings['signal_time_range'][0], verbose=False)
                    self._plot_epochs_figs(sim_pre_eeg,
                                           save_path,
                                           tag=f'{savetag}_{key}_preEEG')
            else:
                sim_pre_eeg = mne.EpochsArray(preprocess_eeg, info, tmin=self.settings['signal_time_range'][0], verbose=False)
                self._plot_epochs_figs(sim_pre_eeg,
                                       save_path,
                                       tag=f'{savetag}_preEEG')



    def simulate_source(self):
        # ----------------------------------------------------------------
        # 决定模拟脑源的数量，使用两种可能的策略：
    
        #     1. 标准模式：
        #     - 当source_number_weighting=False时
        #     - 或num_firing_sources是单个数值（而非范围）时
        #     直接从settings['num_firing_sources']均匀随机采样，或返回固定值
            
        #     2. 倒数加权模式:
        #     - 当需要偏向少量源点时（更符合实际脑电活动特性）
        #     - 使用1/n的权重，使小值（少量源点）有更高的选择概率
        #     - 例如范围[1,10]中，1的权重约为10的10倍
        #  num_of_sources: int



        # ---Step1: 获取源信号配置   --- 
        # ------1.1 选择激活源数量：支持标准/倒数加权/自定义权重 ------
        fire_range = self.settings["num_firing_sources"]
        method = str(self.settings.get("random_method_S_fireNum", "standard")).lower()
        random_type = self.settings.get('random_type', 'range')

        if isinstance(fire_range, (float, int)):
            number_of_sources = int(fire_range)
        else:
            # 根据 random_type 构建候选集合
            if random_type == 'choice':
                population = np.array(fire_range)
            else:
                population = np.arange(*fire_range)  # 左闭右开，与旧逻辑保持一致

            if method == "standard":
                number_of_sources = int(random.choice(population))

            elif method == "reciprocal":
                weights = 1 / population
                weights = weights / weights.sum()
                number_of_sources = int(random.choices(population=population, weights=weights, k=1)[0])

            elif method == "weighted":
                custom_w = self.settings.get("fireNum_weights", None)
                if custom_w is not None and len(custom_w) == len(population):
                    weights = np.asarray(custom_w, dtype=float)
                    if np.all(weights <= 0):  # 全非正则退化为均匀
                        number_of_sources = int(random.choice(population))
                    else:
                        number_of_sources = int(random.choices(population=population, weights=weights, k=1)[0])
                else:
                    # 缺少有效权重时退化为均匀
                    number_of_sources = int(random.choice(population))

            else:
                # 未知模式退化为均匀
                number_of_sources = int(random.choice(population))
        
        # ------1.2 选择源扩散方式 ------
        #           混合方式，为每个激活源随机选择使用region_growing或spherical
        #           普通方式：所有激活源都是source_spread 方式
        # source_spreads: list [源扩散方式]*激活源个数
        if self.settings["source_spread"] == 'mixed':
            source_spreads = [np.random.choice(['region_growing', 'spherical']) for _ in range(number_of_sources)]
        else:
            source_spreads = [self.settings["source_spread"] for _ in range(number_of_sources)]
        # -----------------------------------------------------------------
        # 设定spread范围（支持 uniform / reciprocal / weighted 采样）
        # extents: list [int]*激活源个数
        extent_spec = self.settings['extents']
        extent_method = str(self.settings.get('random_method_extents', 'uniform')).lower()

        if isinstance(extent_spec, (int, float)):
            extents = [int(extent_spec)] * number_of_sources
        elif extent_method == 'uniform':
            extents = [self.get_random_param(extent_spec, dtype=int)
                       for _ in range(number_of_sources)]
        else:
            if random_type == 'choice':
                ext_population = np.asarray(extent_spec)
            else:
                ext_population = np.arange(*extent_spec)

            if extent_method == 'reciprocal':
                w = 1.0 / (ext_population.astype(float) + 1)
                w = w / w.sum()
                extents = random.choices(population=list(ext_population),
                                         weights=list(w), k=number_of_sources)
                extents = [int(e) for e in extents]
            elif extent_method == 'weighted':
                custom_w = self.settings.get('extents_weights', None)
                if custom_w is not None and len(custom_w) == len(ext_population):
                    w = np.asarray(custom_w, dtype=float)
                    if np.all(w <= 0):
                        extents = [int(random.choice(ext_population))
                                   for _ in range(number_of_sources)]
                    else:
                        extents = random.choices(population=list(ext_population),
                                                 weights=list(w), k=number_of_sources)
                        extents = [int(e) for e in extents]
                else:
                    extents = [int(random.choice(ext_population))
                               for _ in range(number_of_sources)]
            else:
                extents = [self.get_random_param(extent_spec, dtype=int)
                           for _ in range(number_of_sources)]
        # -----------------------------------------------------------------
        # ------1.3 设定激活源的形状 ------
        if self.settings['shapes'] == 'mixed':
            shapes = ['gaussian', 'flat']*number_of_sources
            np.random.shuffle(shapes)
            shapes = shapes[:number_of_sources]
            if type(shapes) == str:
                shapes = [shapes]

        elif self.settings['shapes'] == 'gaussian' or self.settings['shapes'] == 'flat':
            shapes = [self.settings['shapes']] * number_of_sources
        # ------1.4 获取激活源的振幅 ------
        amplitudes = [self.get_random_param(self.settings['amplitudes'], dtype=int) * 1e-9 for _ in range(number_of_sources)]
        # -----------------------------------------------------------------
        # ------1.5 获得随机源点位置，和试次持续时间 ------
        src_centers = np.random.choice(np.arange(self.source_pos.shape[0]), \
            number_of_sources, replace=False)
        duration_of_stimulate = self.get_random_param(
            self.settings['duration_of_stimulate'], dtype=float
        )
        # -----------------------------------------------------------------
        # ------1.6 获取源信号波形时间轴 ------
        signal_length = int(round(self.settings['sample_frequency']*(self.settings['signal_time_range'][1]-self.settings['signal_time_range'][0])))
        
        
        signals = []

        # ---2. 生成源信号波形 ---
        # ------2.1 根据source_time_course选择波形生成方式，并生成单位幅值源信号波形 ------
        if self.settings["source_time_course"].lower() == "pulse":
            signals = [self.get_biphasic_pulse(signal_length, temporal_jitter=self.settings['time_jitter']) for _ in range(number_of_sources)]
        
        elif self.settings["source_time_course"].lower() == "bipulse":
                
            times = np.linspace(self.settings['signal_time_range'][0], self.settings['signal_time_range'][1], signal_length)
            signals = [generate_biphasic_pulse(times=times,
                                               amplitude_mA=1e3,
                                               contact_distance_mm=1e3,
                                               pulse_width_ms=duration_of_stimulate,
                                               sampling_rate=self.settings['sample_frequency'],
                                               start_time_ms=0.0,
                                               time_jitter=self.settings['time_jitter']) for _ in range(number_of_sources)]
        elif self.settings["source_time_course"].lower() == "mono_pulse":
            times = np.linspace(self.settings['signal_time_range'][0], self.settings['signal_time_range'][1], signal_length)
            signals = [generate_monophasic_pulse(times=times,
                                               amplitude_mA=1e3,
                                               contact_distance_mm=1e3,
                                               pulse_width_ms=duration_of_stimulate,
                                               sampling_rate=self.settings['sample_frequency'],
                                               start_time_ms=0.0,
                                               time_jitter=self.settings['time_jitter']) for _ in range(number_of_sources)]
        else:
            raise ValueError("source_time_course must be pulse, bipulse or mono_pulse")
                
        
        # sample_frequency = self.settings['sample_frequency']
        # ------2.2 根据激活源数量、形状、扩散范围和振幅生成最终源信号矩阵 ------
        source = np.zeros((self.source_pos.shape[0], signal_length))
        
        for i, (src_center, shape, amplitude, signal, source_spread) in enumerate(zip(src_centers, shapes, amplitudes, signals, source_spreads)):
            
            extent_radius_m = float(extents[i]) / 2.0
            
            # 1. 确定 ROI 索引 (d)
            if source_spread == "region_growing":
                order = self.extents_to_orders(extents[i])
                d = np.array(get_n_order_indices(order, src_center, self.neighbors))
                # 仅计算 ROI 内距离
                dists_roi = np.sqrt(np.sum((self.source_pos[d] - self.source_pos[src_center, :])**2, axis=1))
            else:
                # Spherical: 实时计算当前源到所有源的距离（单行向量，O(n) 内存）
                dists_all = np.linalg.norm(
                    self.source_pos - self.source_pos[src_center], axis=1)
                d = np.where(dists_all < extent_radius_m)[0]
                dists_roi = dists_all[d]
            
            # 2. 生成信号并赋值
            if shape == 'gaussian':
                if len(d) == 0: continue
                
                if len(d) < 2:
                    # 单点直接赋值 (注意 signal 是一维的，numpy 会自动广播)
                    source[d, :] += amplitude * signal
                else:
                    # 3-sigma 规则 (您已正确实现)
                    sd = max(extent_radius_m / 3.0, 1e-4)
                    
                    # [优化] 只计算 ROI 区域的权重
                    weights = gaussian(dists_roi, 0, sd) * amplitude # shape: (N_roi,)
                    
                    # [优化] 外积计算: (N_roi, 1) * (1, T) -> (N_roi, T)
                    # 避免生成全脑大矩阵
                    activity_roi = np.outer(weights, signal)
                    
                    source[d, :] += activity_roi

            elif shape == 'flat':
                if len(d) > 0:
                    # Flat 模式优化
                    # amplitude * signal 是 (T,)
                    # 直接加到 source 的 d 行上，Numpy 会自动把 (T,) 广播到 (N_roi, T)
                    source[d, :] += amplitude * signal
            else:
                 msg = BaseException("shape must be ...")
                 raise(msg)
        

        d = {
            'num_firing_sources': np.int32(number_of_sources),
            'positions': self.source_pos[src_centers],
            'extents': np.array(extents, dtype=np.float32),
            'amplitudes': np.array(amplitudes, dtype=np.float64),
            'shapes': np.array(shapes),
            'source_spreads': np.array(source_spreads),
            'duration_of_stimulate': np.float32(duration_of_stimulate),
        }

        # --- 3. 对源信号进行 Frobenius 归一化 ---
        if self.settings.get('fro_normalize_source', True):
            fro = np.linalg.norm(source, ord='fro')
            if fro > 0:
                source = source / fro
        return source, d
    
    @staticmethod
    def get_biphasic_pulse(pulse_len, center_fraction=1, temporal_jitter=0.):
        ''' Returns a biphasic pulse of given length.
        
        Parameters
        ----------
        x : int
            the number of data points

        '''
        pulse_len = int(pulse_len)
        freq = (1/pulse_len) *center_fraction#/ 2
        time = np.linspace(-pulse_len/2, pulse_len/2, pulse_len)
        
        jitter = np.random.randn()*temporal_jitter
        signal = np.sin(2*np.pi*freq*time + jitter)
        crop_start = int(pulse_len/2 - pulse_len/center_fraction/2)
        crop_stop = int(pulse_len/2 + pulse_len/center_fraction/2)
        
        # signal[(time<-1) | (time>1)] = 0
        signal[:crop_start] = 0
        signal[crop_stop:] = 0
        signal *= np.random.choice([-1,1])
        return signal
    
    def extents_to_orders(self, extents):
        ''' Convert extents (source diameter in ) to neighborhood orders.
        '''
        if self.diams is None:
            self.get_diams_per_order()
        if isinstance(extents, (int, float)):
            order = np.argmin(abs(self.diams-extents))
        else:
            order = (np.argmin(abs(self.diams-extents[0])), np.argmin(abs(self.diams-extents[1])))

        return order
    
    def _precompute_diams(self):
        '''用 cKDTree 计算最近邻距离中位数，推导各阶邻域直径。
        内存占用 O(n) 而非 O(n²)，避免存储完整距离矩阵。
        '''
        from scipy.spatial import cKDTree
        tree = cKDTree(self.source_pos)
        nearest_dists, _ = tree.query(self.source_pos, k=2)
        base_diam = np.median(nearest_dists[:, 1])

        diams = []
        order = 0
        diam = 0
        while diam < 100:
            diam = base_diam * (2 + order)
            diams.append(diam)
            order += 1
        self.diams = np.array(diams)

    def get_diams_per_order(self):
        '''兼容旧调用，内部委托 _precompute_diams。'''
        self._precompute_diams()
    
    def simulate_run_eeg(self, num_trials, sim_source):
        '''
        模拟多个 EEG 试次：先投影源到传感器，再按 trial 加噪并记录实际 SNR。
        sim_source: (n_sources, n_times)
        num_trials: 试次数量
        '''
        # ---1. 获取参数与 SNR 规格 ---
        n_elec = self.leadfield.shape[0]
        n_times = sim_source.shape[1]
        
        snr_spec = self.settings.get('eeg_snr',
                         self.settings.get('target_snr', DEFAULT_SETTINGS['eeg_snr'])) # 尝试获取 eeg_snr，如果没配置它，就试着找 target_snr，如果也没配置，最后就用默认的全局设置。
        
        eeg_snrs = [self.get_random_param(snr_spec, dtype=int) for _ in range(num_trials)]
        betas_noise = [self.get_random_param(self.settings['beta_noise'], dtype=int) for _ in range(num_trials)]
        eeg_snirs = [self.get_random_param(self.settings.get('eeg_snir', DEFAULT_SETTINGS['eeg_snir']), dtype=int) for _ in range(num_trials)]

        # ---2. 生成干净 EEG，并计算范数用于 SNR 估计 ---
        eeg_clean = self.project_sources(sim_source)
        clean_norm = np.linalg.norm(eeg_clean, ord='fro')
        
        # ---3. 按 trial 加噪并估计实际 SNR ---
        eeg_trials_noisy = []
        noise_params_list = []
        for sample in range(num_trials):
            target_snr = eeg_snrs[sample] # 传感器噪声 SNR
            background_noise_type = betas_noise[sample]    # 粉红噪声 Beta
            # A. 生成粉红噪声 (背景脑活动) - 具有空间相关性
            # 使用 NoiseGenerator 生成
            noise_pink = self.noise_generator.get_noise(n_time=n_times, exponent=background_noise_type)
            # B. 生成白噪声 (传感器噪声) - 空间独立
            noise_sensor = np.random.randn(n_elec, n_times)
            
            # C. 归一化并混合
            # 输入：干净信号, 粉红噪声, 白噪声, 目标SNIR, 目标SNR
            noisy_trial, log_info = self.mix_signal_and_noise(
                clean = eeg_clean, 
                norm_clean=clean_norm,
                noise_pink = noise_pink, 
                noise_sensor = noise_sensor, 
                target_snir=eeg_snirs[sample], 
                target_snr=target_snr
            )
            
            eeg_trials_noisy.append(noisy_trial)
            
            # 记录详细日志
            log_info.update({
                'target_snr': target_snr,
                'target_snir': eeg_snirs[sample],
                'beta_noise': background_noise_type
            })
            noise_params_list.append(log_info)

        # ---4. 统一数据格式 ---
        eeg_trials_noisy = np.stack(eeg_trials_noisy, axis=0) # (trials, channels, times)
        
        # ---5. 整理返回的噪声参数 ---
        # 将 list of dict 转换为 dict of lists
        keys = noise_params_list[0].keys()
        noise_params = {k: [d[k] for d in noise_params_list] for k in keys}
        # 为了兼容旧代码的命名，做一些映射
        noise_params['eeg_snrs'] = noise_params['target_snr'] 
        noise_params['betas_noise'] = noise_params['beta_noise']

        return eeg_trials_noisy, noise_params

    def mix_signal_and_noise(self, clean, norm_clean, noise_pink, noise_sensor, target_snir, target_snr):
        """
        核心混合函数：
        1. 对 Clean, Pink, Sensor 进行 Frobenius 归一化
        2. 根据 SNIR 计算 Pink 权重, 根据 SNR 计算 Sensor 权重
        3. 混合: Clean_norm + w_p * Pink_norm + w_s * Sensor_norm
        """
        # 1. 计算范数
        norm_pink = np.linalg.norm(noise_pink, ord='fro')
        norm_sensor = np.linalg.norm(noise_sensor, ord='fro')
        
        # 2. 归一化 (防除零)
        clean_unit = clean / (norm_clean + 1e-12)
        pink_unit = noise_pink / (norm_pink + 1e-12)
        sensor_unit = noise_sensor / (norm_sensor + 1e-12)
        
        # 3. 计算权重系数
        # SNIR = 20 * log10( ||Clean|| / ||Pink_scaled|| ) 
        # 因为已经归一化，||Clean_unit||=1, ||Pink_unit||=1
        # 5dB = 20 * log10( 1 / scale_pink )  =>  scale_pink = 10^(-5/20)
        scale_pink = 10 ** (-target_snir / 20.0)
        
        # SNR = 20 * log10( ||Clean|| / ||Sensor_scaled|| )
        scale_sensor = 10 ** (-target_snr / 20.0)
        
        # 4. 混合
        # 最终信号 = 干净信号(单位) + 背景噪声(缩放) + 传感器噪声(缩放)
        noisy_data = clean_unit + scale_pink * pink_unit + scale_sensor * sensor_unit
        
        
        # 全局 SNR (Signal / All_Noise)
        # 噪声部分 = scale_pink * pink_unit + scale_sensor * sensor_unit
        total_noise = scale_pink * pink_unit + scale_sensor * sensor_unit
        norm_total_noise = np.linalg.norm(total_noise, ord='fro')
        actual_global_snr = 20 * np.log10(1.0 / (norm_total_noise + 1e-12))
        
        log_info = {
            'actual_global_snr': actual_global_snr
        }
        
        return noisy_data, log_info
        
    
    def project_sources(self, sources):
        result = np.matmul(self.leadfield, sources)
        return result

    def _preprocess_sim_eegs(self, sim_run_eegs: np.ndarray) -> Dict[str, np.ndarray]:
        """根据设置返回 ESA/白化后的 EEG 版本。"""
        if sim_run_eegs.ndim == 2:
            eegs = sim_run_eegs[np.newaxis, ...]
        else:
            eegs = sim_run_eegs
        outputs: Dict[str, np.ndarray] = {}
        if self.settings.get('Pre_EA'):
            outputs['preEA'] = align_eeg_euclidean(eegs)
        if self.settings.get('Pre_White'):
            white_trials = []
            for trial in eegs:
                trial_white, _ = whiten_eeg_channels(trial)
                white_trials.append(trial_white)
            outputs['preWhite'] = np.stack(white_trials, axis=0).astype(np.float32)
        return outputs

from mne.channels.layout import _find_topomap_coords
from scipy.spatial import cKDTree # 建议使用 cKDTree 替代手动 argsort，速度更快
class NoiseGenerator:
    ''' Generates multidimensional colored noise automatically adapted to the sensor layout.
    '''

    def __init__(self, info):
        '''
        Parameters
        ----------
        info : mne.Info
            The mne-python Info object.
        '''

        self.info = info
        # 初始化时不传入参数，直接在内部计算
        self.prepare()

    def prepare(self):
        ''' Prepare the regularly spaced grid based on sensor density. '''
        
        # --- 1. 获取电极位置 ---
        # _find_topomap_coords 会自动处理 3D->2D 投影
        self.elec_pos = _find_topomap_coords(self.info, self.info.ch_names, ignore_overlap=True)
        n_channels = self.elec_pos.shape[0]

        # --- 2. 自动计算 Grid 分辨率 ---
        # 逻辑：分辨率应足以覆盖电极密度。
        # 经验公式：Resolution = sqrt(N_channels) * 2
        # 例如 64 通道 -> 16x16 Grid; 256 通道 -> 32x32 Grid
        auto_res = int(np.sqrt(n_channels) * 2)
        self.resolution = max(8, auto_res) # 设定一个下限，比如 8x8

        # --- 3. 自动设定插值邻居数 ---
        # 默认 5，但不能超过网格总点数
        self.k_neighbors = min(5, self.resolution ** 2)

        # --- 4. 构建网格 ---
        # 获取电极覆盖范围，并稍微外扩一点避免边缘伪影
        x_min, x_max = self.elec_pos[:, 0].min(), self.elec_pos[:, 0].max()
        y_min, y_max = self.elec_pos[:, 1].min(), self.elec_pos[:, 1].max()
        margin = (x_max - x_min) * 0.05 # 5% margin
        
        x = np.linspace(x_min - margin, x_max + margin, num=self.resolution)
        y = np.linspace(y_min - margin, y_max + margin, num=self.resolution)

        # 生成网格坐标 (resolution, resolution, 2)
        grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
        self.grid_flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # --- 5. 预计算最近邻索引 (Grid -> Electrodes) ---
        # 我们需要知道：对于每个电极，哪几个网格点离它最近？
        # 使用 cKDTree 加速搜索
        tree = cKDTree(self.grid_flat)
        # query 返回 (distance, index)
        _, self.neighbor_indices = tree.query(self.elec_pos, k=self.k_neighbors)

    def get_noise(self, n_time, exponent=2):
        ''' Create colored noise.
        Parameters
        ----------
        n_time : int
            Number of time points.
        exponent : float
            1/f^beta exponent.
        '''
        
        # 1. 在虚拟网格上生成 3D 噪声 (Space X, Space Y, Time)
        # shape: (resolution, resolution, n_time)
        noise_grid_3d = create_n_dim_noise((self.resolution, self.resolution, n_time), exponent=exponent)

        # 2. 将 3D 网格噪声 Flatten 为 (N_grid_points, n_time)
        # 注意：create_n_dim_noise 生成的是 (res, res, time)
        # 我们需要将其 reshape 为 (res*res, time) 以便进行索引
        noise_grid_flat = noise_grid_3d.reshape(self.resolution**2, n_time)

        # 3. 插值到电极 (Spatial Interpolation)
        # 对于每个电极，取其最近的 k 个网格点的噪声均值
        noise_elec = np.zeros((self.elec_pos.shape[0], n_time))
        
        # 向量化操作优化：
        # self.neighbor_indices shape: (n_elec, k_neighbors)
        # 我们要从 noise_grid_flat 中取出这些行，然后求均值
        
        for e in range(self.elec_pos.shape[0]):
            # 取出当前电极对应的 k 个网格点的噪声数据 (k, n_time)
            neighbors_noise = noise_grid_flat[self.neighbor_indices[e], :]
            # 平均
            noise_elec[e, :] = np.mean(neighbors_noise, axis=0)
            
        return noise_elec


        

def align_eeg_euclidean(eeg: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    按照《Transfer Learning for Brain-Computer Interfaces: A Euclidean Space
    Data Alignment Approach》中提出的 ESA 方法进行欧式空间对齐。

    步骤：
        1. 对每个 trial 计算通道协方差 R_i = X_i X_i^T / (T-1)，并在 trial 间取平均
           \bar{R}。
        2. 计算 \bar{R}^{-1/2} 作为全局白化矩阵 P。
        3. 对每个 trial 执行 X_i' = P X_i，实现跨被试/场景的欧式对齐。

    参数
    ------
    eeg : np.ndarray
        形状为 (n_trials, n_channels, n_times) 或 (n_channels, n_times) 的 EEG 数据。
    eps : float
        特征值正则项，避免奇异协方差矩阵。

    返回
    ------
    np.ndarray
        与输入形状一致的对齐结果。
    """
    if eeg.ndim == 2:
        eeg = eeg[None, ...]
        squeeze_back = True
    elif eeg.ndim == 3:
        squeeze_back = False
    else:
        raise ValueError("eeg 应为二维或三维数组")

    n_trials, n_channels, n_times = eeg.shape
    cov_acc = np.zeros((n_channels, n_channels), dtype=np.float64)
    for trial in eeg:
        trial_centered = trial - trial.mean(axis=1, keepdims=True)
        cov = trial_centered @ trial_centered.T / max(n_times - 1, 1)
        cov_acc += cov
    mean_cov = cov_acc / n_trials

    eigvals, eigvecs = np.linalg.eigh(mean_cov)
    whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T

    aligned = np.empty_like(eeg, dtype=np.float32)
    for idx, trial in enumerate(eeg):
        trial_centered = trial - trial.mean(axis=1, keepdims=True)
        aligned[idx] = (whitener @ trial_centered).astype(np.float32)

    if squeeze_back:
        return aligned[0]
    return aligned


def whiten_eeg_channels(eeg: np.ndarray, eps: float = 1e-6):
    """对白化 EEG 通道，返回白化后的信号以及白化矩阵。"""
    if eeg.ndim != 2:
        raise ValueError("eeg 应为二维数组 (channels, timepoints)")
    eeg = eeg-eeg.mean(axis=1, keepdims=True)
    cov = np.cov(eeg, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    eeg_white = whitener @ eeg
    return eeg_white.astype(np.float32), whitener.astype(np.float32)


def whiten_leadfield_channels(leadfield: np.ndarray, eps: float = 1e-6):
    """对白化 L 矩阵的通道（行）以匹配 EEG 白化操作。"""
    if leadfield.ndim != 2:
        raise ValueError("leadfield 应为二维数组 (channels, sources)")
    cov = np.cov(leadfield, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    L_white = whitener @ leadfield
    return L_white.astype(np.float32), whitener.astype(np.float32)


def generate_gaussian_sinusoid(times: np.ndarray,
                               frequency_hz: float,
                               center_time: float,
                               width: float,
                               phase: float = 0.0,
                               amplitude: float = 1.0):
    """生成 s(t)=A·sin(2πft+φ)·exp(-((t-τ)/ω)^2) 形式的高斯调制正弦波。"""
    times = np.asarray(times, dtype=np.float32)
    sinusoid = np.sin(2 * np.pi * frequency_hz * times + phase)
    gaussian_envelope = np.exp(-((times - center_time) / width) ** 2)
    signal = amplitude * sinusoid * gaussian_envelope
    return signal.astype(np.float32)


def get_n_order_indices(order, pick_idx, neighbors):
    ''' Iteratively performs region growing by selecting neighbors of 
    neighbors for <order> iterations.
    '''
    assert order == round(order), "Neighborhood order must be a whole number"
    order = int(order)
    if order == 0:
        return [pick_idx,]
    flatten = lambda t: [item for sublist in t for item in sublist]
    # print("y")
    current_indices = [pick_idx,]
    for cnt in range(order):
        # current_indices = list(np.array( current_indices ).flatten())
        new_indices = [neighbors[i] for i in current_indices]
        new_indices = flatten( new_indices )
        current_indices.extend(new_indices)
        
        current_indices = list(set(current_indices))
    return current_indices


