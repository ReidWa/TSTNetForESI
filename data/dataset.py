import sys
from pathlib import Path
nb_dir = Path().resolve()
parent_dir = nb_dir.parent
sys.path.insert(0, 'C:/ExperimentCodes/ESIdiff/')
# import io
import mne
import pandas as pd
import mne

from utils.fx_bids_local_mi import load_bids, load_sti_info
from utils.utils_train import load_forward_fixed
from data.sigGen import build_source_matrix_multi_trials
import os.path as op
import re
import numpy as np
from tqdm import tqdm 
from einops import rearrange
from tqdm import tqdm
from glob import glob
import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
# import lmdb
# import pickle
from tqdm import tqdm

from functools import reduce

def list_run_dirs(base_path: str, digits: int = 5):
    """Return sorted list of subdirectories named like run-xxxxx under base_path."""
    pattern = rf"run-\d{{{digits}}}"
    candidates = glob(op.join(base_path, "run-*"))
    run_dirs = [p for p in candidates if op.isdir(p) and re.fullmatch(pattern, op.basename(p))]
    return sorted(run_dirs)

def euclidAlign(X):
    R = np.zeros((X.shape[1], X.shape[1]))
    R = reduce(lambda R, y: R+np.dot(y, y.T), X, R)
    R /= X.shape[0]
    for i in range(X.shape[0]):
        X[i, :, :] = np.dot(fuduiban(R), X[i, :, :])
    return X
def fuduiban(R):
    v, Q = np.linalg.eig(R)
    ss = np.diag(v**(-0.5))
    ss[np.isinf(ss)] = 0
    re = np.dot(Q, np.dot(ss, np.linalg.inv(Q)))
    return np.real(re)

# GPU加速的 ZCA 白化和 PCA 白化

def zca_whiten_torch(X, epsilon=1e-5):
    """
    Perform ZCA whitening on the input EEG data matrix using PyTorch (can be GPU accelerated).
    
    X: (n_channels, n_samples) - Input EEG data as a PyTorch tensor.
    epsilon: Regularization term to prevent division by zero.
    
    Returns:
    Whitened signal as a PyTorch tensor.
    """
    # Center the data
    X -= X.mean(dim=1, keepdim=True)
    
    # Compute the covariance matrix
    cov_matrix = torch.cov(X)
    
    # Eigenvalue decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    
    # Clip small eigenvalues to avoid numerical issues
    eigvals = torch.clamp(eigvals, min=epsilon)
    
    # Whitening transformation matrix (ZCA)
    whitening_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    
    # Apply ZCA whitening
    X_whitened = whitening_matrix @ X
    
    return X_whitened



def pca_whiten_torch(X, epsilon=1e-5):
    """
    Perform PCA whitening on the input EEG data matrix using PyTorch (can be GPU accelerated).
    
    X: (n_channels, n_samples) - Input EEG data as a PyTorch tensor.
    epsilon: Regularization term to prevent division by zero.
    
    Returns:
    Whitened signal as a PyTorch tensor.
    """
    # Center the data
    X -= X.mean(dim=1, keepdim=True)

    # Compute the covariance matrix
    cov_matrix = torch.cov(X)
    
    # Eigenvalue decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    
    # Clip small eigenvalues to avoid numerical issues
    eigvals = torch.clamp(eigvals, min=epsilon)
    
    # Whitening transformation matrix
    whitening_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    
    # Apply whitening
    X_whitened = whitening_matrix @ X
    
    return X_whitened

# 数据集：对应模拟数据集
def pack_runs(root_path, subid, h5_path):
    run_dirs = sorted(glob(op.join(root_path, subid, "runs_data", "run-*")))
    with h5py.File(h5_path, "w") as f:
        src_ds = None
        eeg_ds = None
        for run_idx, run_dir in enumerate(run_dirs):
            src = np.load(op.join(run_dir, "sim_source.npz"))['data']
            eeg = np.load(op.join(run_dir, "sim_run_eegs.npz"))['data']
            if src_ds is None:
                src_ds = f.create_dataset("source", shape=(len(run_dirs),) + src.shape,
                                          dtype=src.dtype, compression="gzip")
                eeg_ds = f.create_dataset("eeg", shape=(len(run_dirs),) + eeg.shape,
                                          dtype=eeg.dtype, compression="gzip")
            src_ds[run_idx] = src
            eeg_ds[run_idx] = eeg

# def get_trainsim_testsim_dataset_erp(data_path, train_rate, num_runs, rundigits = 2, seed=42, PreProcess_method = None):
#     pass

# class datasetsim_erp(Dataset):
#     def __init__(self,data_path, PreProcess_method = None):
#         run_eeg_file_name = f'sim_run_eegs_pre{PreProcess_method}.npy' if PreProcess_method else 'sim_run_eegs.npy'
#         eeg_file = os.path.join(data_path, run_eeg_file_name)
#         eeg = np.load(eeg_file).astype(np.float32)
#         source_file = os.path.join(data_path, 'sim_source.npy')
#         source = np.load(source_file).astype(np.float32)
#         self.eps = 1e-6
#         self.source = source/(source.std(axis = (-1, -2),keepdims=True) + self.eps)  # 对 source 信号做通道与时间最大值归一化，降维为 (n_trials, 1, 1)
#         self.eeg =(eeg- eeg.mean(axis = (-1, -2),keepdims=True))/(eeg.std(axis = (-1, -2),keepdims=True) + self.eps)  # 对 EEG 信号做通道与时间平均，降维为 (n_trials, 1, 1)



    

def get_trainsim_testsim_dataset(
    data_path,
    train_rate,
    num_runs,
    rundigits=2,
    seed=42,
    PreProcess_method=None,
):
    """构建模拟数据 train/test ``ConcatDataset``（仅 ``source`` + ``eeg``，无物理标签）。"""
    run_ids = np.arange(num_runs)
    shuffle_rng = np.random.default_rng(seed=seed)
    shuffle_rng.shuffle(run_ids)
    split_idx = int(len(run_ids) * train_rate)
    train_ids = run_ids[:split_idx]
    test_ids = run_ids[split_idx:]
    train_set_list = []
    test_set_list = []
    assert PreProcess_method in [None, 'EA', 'White'], "PreProcess_method must be one of None, 'EA', or 'White'"
    for run_id in tqdm(train_ids, desc="Loading train runs"):
        data_run_path = op.join(data_path, f'run-{run_id:0{rundigits}d}')
        train_set_list.append(datasetsim_singlerun(data_run_path, PreProcess_method=PreProcess_method))

    for run_id in tqdm(test_ids, desc="Loading test runs"):
        data_run_path = op.join(data_path, f'run-{run_id:0{rundigits}d}')
        test_set_list.append(datasetsim_singlerun(data_run_path))
    train_dataset = ConcatDataset(train_set_list)
    test_dataset = ConcatDataset(test_set_list)
    return train_dataset, test_dataset


class datasetsim_singlerun(Dataset):
    def __init__(self, data_path, PreProcess_method = None, evoked = False):
        
        if PreProcess_method:
            eeg_file = os.path.join(data_path, f'sim_run_eegs_pre{PreProcess_method}.npy')
            eeg = np.load(eeg_file).astype(np.float32)
        else:
            eeg = np.load(os.path.join(data_path, 'sim_run_eegs.npz'))['data'].astype(np.float32)
        source = np.load(os.path.join(data_path, 'sim_source.npz'))['data'].astype(np.float32)
        self.eps = 1e-6
        
        # x_nz = source[source != 0].astype(np.float32)
        # if x_nz.size == 0:
        #     scale = 1.0
        # else:
        #     scale = float(np.sqrt(np.mean(x_nz * x_nz) + self.eps))
        #     if scale < self.eps:
        #         scale = 1.0
        # self.source_scale = np.float32(scale)
        # self.source = (source / scale).astype(np.float32)
        # self.source = source/(source.std(axis = (-1, -2),keepdims=True)+self.eps)

        # self.eeg =(eeg- eeg.mean(axis = (-1, -2),keepdims=True))/(eeg.std(axis = (-1, -2),keepdims=True) + self.eps)  # 对 EEG 信号做通道与时间平均，降维为 (n_trials, 1, 1)
        self.source = _normalize_fro_np(source, axis=(-1, -2), eps=self.eps)
        self.eeg = _normalize_fro_np(eeg, axis=(-1, -2), eps=self.eps)
        if evoked is True:
            self.eeg = self.eeg.mean(axis=0, keepdims=True)  # 计算平均 ERP，降维为 (1, n_channels, n_times)
        # 全 run 共享同一份 source，避免每个 sample 重复 numpy→torch 拷贝
        self._source_tensor = torch.from_numpy(self.source).contiguous().float()

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx]).float()
        return self._source_tensor, eeg


def _load_physics_labels_npz(label_npz: str, n_eeg_trials: int, evoked: bool):
    """读取 ``labels.npz``，返回与 EEG trial 对齐的 ``noise_ratio`` / ``sparsity`` (1d float32)。"""
    if not os.path.isfile(label_npz):
        raise FileNotFoundError(
            f'需要物理标签但未找到 {label_npz}，请先运行 00-2-sim_03_generate_labels.py'
        )
    lz = np.load(label_npz, allow_pickle=True)
    try:
        nr = np.asarray(lz['noise_ratio'], dtype=np.float32).reshape(-1)
        sp = np.asarray(lz['sparsity'], dtype=np.float32).reshape(-1)
    finally:
        lz.close()
    if nr.shape != sp.shape:
        raise ValueError(f'{label_npz} 中 noise_ratio 与 sparsity 长度不一致: {nr.shape} vs {sp.shape}')
    if nr.shape[0] == n_eeg_trials:
        return nr, sp
    if evoked and n_eeg_trials == 1 and nr.shape[0] > 1:
        return (
            np.array([float(nr.mean())], dtype=np.float32),
            np.array([float(sp.mean())], dtype=np.float32),
        )
    raise ValueError(
        f'{label_npz} 标签条数应与 EEG trial 数一致：labels {nr.shape[0]} vs eeg {n_eeg_trials}'
    )


def _normalize_fro_np(x: np.ndarray, axis, eps: float = 1e-6) -> np.ndarray:
    """``x / (||x||_F + eps)``，在 ``axis`` 所指维度上求 Frobenius 范数（平方和再开方）。"""
    n = np.sqrt(np.sum(x * x, axis=axis, keepdims=True, dtype=np.float64)).astype(x.dtype)
    return (x / (n + eps)).astype(np.float32, copy=False)


class datasetsim_singlerun_physics(Dataset):
    """第二版：源/EEG 在最后一维与次末维上 **仅除以 Frobenius 范数**（不减均值）；**必须** 提供 ``labels.npz``。

    - ``source`` / ``eeg``：均为 ``x / (||x||_F + eps)``，归约轴 ``(-1,-2)``。

    ``__getitem__`` 固定返回 ``(source, eeg, noise_ratio, sparsity)``，供
    ``ISTANet_cb_EEGsubnetEmb_FilmV2`` + ``Experiment.trainer`` 使用。
    """

    def __init__(self, data_path, PreProcess_method=None, evoked=False):
        if PreProcess_method:
            eeg_file = os.path.join(data_path, f'sim_run_eegs_pre{PreProcess_method}.npy')
            eeg = np.load(eeg_file).astype(np.float32)
        else:
            eeg = np.load(os.path.join(data_path, 'sim_run_eegs.npz'))['data'].astype(np.float32)
        source = np.load(os.path.join(data_path, 'sim_source.npz'))['data'].astype(np.float32)
        self.eps = 1e-6
        self.source = _normalize_fro_np(source, axis=(-1, -2), eps=self.eps)
        self.eeg = _normalize_fro_np(eeg, axis=(-1, -2), eps=self.eps)
        if evoked is True:
            self.eeg = self.eeg.mean(axis=0, keepdims=True)

        label_npz = os.path.join(data_path, 'labels.npz')
        n_eeg = len(self.eeg)
        self._noise_ratio, self._sparsity = _load_physics_labels_npz(
            label_npz, n_eeg, evoked=evoked
        )
        self._source_tensor = torch.from_numpy(self.source).contiguous().float()

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx]).float()
        nr = torch.tensor(self._noise_ratio[idx], dtype=torch.float32)
        sp = torch.tensor(self._sparsity[idx], dtype=torch.float32)
        return self._source_tensor, eeg, nr, sp


def get_trainsim_testsim_dataset_physics(
    data_path,
    train_rate,
    num_runs,
    rundigits=2,
    seed=42,
    PreProcess_method=None,
):
    """第二版：构建带 ``labels.npz`` 的 train/test ``ConcatDataset``（每样本 4 元组）。"""
    run_ids = np.arange(num_runs)
    shuffle_rng = np.random.default_rng(seed=seed)
    shuffle_rng.shuffle(run_ids)
    split_idx = int(len(run_ids) * train_rate)
    train_ids = run_ids[:split_idx]
    test_ids = run_ids[split_idx:]
    train_set_list = []
    test_set_list = []
    assert PreProcess_method in [None, 'EA', 'White'], "PreProcess_method must be one of None, 'EA', or 'White'"
    for run_id in tqdm(train_ids, desc="Loading train runs (physics labels)"):
        data_run_path = op.join(data_path, f'run-{run_id:0{rundigits}d}')
        train_set_list.append(
            datasetsim_singlerun_physics(data_run_path, PreProcess_method=PreProcess_method)
        )
    for run_id in tqdm(test_ids, desc="Loading test runs (physics labels)"):
        data_run_path = op.join(data_path, f'run-{run_id:0{rundigits}d}')
        test_set_list.append(datasetsim_singlerun_physics(data_run_path, PreProcess_method=None))
    train_dataset = ConcatDataset(train_set_list)
    test_dataset = ConcatDataset(test_set_list)
    return train_dataset, test_dataset


def get_test_sim_dataset(test_data_path, evoked, test_name, PreProcess_method = None):
    data_run_path = op.join(test_data_path, test_name, 'sub-01', 'runs_data')
    runs = list_run_dirs(data_run_path, digits=5)
    data_sets = []
    for run_dir in runs:
        dataset_single = datasetsim_singlerun(run_dir, PreProcess_method=PreProcess_method, evoked=evoked)
        data_sets.append(dataset_single)
    test_set = ConcatDataset(data_sets)
    return test_set


def get_test_sim_dataset_physics(test_data_path, evoked, test_name, PreProcess_method=None):
    """第二版：测试集 ConcatDataset，每样本含 ``labels.npz`` 物理标签。"""
    data_run_path = op.join(test_data_path, test_name, 'sub-01', 'runs_data')
    runs = list_run_dirs(data_run_path, digits=5)
    data_sets = []
    for run_dir in runs:
        data_sets.append(
            datasetsim_singlerun_physics(
                run_dir, PreProcess_method=PreProcess_method, evoked=evoked
            )
        )
    return ConcatDataset(data_sets)


def get_trainsim_testreal_dataset(sim_datapath, num_runs, realdatapath): 
    run_ids = np.arange(num_runs)
    train_set_list = []
    for run_id in tqdm(run_ids, desc="Loading train runs"):
        data_run_path = op.join(sim_datapath, f'run-{run_id:02d}')
        train_set_list.append(datasetsim_singlerun(data_run_path))
    train_set = ConcatDataset(train_set_list)
    test_set = dataset_real_singlesub(realdatapath)
    return train_set, test_set


def get_test_real_dataset(real_datapath: str):
    """构建单被试真实 EEG 测试数据集。

    Parameters
    ----------
    real_datapath : str
        单个被试的数据目录，包含 real_eeg_data.npy 和 real_source_labels.npy。

    Returns
    -------
    dataset_real_singlesub
    """
    return dataset_real_singlesub(real_datapath)


class dataset_real_singlesub(Dataset):
    def __init__(self, real_datapath):
        self.eeg = np.load(op.join(real_datapath, 'real_eeg_data.npy')).astype(np.float32)  
        label_path = op.join(real_datapath, 'real_source_labels.npy')
        label_obj = np.load(label_path, allow_pickle=True)
        labels_dict = label_obj.item()
        self.source_locs = labels_dict['source_locs'].astype(np.float32)
        self.eeg = self._standardize_trials(self.eeg)
    
    @staticmethod
    def _standardize_trials(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = arr.mean(axis=-1, keepdims=True)
        std = arr.std(axis=-1, keepdims=True)
        std = np.where(std < eps, eps, std)
        return (arr - mean) / std
    
    def __len__(self):
        return len(self.eeg)
    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg[idx]).float()
        source_locs = torch.from_numpy(self.source_locs[idx]).float()
        return eeg, source_locs
    
# class DatasetSimLMDB(Dataset):
#     def __init__(self, lmdb_path, cache_source=True):
#         self.lmdb_path = lmdb_path
#         self.cache_source = cache_source
#         self.source_cache = {}
#         env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=True)
#         with env.begin(write=False) as txn:
#             meta = pickle.loads(txn.get(b"meta"))
#             self.num_trials = int(meta["num_trials"])
#             self.dtype = np.dtype(meta["dtype"])
#             self.trial_run_idx = pickle.loads(txn.get(b"trial_run_idx"))
#             self.trial_pos = pickle.loads(txn.get(b"trial_pos"))
#         env.close()

#     def __len__(self):
#         return self.num_trials

#     def _open_env(self):
#         return lmdb.open(
#             self.lmdb_path,
#             subdir=False,
#             readonly=True,
#             lock=False,
#             readahead=True,
#             meminit=False,
#         )

#     def _get_source(self, txn, run_idx):
#         if self.cache_source and run_idx in self.source_cache:
#             return self.source_cache[run_idx]
#         buf = txn.get(f"source/{run_idx:05d}".encode())
#         source = torch.from_numpy(np.load(io.BytesIO(buf))).float()
#         if self.cache_source:
#             self.source_cache[run_idx] = source
#         return source

#     def __getitem__(self, idx):
#         env = self._open_env()
#         try:
#             with env.begin(write=False) as txn:
#                 run_idx = int(self.trial_run_idx[idx])
#                 eeg_buf = txn.get(f"eeg/{idx:08d}".encode())
#                 eeg = torch.from_numpy(np.load(io.BytesIO(eeg_buf))).float()
#                 source = self._get_source(txn, run_idx)
#         finally:
#             env.close()
#         return source, eeg
        


# 数据集A：对应 localize-mi 数据集
class DatasetA_SingSub(Dataset):
    def __init__(self, subj, dir_bids, estimate_method):
        data_path = op.join(dir_bids, 'derivatives', 'dataforDiffusion', estimate_method,subj)
        self.estimated_source_normed = np.load(op.join(data_path, 'estimated_source_normed.npy'))  # shape: [N, n_dipoles, T]
        self.source_GT_normed = np.load(op.join(data_path, 'source_GT_normed.npy'))
        self.edge_index = np.load(op.join(data_path, 'edge_index.npy'))
        self.edge_attr = np.load(op.join(data_path, 'edge_attr.npy'))

    def __len__(self):
        return len(self.estimated_source_normed)
    def __getitem__(self, idx):
        estimated_source = torch.from_numpy(self.estimated_source_normed[idx,:,:,:24]).float()  # [n_dipoles, T]
        source_GT = torch.from_numpy(self.source_GT_normed[idx,:,:,:24]).float()  # [n_dipoles, T]
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(self.edge_attr, dtype=torch.float)
        return estimated_source, source_GT, edge_index, edge_attr


class DatasetA_SingSub_with_preprocess(Dataset):
    def __init__(self, subj, dir_bids, task, t_range=(-0.001, 0.002), for_low_memory=False, k_neighbors=5):
        """
        Initialize the dataset for a single subject.
        Parameters:
        - bids dataset: 
            - subj: str, subject identifier.
            - dir_bids: str, path to the BIDS dataset directory.
            - task: str, task name.
        - t_range: tuple, time range for the EEG data (start, end) in seconds.
        - for_low_memory: bool, whether to use low memory mode for source estimation.
        - k_neighbors: int, number of neighbors to use for distance matrix discretization.
        """
        self.fwd = load_forward_fixed(op.join(dir_bids, 'derivatives', 'sourcemodelling', subj, 'fwd', '%s_fwd.fif' % subj)) # load forward solution
        self.seeg_ch_info = pd.read_csv(op.join(dir_bids, 'derivatives', 'epochs', subj, 'ieeg', '%s_task-%s_space-surface_electrodes.tsv' % (subj, task)), sep='\t') # load SEEG channel info
        runids = sorted(set(int(match) for match in re.findall(r'run-(\d+)', open(op.join(dir_bids, 'derivatives/epochs/', subj, subj+'_scans.tsv')).read())))
        runs = ['run-%02d' % runid for runid in runids]  # format run ids to match BIDS naming convention
        runs = ['run-01']
        
        self.t_range = t_range
        normalized_estimated_source = []
        normalized_source_GT = []

        for run in tqdm(runs, desc="Processing runs"):
            # get EEG data for each run
            run_eeg = load_bids(dir_bids, subj, task, run_id=run)
            # run_eeg.crop(t_range[0], t_range[1])  # Crop EEG data to the specified time range
            # Get Ground Truth source from stimulation info
            run_sti = load_sti_info(dir_bids,subj, task, run)
            source_GT = build_source_matrix_multi_trials(self.fwd['source_nn'], 
                                                         run_sti, 
                                                         True, 
                                                         run_eeg.info['sfreq'],
                                                         t_min=t_range[0], 
                                                         t_max=t_range[1])
            
            
            est_sor = self.estimate_run(run_eeg, for_low_memory, snr=1.0, method='eLORETA')

            for i in range(len(run_eeg)):
            # for i in range(4):
                if for_low_memory:
                    stc = next(est_sor)
                else:
                    stc = est_sor[i]
                data = stc.data  # shape: [n_sources*3, n_times] or [n_sources, n_times]
            # 按 trial 做 min-max 归一化
                data_min = data.min()
                data_max = data.max()
                norm_data = (data - data_min) / (data_max - data_min + 1e-8)
                norm_data = rearrange(norm_data, 'n o t -> (n o) t', o=3)
                norm_data = norm_data[np.newaxis, :, :]  # 添加 batch 维度

                # 构造新的 VectorSourceEstimate，保持 mne 类型
                

                # source_GT 也按 trial 归一化
                gt = source_GT[i]
                
                gt_min = gt.min()
                gt_max = gt.max()
                norm_gt = (gt - gt_min) / (gt_max - gt_min + 1e-8)
                norm_gt = norm_gt[np.newaxis, :, :]
                
                min_T = int(np.min([norm_data.shape[-1], norm_gt.shape[-1]]))
                
                
                norm_gt = norm_gt[:, :, :min_T]
                norm_data = norm_data[:, :, :min_T]
                assert norm_gt.shape[-1] == norm_data.shape[-1], "EEG data and source GT must have the same time points"
                normalized_estimated_source.append(norm_data.astype(np.float32))
                normalized_source_GT.append(norm_gt.astype(np.float32))


        self.estimated_source = np.stack(normalized_estimated_source, axis=0)
        self.source_GT = np.stack(normalized_source_GT, axis=0)
        self.estimated_source = torch.from_numpy(self.estimated_source)
        self.source_GT = torch.from_numpy(self.source_GT)   


        distance_matrix = self.compute_distance_matrix_broadcast(self.fwd['source_rr'])
        self.edge_index, self.edge_attr = self.discretize_distance_matrix(distance_matrix, k_neighbors=k_neighbors)

    def __len__(self):
        return len(self.estimated_source)
    
    def __getitem__(self, idx):
        estimated_source = self.estimated_source[idx]  # [n_dipoles, T]

        source_GT = self.source_GT[idx]  # [n_dipoles, T]
        return estimated_source, source_GT, self.edge_index, self.edge_attr

    def estimate_run(self,epo, for_low_memory=False, snr=1.0, method='eLORETA'):
        lambda2 = 1.0 / snr ** 2
        epo = epo.set_eeg_reference('average', projection=True)
        epo.apply_proj()
        cov = mne.compute_covariance(epo, method='auto', tmin=epo.times[0], tmax=0, verbose=False)
        epo.crop(tmin=self.t_range[0], tmax=self.t_range[1])  # Crop EEG data to the specified time range
        inv = mne.minimum_norm.make_inverse_operator(epo.info, self.fwd, cov, loose=1, depth=0.1, verbose=False)
        stc_vec_list = mne.minimum_norm.apply_inverse_epochs(epo, 
                                                             inv, 
                                                             method=method, 
                                                             lambda2=lambda2, 
                                                             pick_ori='vector',  
                                                             return_generator=for_low_memory, 
                                                             verbose=False)
        
        return stc_vec_list
    
    def compute_distance_matrix_broadcast(self, source_rr):
        """
        使用广播计算源点之间的欧几里得距离矩阵
        """
        # 计算源点之间的差异
        diff = source_rr[:, np.newaxis, :] - source_rr[np.newaxis, :, :]  # [n_sources, n_sources, 3]
        
        # 计算欧几里得距离
        distance_matrix = np.linalg.norm(diff, axis=-1)  # 沿着最后一个轴计算L2范数（即距离）
        
        return  distance_matrix / distance_matrix.max()
    
    def discretize_distance_matrix(self, distance_matrix, k_neighbors=5):
        """
        基于 K-NN 离散化距离矩阵，生成图的边和边权重
        """
        from sklearn.neighbors import NearestNeighbors
        
        # 使用 K-NN 来找到每个源点的 k 个最近邻
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='precomputed')
        nbrs.fit(distance_matrix)
        distances, indices = nbrs.kneighbors(distance_matrix)
        
        edge_index = []
        edge_attr = []
        
        for i in range(len(indices)):
            for j in range(1, k_neighbors):  # 排除自己
                edge_index.append([i, indices[i, j]])  # 连接节点 i 和第 j 个邻居
                edge_attr.append([distances[i, j]])  # 邻居之间的距离作为边的权重
        
        edge_index = np.array(edge_index).T  # 转换为 [2, E] 形式
        edge_attr = np.array(edge_attr)  # 边的权重，形状为 [E, 1]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return edge_index, edge_attr

# 数据集B：对应 osfstorage 数据集
# class DatasetB:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         # 初始化数据集B的相关参数和加载数据

#     def load_data(self):
#         # 实现加载数据的逻辑
#         pass

#     def preprocess(self):
#         # 实现数据预处理的逻辑
#         pass

# if __name__ == "__main__":
#     dir_bids = 'C:/Dataset/localize-mi-data/' # base directory of the BIDS dataset
#     task = 'seegstim' # task name
#     subj = 'sub-02' # subject id
#     dataset = DatasetA_SingSub(subj=subj,dir_bids=dir_bids, task=task, t_range=(-0.002, 0.002), for_low_memory=True, k_neighbors=5)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#     # 取一个 batch
#     for batch in dataloader:
#         estimated_source, source_GT, edge_index, edge_attr = batch
#         # 计算 estimated_source 占用空间（字节）
#         est_bytes = estimated_source.element_size() * estimated_source.nelement()
#         gt_bytes = source_GT.element_size() * source_GT.nelement()
#         print(f"estimated_source shape: {estimated_source.shape}, 占用空间: {est_bytes/1024/1024:.2f} MB")
#         print(f"source_GT shape: {source_GT.shape}, 占用空间: {gt_bytes/1024/1024:.2f} MB")
#         break  # 只看第一个 batch
#     # print("Total samples in dataset:", len(dataset))
#     # print("Distance matrix shape:", distance_matrix['edge_index'].shape, distance_matrix['edge_attr'].shape)
#     # # 取单个样本
#     sample = dataset[0]
#     print("Single sample estimated_source shape:", sample[0].shape)
#     print("Single sample source_GT shape:", sample[1].shape)
