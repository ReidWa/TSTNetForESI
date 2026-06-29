import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import datasetsim_singlerun, list_run_dirs
from Experiment.evaluator_torch import EvaluatorTorch as Evaluator
from model.ISTANet.ori_ISTANetwithSVDgrad import ISTA as TSTNet
from utils.simtest_result_export import export_simtest_results
from utils.utils_train import load_forward_fixed


SUBJECTS = ['sub-01']
TEST_SCRIPTS = {
    'extents': ['0mm', '10mm', '20mm', '30mm'],
    'firingsource': [1, 2, 3, 4],
    'snr': ['minus5dB', '0dB', '5dB', '10dB'],
}
CFG = {
    'method': 'TSTNet-Medium',
    'dir_bids': r'E:\2_ESIdiff\Dataset\localize-mi-data',
    'test_data_path': r'E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\simulationV3\testset',
    'test_preprocess': None,
    'evoked': False,
    'test_batch_size': 32,
    'num_workers': 0,
    'subjects': SUBJECTS,
    'm_matrix_path': r'E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\M_matrix.pt',
    'checkpoint_path': (
        r"E:\2_ESIdiff\实验数据\results-模拟测试集-sub01\TSTNet-High\ori_ISTANet_withSVDgradV0_420260609_104308\model-best.pt"
    ),
    'save_dir': r'E:\2_ESIdiff\实验数据\results-模拟测试集-sub01\TSTNet-High\model-best',
    # Hyperparameters must match model-best.pt tensor shapes:
    # Phi.fc.0.weight -> [Phi_1_dim, n_source, 1]
    # Phi.fc.2.weight -> [Phi_2_dim, Phi_1_dim, 1]
    'ISTANet_layers': 1,
    'alpha_loss': 0.1,
    'beta_loss': 0.1,
    'truned_num': 64,
    'grad_step_alpha': 0.1,
    'Phi_1_dim': 2048,
    'Phi_2_dim': 1024,
    'eval_count_source_power_greater_than': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype': torch.float32,
}


def normalize_L(L, verbose=False):
    n_channels, _ = L.shape
    L = L * (np.sqrt(n_channels) / np.linalg.norm(L, ord='fro'))
    if verbose:
        print('[L Normalization Info]')
        print(f'  Input Shape: {L.shape[0]} channels x {L.shape[1]} sources')
        print(f'  Actual ||L||_F: {np.linalg.norm(L, ord="fro"):.4f}')
    return L


def torch_load_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if 'Weights only load failed' not in str(exc):
            raise
        print('[torch.load] PyTorch weights_only compatibility fallback: weights_only=False')
        return torch.load(path, map_location=map_location, weights_only=False)


def load_M_matrix(path, device):
    payload = torch_load_compat(path, map_location='cpu')
    M = payload['M_matrix'] if isinstance(payload, dict) else payload
    if not torch.is_tensor(M):
        M = torch.tensor(M, dtype=torch.float32)
    return M.to(device=device, dtype=torch.float32)


def load_model(cfg, lead_tensor, M_matrix, cor_sources_tensor):
    model = TSTNet(
        L=lead_tensor,
        LayerNo=cfg['ISTANet_layers'],
        n_source=lead_tensor.shape[1],
        M_matrix=M_matrix,
        cor_sources=cor_sources_tensor,
        truned_channels=cfg['truned_num'],
        alpha_loss=cfg['alpha_loss'],
        beta_loss=cfg['beta_loss'],
        grad_step_alpha=cfg['grad_step_alpha'],
        Phi_1_dim=cfg['Phi_1_dim'],
        Phi_2_dim=cfg['Phi_2_dim'],
    )
    checkpoint = torch_load_compat(cfg['checkpoint_path'], map_location=cfg['device'])
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device=cfg['device'], dtype=cfg['dtype'])
    model.eval()
    return model


def load_run_items(root, test_name, subj_id, preprocess=None, evoked=False):
    runs_root = os.path.join(root, test_name, subj_id, 'runs_data')
    run_dirs = list_run_dirs(runs_root, digits=5)
    return [
        (os.path.basename(run_dir), datasetsim_singlerun(run_dir, PreProcess_method=preprocess, evoked=evoked))
        for run_dir in run_dirs
    ]


def evaluate_runs(model, run_items, evaluator, cfg, subj_id, test_name):
    rows = []
    global_trial_idx = 0
    method = cfg.get('method', 'TSTNet-low')
    with torch.no_grad():
        for run_index, (run_id, run_dataset) in enumerate(
            tqdm(run_items, desc=f'Evaluating {subj_id}-{test_name}', leave=False)
        ):
            loader = DataLoader(
                run_dataset,
                batch_size=cfg['test_batch_size'],
                shuffle=False,
                num_workers=cfg['num_workers'],
                pin_memory=(cfg['device'] == 'cuda'),
                drop_last=False,
            )
            trial_index_in_run = 0
            for source, eeg in loader:
                source = source.to(device=cfg['device'], dtype=cfg['dtype'])
                eeg = eeg.to(device=cfg['device'], dtype=cfg['dtype'])
                pred = model.predict(eeg)
                for i in range(pred.shape[0]):
                    metrics = evaluator.evaluate(
                        pred=pred[i:i + 1],
                        true_cor=None,
                        true_sig=source[i:i + 1],
                    )
                    rows.append({
                        'run_id': run_id,
                        'test_name': test_name,
                        'method': method,
                        'run_index': run_index,
                        'trial_index_in_run': trial_index_in_run,
                        'trial_index_global': global_trial_idx,
                        'n_channels': int(eeg.shape[-2]),
                        'n_times': int(eeg.shape[-1]),
                        **metrics,
                    })
                    trial_index_in_run += 1
                    global_trial_idx += 1
    summary = {
        'subject_id': subj_id,
        'method': method,
        'test_name': test_name,
        'n_runs': len(run_items),
        'n_trials': len(rows),
    }
    for key in ['SD', 'DLE', 'AUC', 'SE']:
        values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        summary[key] = float(np.mean(values)) if values else float('nan')
    return summary, rows


def main():
    cfg = CFG
    test_names = [f'{name}_{value}' for name, values in TEST_SCRIPTS.items() for value in values]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_rows, trial_rows = [], []

    print('=' * 60)
    print(f"{cfg['method']} on simulated test set")
    print('=' * 60)
    print(f"Device: {cfg['device']}")
    print(f"Subjects: {cfg['subjects']}")
    print(f"Checkpoint: {cfg['checkpoint_path']}")

    for subj_id in cfg['subjects']:
        print('\n' + '=' * 60)
        print(f'[Subject] {subj_id}')
        print('=' * 60)
        try:
            fwd_path = os.path.join(cfg['dir_bids'], 'derivatives', 'sourcemodelling', subj_id, 'fwd', f'{subj_id}_fwd.fif')
            fwd = load_forward_fixed(fwd_path)
            leadfield = normalize_L(np.array(fwd['sol']['data'], dtype=np.float32, copy=True), verbose=True)
            lead_tensor = torch.from_numpy(leadfield).to(device=cfg['device'], dtype=cfg['dtype'])
            cor_sources_tensor = torch.from_numpy((fwd['source_rr'] * 1e3).astype(np.float32)).to(device=cfg['device'], dtype=cfg['dtype'])
            M_matrix = load_M_matrix(cfg['m_matrix_path'], cfg['device'])
            evaluator = Evaluator(
                lead_tensor=lead_tensor,
                cor_sources_tensor=cor_sources_tensor,
                count_source_power_greater_than=cfg['eval_count_source_power_greater_than'],
            )
            model = load_model(cfg, lead_tensor, M_matrix, cor_sources_tensor)

            for test_name in test_names:
                print(f'\n[Test] {subj_id} / {test_name}')
                try:
                    run_items = load_run_items(cfg['test_data_path'], test_name, subj_id, cfg['test_preprocess'], cfg['evoked'])
                    print(f'  n_runs: {len(run_items)}')
                    print(f'  n_trials: {sum(len(ds) for _, ds in run_items)}')
                    summary, rows = evaluate_runs(model, run_items, evaluator, cfg, subj_id, test_name)
                    summary_rows.append(summary)
                    trial_rows.extend(rows)
                    for key, value in summary.items():
                        if key in ('subject_id', 'method', 'test_name'):
                            continue
                        print(f'  {key}: {value}' if isinstance(value, (int, np.integer)) else f'  {key}: {value:.6f}')
                except Exception as exc:
                    print(f'  [Error] {exc}')
                    summary_rows.append({
                        'subject_id': subj_id,
                        'method': cfg['method'],
                        'test_name': test_name,
                        'error': str(exc),
                    })
        except Exception as exc:
            print(f'[Subject Error] {subj_id}: {exc}')
            for test_name in test_names:
                summary_rows.append({
                    'subject_id': subj_id,
                    'method': cfg['method'],
                    'test_name': test_name,
                    'error': f'subject_init_failed: {exc}',
                })

    paths = export_simtest_results(
        summary_rows,
        trial_rows,
        cfg['save_dir'],
        timestamp,
        method=cfg['method'],
    )
    print('\n' + pd.DataFrame(summary_rows).to_string(index=False))
    print('\nResult files:')
    for name, path in paths.items():
        print(f'  {name}: {path}')


if __name__ == '__main__':
    main()
