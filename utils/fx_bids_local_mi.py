# -*- coding: utf-8 -*-
def load_sti_info(bids_root, subj_id, task, run):
    import os.path as op
    import csv
    import pandas as pd
    import numpy as np  

    sti_csv_file = op.join(bids_root, 'derivatives', 'epochs', subj_id, 'eeg',
                           '%s_task-%s_%s_epochs.tsv' % (subj_id, task, run))
    seeg_ch_info = pd.read_csv(op.join(bids_root, 'derivatives', 'epochs', subj_id, 'ieeg', '%s_task-%s_space-surface_electrodes.tsv' % (subj_id, task)), sep='\t')
    durations = []
    electrodes = []
    strengths_mA = []
    stim_chs_loc = []
    stim_coords_bips = []
    stim_hemis_idx = []
    stim_len_contacts = [] # 单位mm
    stim_len_between_contacts = [] # 单位mm
    stim_freq = []
    pulse_width = []  # 假设脉冲宽度为0.5ms
    stim_hemis = []
    with open(sti_csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            # 记录 duration、zero_time
            
            # durations.append(float(float(row['duration'])*1000.0-float(row['zero_time'])*1000.0))
            
            # trial_type示例: "K13-14 1mA"
            trial_type = row['trial_type'].strip()
            
            # 简单按空格拆分
            parts = trial_type.split()
            if len(parts) == 2:  
                electrode, current = parts
                current_value = ''.join(filter(lambda x: x.isdigit() or x == '.', current))
                current = float(current_value) if current_value else None
                if current is not None:
                    if current == 1 or current == 5:
                        stim_freq.append([0.5])
                    else:
                        stim_freq.append([1.0])
            else:
                # 若格式不标准，可根据需要进行更健壮的处理
                electrode = trial_type
                current = ""
            stim_chs = [electrode.split('-')[0], ''.join(n for n in electrode.split('-')[0] if not n.isdigit()) + electrode.split('-')[1]] # get monopolar names
            

            stim_ch_loc = seeg_ch_info.loc[seeg_ch_info.name.isin(stim_chs)][['x', 'y', 'z']].values.squeeze() # find the coordinates of the stimulating channels
            stim_coords_bip = np.mean(stim_ch_loc, 0)  # take the mean between the two contacts
            hemi_ix = 0 if '\'' in stim_chs[0] else 1
            hemi = 'lh' if '\'' in stim_chs[0] else 'rh'
            electrodes.append(stim_chs)
            strengths_mA.append(current)
            stim_chs_loc.append(stim_ch_loc)
            stim_coords_bips.append(stim_coords_bip)
            stim_hemis_idx.append(hemi_ix)
            stim_hemis.append(hemi)
            pulse_width.append(0.5)
            durations.append(1) 
            stim_len_contacts.append(2)
            stim_len_between_contacts.append(1.5)
    # flat_list = [item for pair in electrodes for item in pair]
    # unique_electrodes = sorted(set(flat_list))
    # print(unique_electrodes)
    # stim_coords = seeg_ch_info.loc[seeg_ch_info.name.isin(unique_electrodes)][['x', 'y', 'z']].values.squeeze() # find the coordinates of the stimulating channels
    
    # for ch, coord in zip(unique_electrodes, stim_coords):
    #     sti_ch_loc[ch] = coord
    

    # 返回一个字典，包含所有信息
    return {
        'durations': durations,
        'electrodes': electrodes,
        'strengths_mA': strengths_mA,
        'stim_coords': stim_chs_loc,
        'stim_coords_bips': stim_coords_bips,
        'stim_hemis_idx': stim_hemis_idx,
        'stim_hemis': stim_hemis,
        'stim_len_contacts': stim_len_contacts,
        'stim_len_between_contacts': stim_len_between_contacts,
        'stim_freq': stim_freq,
        'pulse_width': pulse_width # 脉冲宽度为0.5ms
    }
    



# ====================================================================
# 以下是数据集提供的原始函数
# ====================================================================
def make_runs_table(dir_base):
    import pandas as pd
    import os
    import natsort
    import re

    dir_data = os.path.join(dir_base, 'data', 'fif')
    files = os.listdir(dir_data)
    files = natsort.natsorted(files)

    subj_id, subj_code, run_id, fname = [], [], [], []

    for f in files:
        spl = f.split('_')
        subj_id.append(re.findall(r'\d', spl[0])[0].zfill(2))
        subj_code.append(spl[0])
        run_id.append('%s_%s' % (spl[1], spl[2]))
        fname.append(os.path.join(dir_data, f))
    runs_table = pd.DataFrame({'subj_id': subj_id, 'subj_code': subj_code,
                               'run_id': run_id, 'fname': fname})
    runs_table['run_nr'] = runs_table.groupby('subj_id').cumcount() + 1

    fname_save = os.path.join(dir_base, 'share', 'runs_table.csv')
    runs_table.to_csv(fname_save, index=False)



def make_coordsys_ieeg_json(dir_json_ieeg, subj_id, task):
    import os
    import json

    for system in ['T1w', 'mod-T1w_maskface', 'MNI152NLin2009aSym', 'surface']:
        coordys = {"iEEGCoordinateUnits": "m"}

        if 'T1w' in system:
            coordys["iEEGCoordinateSystem"] = 'Other'
            coordys["iEEGCoordinateSystemDescription"] = 'Native MRI space'
            coordys['iEEGCoordinateProcessingReference'] = 'Narizzano, M., Arnulfo, G., Ricci, S., Toselli, B., Tisdall, M., Canessa, A., Fato, M.M., Cardinale, F., 2017. SEEG assistant: a 3DSlicer extension to support epilepsy surgery. BMC Bioinformatics 18. https://doi.org/10.1186/s12859-017-1545-8'
            if 'maskface' in system:
                coordys['IntendedFor'] = '/derivatives/%s/epochs/anat/%s_%s.nii' % (subj_id, subj_id, system)

            else:
                coordys['IntendedFor'] = '/%s/anat/%s_%s.nii' % (subj_id, subj_id, system)

        elif 'MNI' in system:
            coordys['iEEGCoordinateSystem'] = 'ICBM 2009a Nonlinear Symmetric'
            coordys['iEEGCoordinateProcessingReference'] = 'Narizzano, M., Arnulfo, G., Ricci, S., Toselli, B., Tisdall, M., Canessa, A., Fato, M.M., Cardinale, F., 2017. SEEG assistant: a 3DSlicer extension to support epilepsy surgery. BMC Bioinformatics 18. https://doi.org/10.1186/s12859-017-1545-8 /// Avants, B.B., Tustison, N.J., Song, G., Cook, P.A., Klein, A., Gee, J.C., 2011. A reproducible evaluation of ANTs similarity metric performance in brain image registration. NeuroImage 54, 2033–2044. https://doi.org/10.1016/j.neuroimage.2010.09.025'
            coordys['iEEGCoordinateProcessingDescription'] = 'Skull stripping and SyN Registration (ANTs) to ICBM152'

        else:
            coordys["iEEGCoordinateSystem"] = 'Other'
            coordys["iEEGCoordinateSystemDescription"] = 'Native surface space'
            coordys['IntendedFor'] = '/derivatives/sourcemodelling/%s/anat/%s_hemi-[L-R]_pial.surf.gii' % (subj_id, subj_id)
            coordys['iEEGCoordinateProcessingReference'] = 'Narizzano, M., Arnulfo, G., Ricci, S., Toselli, B., Tisdall, M., Canessa, A., Fato, M.M., Cardinale, F., 2017. SEEG assistant: a 3DSlicer extension to support epilepsy surgery. BMC Bioinformatics 18. https://doi.org/10.1186/s12859-017-1545-8'

        fpath_json_sys = os.path.join(dir_json_ieeg, '%s_task-%s_space-%s_coordsystem.json' % (subj_id, task, system))
        with open(fpath_json_sys, 'w') as f:
            json.dump(coordys, f, indent=4)


def make_coordsys_eeg_json(dir_coordsys_eeg_json, fname_epo, subj_id, task):
    import os.path as op
    import json
    from mne.viz._3d import _fiducial_coords
    import mne

    epo = mne.read_epochs(fname_epo)

    coordys = {"EEGCoordinateSystem": "T1w",
               "EEGCoordinateUnits": "m",
               "AnatomicalLandmarkCoordinates": {}}

    fids = _fiducial_coords(epo.info['dig'])
    coordys['AnatomicalLandmarkCoordinates']['LPA'] = [float(n) for n in fids[0]]  # has to be float and not np.float to be able to json dump
    coordys['AnatomicalLandmarkCoordinates']['NAS'] = [float(n) for n in fids[1]]
    coordys['AnatomicalLandmarkCoordinates']['RPA'] = [float(n) for n in fids[2]]
    coordys['AnatomicalLandmarkCoordinateSystem'] = "T1w"
    coordys['IntendedFor'] = '/%s/anat/%s_T1.nii' % (subj_id, subj_id)
    coordys['AnatomicalLandmarkCoordinateUnits'] = "m"

    fname_save = op.join(dir_coordsys_eeg_json, '%s_task-%s_coordsystem.json' % (subj_id, task))
    with open(fname_save, 'w') as f:
        json.dump(coordys, f, indent=4)


def make_electrodes_ieeg_tsv(dir_electrodes_ieeg_tsv, subj_id, seeg_coords, taskname):
    import pandas as pd
    import os.path as op

    for system in ['T1w', 'mod-T1w_maskface', 'MNI152NLin2009aSym', 'surface']:
        if 'T1w' in system:
            elec = pd.DataFrame({'name': seeg_coords.name, 'x': seeg_coords.x_mri/1e3, 'y': seeg_coords.y_mri/1e3,
                                 'z': seeg_coords.z_mri/1e3})
        elif 'MNI' in system:
            elec = pd.DataFrame({'name': seeg_coords.name, 'x': seeg_coords.x_norm_mri/1e3, 'y': seeg_coords.y_norm_mri/1e3,
                                 'z': seeg_coords.z_norm_mri/1e3})
        else:
            elec = pd.DataFrame({'name': seeg_coords.name, 'x': seeg_coords.x_surf/1e3, 'y': seeg_coords.y_surf/1e3,
                                 'z': seeg_coords.z_surf/1e3})
        elec['size'] = 7.5
        elec['manufacturer'] = 'Dixi Medical'
        elec['material'] = 'PtIr'
        elec.sort_values(['name'])
        elec = elec.round(5)
        fname_save = op.join(dir_electrodes_ieeg_tsv, '%s_task-%s_space-%s_electrodes.tsv' % (subj_id, taskname, system))
        elec.to_csv(fname_save, sep='\t', index=False)
    print('Done creating electrodes.tsv')


def make_electrodes_eeg_tsv(dir_electrodes_eeg_tsv, fname_epo, subj_id, task):
    import pandas as pd
    import os.path as op
    import mne

    epo = mne.read_epochs(fname_epo)
    dig = {epo.info['chs'][ix]['ch_name']: epo.info['chs'][ix]['loc'][:3]
           for ix in range(len(epo.info['chs']))}
    chans = list(dig.keys())
    chans.remove('STI')

    x, y, z = [dig[c][0] for c in chans], [dig[c][1] for c in chans], [dig[c][2] for c in chans]
    elec = pd.DataFrame({'name': chans, 'x': x, 'y': y, 'z': z})
    elec['material'] = 'HydroCel CleanLeads'
    elec = elec.round(5)

    fname_save = op.join(dir_electrodes_eeg_tsv, '%s_task-%s_electrodes.tsv' % (subj_id, task))
    elec.to_csv(fname_save, sep='\t', index=False)


def make_events_json(fname_events_json):
    import json
    events_json = {'electrical_stimulation_site': 'Electrodes involved in the stimulation',
                   'electrical_stimulation_current': 'Stimulation current (A)',
                   'electrical_stimulation_frequency': 'Frequency of stimulation (Hz)',
                   'electrical_stimulation_type': 'Kind of wave'}

    with open(fname_events_json, 'w') as f:
        json.dump(events_json, f, indent=4)


def export_data_bids(fpath_epo, dir_out, task, run, subject_id, event_id):
    import mne
    from mne_bids import make_bids_basename
    from mne_bids.write import _participants_json, _participants_tsv

    import os.path as op
    import pandas as pd
    import numpy as np
    import json

    epo = mne.read_epochs(fpath_epo)
    epo = epo.crop(None, 0.01)
    bids_basename = make_bids_basename(subject=subject_id, task=task, run=run)
    print(bids_basename)
    subj_id_bids = 'sub-' + subject_id
    intensity = fpath_epo.split('_')[2]
    intensity = intensity.replace('a', 'A')
    if len(intensity) > 3:
        intensity = intensity.replace('0', '0.')  # add decimal


    # participants
    fname_participants = op.join(dir_out, 'participants')
    _participants_tsv(epo, subject_id, fname_participants + '.tsv')
    _participants_json(fname_participants + '.json', overwrite=True)

    # channels tsv
    ch_names = epo.ch_names.copy()
    ch_names.remove('STI')
    status = ['good' if c not in epo.info['bads'] else 'bad' for c in ch_names]

    chans = {'name': ch_names, 'type': ['EEG']*len(ch_names),
             'units': ['V']*len(ch_names), 'low_cutoff': ['0.1']*len(ch_names),
             'high_cutoff': ['n/a']*len(ch_names),
             'sampling_frequency': [8000]*len(ch_names), 'status': status}

    chans_tsv = pd.DataFrame(chans)
    fname_chans_tsv = op.join(dir_out, 'derivatives', 'epochs', subj_id_bids,
                              'eeg', bids_basename + '_channels.tsv')
    chans_tsv.to_csv(fname_chans_tsv, index=False, sep='\t')

    # epochs.tsv
    stim_ch = op.split(fpath_epo)[-1].split('_')[1]
    n_epo = len(epo)
    duration = np.abs(epo.times[0]) + epo.times[-1]
    epo_tsv = pd.DataFrame({'duration': [duration] * n_epo,
                            'zero_time': [np.abs(epo.times[0])] * n_epo,
                            'trial_type': ['%s %s' % (stim_ch, intensity)] * n_epo})

    fname_epo_tsv = op.join(dir_out, 'derivatives', 'epochs',
                            subj_id_bids, 'eeg',
                            bids_basename + '_epochs.tsv')
    epo_tsv.to_csv(fname_epo_tsv, index=False, sep='\t')

    # epochs json
    epo_json = {'Description': 'Stimulation of channel %s %s' % (stim_ch, intensity),
                'Sources': '/%s/eeg/%s_eeg.npy' % (subj_id_bids, bids_basename),
                'BaselineCorrection': True,
                'BaselineCorrectionMethod': 'mean subtraction',
                'BaselinePeriod': [-0.3, -0.05]}

    fname_epo_json = op.join(dir_out, 'derivatives', 'epochs',
                             subj_id_bids, 'eeg',
                             bids_basename + '_epochs.json')

    with open(fname_epo_json, 'w') as f:
        json.dump(epo_json, f, indent=4)

    # data
    dat = epo.get_data()[:, :-1, :]  # omit trigger
    fname_dat = op.join(dir_out, 'derivatives', 'epochs',
                        subj_id_bids, 'eeg',
                        bids_basename + '_epochs.npy')
    np.save(fname_dat, dat)


def make_mri_json(subj_id, dir_out):
    import os.path as op
    import json
    fname_deface_json = op.join(dir_out, subj_id, 'anat', '%s_T1w.json' % subj_id)
    deface_info = {'ImageProcessingSoftware': "Pydeface - https://github.com/poldracklab/pydeface"}

    with open(fname_deface_json, 'w') as f:
        json.dump(deface_info, f, indent=4)

    fname_maskface_json = op.join(dir_out, 'derivatives', 'epochs', subj_id, 'anat', '%s_mod-T1w_maskface.json' % subj_id)
    maskface_info = {'ImageProcessingSoftware': "Face Masking - Milchenko, M. & Marcus, D. Obscuring Surface Anatomy "
                                                "in Volumetric Imaging Data. Neuroinform 11, 65–75 (2013) - "
                                                "https://nrg.wustl.edu/software/face-masking/"}

    with open(fname_maskface_json, 'w') as f:
        json.dump(maskface_info, f, indent=4)


def load_bids(dir_bids, subj_id, task, run_id):
    import mne
    import os.path as op
    import numpy as np
    import pandas as pd
    import json

    bids_fname_base = op.join(dir_bids, 'derivatives', 'epochs', subj_id, 'eeg',
                             '%s_task-%s_%s' % (subj_id, task,  run_id))

    fname_eeg = bids_fname_base + '_epochs.npy'
    fname_chans = bids_fname_base + '_channels.tsv'
    fname_elecs = bids_fname_base.replace(run_id, '') + 'electrodes.tsv' 
    fname_coordsys = bids_fname_base.replace(run_id, 'coordsystem.json')

    with open(fname_coordsys) as json_file:
        coordsys = json.load(json_file)

    fiducials = coordsys['AnatomicalLandmarkCoordinates']

    data = np.load(fname_eeg)
    chans = pd.read_csv(fname_chans, sep='\t')
    ch_names = chans.name.tolist()

    elecs = pd.read_csv(fname_elecs, sep='\t')
    dig_ch_pos = dict(zip(ch_names, elecs[['x', 'y', 'z']].values))

    mont = mne.channels.make_dig_montage(dig_ch_pos, nasion=fiducials['NAS'],
                                         rpa=fiducials['RPA'], lpa=fiducials['LPA'],
                                         coord_frame='head')

    info = mne.create_info(ch_names, sfreq=8000,  # todo: srate from bids file
                           ch_types=['eeg']*len(chans))
    epo = mne.EpochsArray(data, info, tmin=-0.25)  # todo: tmin from bids file
    epo.set_montage(mont)

    ch_status = chans.status.tolist()
    bads = [c for c, s in zip(ch_names, ch_status) if s == 'bad']
    epo.info['bads'] = bads
    epo.baseline = (-0.25, -0.05)  # todo: baseline from bids file
    
    return epo


def list_sim_runs(dir_sim_root, test_name, subj_id='sub-01'):
    import os
    import os.path as op
    import re

    runs_dir = op.join(dir_sim_root, test_name, subj_id, 'runs_data')
    if not op.isdir(runs_dir):
        raise FileNotFoundError(f'未找到模拟测试集目录: {runs_dir}')

    run_names = []
    for name in os.listdir(runs_dir):
        full = op.join(runs_dir, name)
        if op.isdir(full) and re.fullmatch(r'run-\d{5}', name):
            run_names.append(name)
    return sorted(run_names)



def load_simulated_run(dir_sim_root,
                       test_name,
                       run_id,
                       subj_id='sub-01',
                       preprocess=None,
                       evoked=False,
                       info=None,
                       dir_bids=None,
                       task=None,
                       ref_run=None,
                       sfreq=8000,
                       tmin=-0.001):
    """读取单个模拟 run，并构造成类似 ``load_bids`` 的输出。

    Parameters
    ----------
    dir_sim_root : str
        模拟测试集根目录，例如 ``.../derivatives/simulationV3/testset``。
    test_name : str
        测试条件名，例如 ``firingsource_1`` / ``snr_5dB``。
    run_id : str
        run 目录名，例如 ``run-00000``。
    subj_id : str
        被试 ID。
    preprocess : str | None
        ``None`` / ``'EA'`` / ``'White'``。分别读取原始 EEG、EA EEG、白化 EEG。
    evoked : bool
        若为 True，则对 trial 求平均并返回单 trial Epochs。
    info : mne.Info | None
        直接提供参考 ``Info``。若为 None，可通过 ``dir_bids`` + ``task`` + ``ref_run``
        调用 ``load_bids`` 自动获取真实 EEG 的通道信息与 montage。
    dir_bids, task, ref_run : str | None
        用于自动构造参考 ``Info``。
    sfreq : float
        当无法获得参考 ``Info`` 时创建简化 ``Info`` 使用的采样率。
    tmin : float
        当无法获得参考 ``Info`` 时创建 ``EpochsArray`` 使用的起始时间。

    Returns
    -------
    tuple
        ``(epochs, source, source_meta, eeg_meta)``
        - ``epochs``: ``mne.EpochsArray``，形状为 ``(n_trials, n_channels, n_times)``
        - ``source``: ``np.ndarray``，形状为 ``(n_sources, n_times)``
        - ``source_meta``: ``sim_source.npz`` 中除 ``data`` 以外的字段
        - ``eeg_meta``: ``sim_run_eegs.npz`` 中除 ``data`` 以外的字段
    """
    import os.path as op
    from copy import deepcopy
    import numpy as np
    import mne

    preprocess_map = {
        None: 'sim_run_eegs.npz',
        'EA': 'sim_run_eegs_preEA.npy',
        'White': 'sim_run_eegs_preWhite.npy',
    }
    if preprocess not in preprocess_map:
        raise ValueError("preprocess 必须为 None, 'EA' 或 'White'")

    run_dir = op.join(dir_sim_root, test_name, subj_id, 'runs_data', run_id)
    if not op.isdir(run_dir):
        raise FileNotFoundError(f'未找到模拟 run 目录: {run_dir}')

    source_npz = np.load(op.join(run_dir, 'sim_source.npz'), allow_pickle=True)
    try:
        source = np.asarray(source_npz['data'], dtype=np.float32)
        source_meta = {k: source_npz[k] for k in source_npz.files if k != 'data'}
    finally:
        source_npz.close()

    eeg_meta = {}
    eeg_fname = preprocess_map[preprocess]
    eeg_path = op.join(run_dir, eeg_fname)
    if preprocess is None:
        eeg_npz = np.load(eeg_path, allow_pickle=True)
        try:
            eeg = np.asarray(eeg_npz['data'], dtype=np.float32)
            eeg_meta = {k: eeg_npz[k] for k in eeg_npz.files if k != 'data'}
        finally:
            eeg_npz.close()
    else:
        eeg = np.asarray(np.load(eeg_path), dtype=np.float32)

    if eeg.ndim == 2:
        eeg = eeg[np.newaxis, ...]
    if evoked:
        eeg = eeg.mean(axis=0, keepdims=True)

    if info is None and dir_bids is not None and task is not None and ref_run is not None:
        info = load_bids(dir_bids, subj_id, task, ref_run).info.copy()

    if info is None:
        ch_names = [f'EEG{i+1:03d}' for i in range(eeg.shape[1])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                               ch_types=['eeg'] * eeg.shape[1])
        epo = mne.EpochsArray(eeg, info, tmin=tmin, verbose=False)
    else:
        info = deepcopy(info)
        if len(info['ch_names']) != eeg.shape[1]:
            raise ValueError(
                f'参考 Info 通道数与模拟 EEG 不一致: info={len(info["ch_names"])} vs eeg={eeg.shape[1]}'
            )
        epo = mne.EpochsArray(eeg, info, tmin=tmin, verbose=False)

    return epo, source, source_meta, eeg_meta



def load_simulated_testset(dir_sim_root,
                           test_name,
                           subj_id='sub-01',
                           preprocess=None,
                           evoked=False,
                           info=None,
                           dir_bids=None,
                           task=None,
                           ref_run=None,
                           sfreq=8000,
                           tmin=-0.001):
    """读取整个模拟测试条件下的全部 run。"""
    runs = list_sim_runs(dir_sim_root, test_name=test_name, subj_id=subj_id)
    loaded = []
    for run_id in runs:
        loaded.append(
            load_simulated_run(
                dir_sim_root=dir_sim_root,
                test_name=test_name,
                run_id=run_id,
                subj_id=subj_id,
                preprocess=preprocess,
                evoked=evoked,
                info=info,
                dir_bids=dir_bids,
                task=task,
                ref_run=ref_run,
                sfreq=sfreq,
                tmin=tmin,
            )
        )
    return loaded



def load_trans(fname_trans):
    import h5py
    import mne

    trans_ori = h5py.File(fname_trans).get('trans')[()]
    trans = mne.transforms.Transform(fro='head', to='mri', trans=trans_ori)
    return trans