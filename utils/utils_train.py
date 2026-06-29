



def load_forward_fixed(fwd_path):
    """加载前向解并转换为固定方向"""
    import mne
    from mne.forward import convert_forward_solution

    fwd = mne.read_forward_solution(fwd_path)
    return convert_forward_solution(fwd, force_fixed=True, copy=True)



def source_to_sourceEstimate(data, fwd, sfreq=1, subject='fsaverage', 
    simulationInfo=None, tmin=0, free_ori=True):
    import mne
    from einops import rearrange
    import numpy as np
    ''' Takes source data and creates mne.SourceEstimate object
    https://mne.tools/stable/generated/mne.SourceEstimate.html

    Parameters:
    -----------
    data : numpy.ndarray, shape (number of dipoles x number of timepoints)
    pth_fwd : path to the forward model files sfreq : sample frequency, needed
        if data is time-resolved (i.e. if last dim of data > 1)

    Return:
    -------
    src : mne.SourceEstimate, instance of SourceEstimate.

    '''
    
    data = np.squeeze(np.array(data))
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    source_model = fwd['src']
    
    

    vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
    source_container = mne.VectorSourceEstimate if free_ori else mne.SourceEstimate
    if free_ori:
        data = rearrange(data, 'b (n o) t -> (b n) o t', o=3)
    else:
        data = rearrange(data, 'b n t -> (b n) t')
    src = source_container(data, vertices, tmin=tmin, tstep=1/sfreq)
    
    
    if simulationInfo is not None:
        setattr(src, 'simulationInfo', simulationInfo)


    return src

def sample_results_to_sourceestimate(tensor_list, fwd, tmin, sfreq, free_ori=True,simulationInfo=None):
    '''
    simulationInfo: list of dict. dict 对应刺激信息。list长度与tensor 的batch 维度一致。
    '''
    import mne
    import torch
    from einops import rearrange
    import numpy as np

    with torch.no_grad():
        source_model = fwd['src']
        vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
        source_list = []
        for tensor in tensor_list:
            b, c, n, t = tensor.shape
            source_container = mne.VectorSourceEstimate if free_ori else mne.SourceEstimate
            tensor = tensor.detach().cpu().numpy()
            source_t_list = []
            
            for i in range(b):
                if free_ori:
                    data = rearrange(tensor[i], 'b (n o) t -> (b n) o t', o=3)
                else:
                    data = rearrange(tensor[i], 'b n t -> (b n) t')
                source = source_container(data, vertices, tmin=tmin, tstep=1/sfreq)
                if simulationInfo is not None:
                    setattr(source, 'simulationInfo', simulationInfo[b])
                source_t_list.append(source)
            source_list.append(source_t_list)

    return source_list


def read_estimateConfig_from_yaml(file_path):
    """
    Reads the source estimation configuration from a YAML file.
    
    Parameters:
    file_path (str): Path to the YAML configuration file.
    
    Returns:
    dict: Configuration parameters for source estimation.
    """
    import yaml
    
    # 定义元组构造器
    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))
    
    # 注册元组类型构造器到SafeLoader
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    
    return config


