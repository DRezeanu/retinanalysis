import torch
import numpy as np
import tqdm.auto as tqdm
from typing import Optional, List
import retinanalysis.config.schema as schema
from retinanalysis.utils.datajoint_utils import get_noise_name_by_exp
import os
from retinanalysis.classes.response import (MEAResponseBlock,
                                            MEAResponseGroup,
                                            create_mea_response_group)
from retinanalysis.classes.stim import (MEAStimBlock,
                                        MEAStimGroup,
                                        create_mea_stim_group)

def compute_stas(
    stim_data: torch.Tensor, 
    binned_responses: torch.Tensor,
    depth: int=60,
    method: str="matmul"
    )->np.ndarray:
    """
    Workhorse compute method for STA 
    Can use for EI as well, where instead of frames you have raw data samples

    Args:
        stim_data (torch.Tensor):
            Stimulus data of shape [N epochs, T frames, *]
            For noise stim, last dims are [H, W, C]
            For EI, last dims are [C] electrodes
        binned_responses (torch.Tensor):
            Binned spikerate/spikecount of shape [N epochs, K cells, T frames]
        method (str): "matmul" or "conv"

    Returns:
        stas (np.ndarray):
            [K cells, D depth, *]
            For noise stim, [H, W, C]
            For EI, [C]
    """
    stim_dims = stim_data.shape[2:]
    n_stim_dims = np.prod(stim_dims)
    n_epochs, n_cells, n_frames = binned_responses.shape

    stim_data = stim_data.reshape(n_epochs, n_frames, -1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stas = torch.zeros(n_cells, depth, n_stim_dims).to(device)

    batched_matmul = torch.vmap(torch.matmul)

    if method=="matmul":
        lags = range(depth)
        for i, lag in tqdm.tqdm(list(enumerate(lags)), desc="STA depth"):
            br = binned_responses[:, :, i:]
            if lag > 0:
                sd = stim_data[:, :, :-lag]
            
            # binned spikes [N, K, T] @ stim [N, T, *] = [N, K, *]
            epoch_stas += batched_matmul(br, sd)
            # Avg across epochs for [K, *]
            stas[:, i] += epoch_stas.mean(axis=0)

        # Reshape back to full stim dims
        stas = stas.reshape(n_cells, depth, *stim_dims)

    elif method=="conv":
        raise NotImplementedError("Conv not implemented yet!")
    else:
        raise ValueError("Method must be either matmul or conv!")

    return stas
    
    
def compute_stas_for_chunk(
        exp_name: str, 
        chunk_name: str, 
        ss_version: Optional[str] = 'kilosort2.5',
        datafile_name: Optional[list] = None,
        stride: Optional[int] = 2,
        depth: int=60,
        verbose: bool=True 
    ):
    # If datafile name(s) not given, get noise datafiles
    if datafile_name is None:
        exp_id = schema.Experiment() & {'exp_name': exp_name}
        if len(exp_id) != 1:
            raise ValueError(f"{len(exp_id)} exps found for given exp_name: {exp_name}")
        exp_id = exp_id.fetch('id')[0]
        chunk_id = schema.SortingChunk() & {'experiment_id' : exp_id, 'chunk_name' : chunk_name}
        chunk_id = chunk_id.fetch('id')[0]
        noise_protocol = get_noise_name_by_exp(exp_name)
        protocol_id = schema.Protocol() & {'name' : noise_protocol}
        protocol_id = protocol_id.fetch('protocol_id')[0]
        
        epoch_blocks = schema.EpochBlock() & {'experiment_id' : exp_id, 'chunk_id' : chunk_id, 'protocol_id' : protocol_id}

        noise_data_dirs = epoch_blocks.fetch('data_dir')
        datafile_name = [os.path.basename(path) for path in noise_data_dirs]
        if verbose:
            print(f'Found noise datafile(s): {datafile_name}')

    # Create stim and response groups
    sg = create_mea_stim_group(exp_name, datafile_name, verbose=verbose)
    rg = create_mea_response_group(
        exp_name, datafile_name, ss_version, 
        b_load_fd = True, verbose=verbose)

    # Collect stim and resp data
    stim_data = []
    resp_data = []
    for i in range(len(sg.ls_blocks)):
        sb = sg.ls_blocks[i]
        rb = rg.ls_blocks[i]
        
        # Regen stim
        sb.regenerate_stimulus()
        # Regenerated frames just have stim time frames (no pre/post grey)
        stim_data.append(sb.stim_data['frames'])
        n_frames = sb.stim_data['frames'].shape[1]

        # Bin spike times
        rb.bin_spike_times_by_frames()
        bs = rb.binned_spikes
        
        # Crop out pre frames
        pre_time_ms = rb.d_timing['pre_time_ms']
        stage_frame_rate = rb.d_timing['stage_frame_rate']
        pre_frames = np.floor(pre_time_ms * 1e-3 * stage_frame_rate).astype(int)
        t_start = pre_frames
        t_end = t_start + n_frames
        if verbose:
            print(f"Block {i}: pre_frames={pre_frames}, binned_spikes shape={bs.shape}")
            print(f"Cropping binned spikes to frames: t_start={t_start}, t_end={t_end}")
        bs = bs[:, :, t_start:t_end]
        resp_data.append(bs)
    
    # Create tensors
    stim_data = np.concatenate(stim_data, axis=0)
    resp_data = np.concatenate(resp_data, axis=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # []
    stim_data = torch.tensor(stim_data, device=device)
    # [K cells, N epochs, T frames]
    binned_responses = torch.tensor(resp_data, device=device)
    # Permute to [N epochs, K cells, T frames]
    binned_responses = binned_responses.permute(1, 0, 2)

    
    # Compute STAs
    stas = compute_stas(stim_data, binned_responses, depth=depth, method='matmul')
    return stas

