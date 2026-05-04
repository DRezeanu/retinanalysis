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
import gc

def _get_n_splits_memory(
    sd: torch.Tensor, 
    br: torch.Tensor, 
    device: torch.device, 
    n_max_usage: float=0.8, 
    method: str="matmul",
    verbose: bool=True)->int:
    """
    Get number of splits for STA compute.

    Args:
        sd (torch.Tensor): _description_
        br (torch.Tensor): _description_
        device (torch.device): _description_
        n_max_usage (float, optional): _description_. Defaults to 0.8.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        int: Number of splits.
    """    
    if device.type == 'cuda':
        # Get max memory GB from available GPU
        max_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
    elif device.type == 'mps':
        max_memory_gb = torch.mps.recommended_max_memory() / 1e9
    
    else:
        # cpu
        return 1
    
    max_memory_gb *= n_max_usage

    # Estimate total data size
    total_data_gb = (sd.element_size() * sd.nelement() + br.element_size() * br.nelement()) / 1e9
    n_splits = int(np.ceil(total_data_gb/max_memory_gb))
    if method == 'conv':
        # Conv uses more memory, so adding some buffer splits
        n_splits += 2
    
    if verbose:
        print(f"Total data size: {total_data_gb:.1f}GB, max available: {max_memory_gb:.1f}GB")
        print(f"Recommending {n_splits} splits of compute.")
    return n_splits

def compute_stas(
    stim_data: torch.Tensor, 
    binned_responses: torch.Tensor,
    depth: int=60,
    method: str="matmul",
    verbose: bool=True
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

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    n_splits = _get_n_splits_memory(
        stim_data, binned_responses, 
        device, method=method, verbose=verbose
    )
    n_split_sz = int(np.ceil(n_stim_dims/n_splits))
    stas = torch.zeros(n_cells, depth, n_stim_dims, dtype=torch.float32)

    if method=="matmul":
        batched_matmul = torch.vmap(torch.matmul)
        lags = range(depth)
        for i in tqdm.tqdm(np.arange(n_splits), desc="STA compute chunk"):
            s_start = i * n_split_sz
            s_end = (i+1) * n_split_sz
            if s_end > n_stim_dims:
                s_end = n_stim_dims
            for j, lag in tqdm.tqdm(list(enumerate(lags)), desc="STA depth"):
            # for j, lag in enumerate(lags):
                br = binned_responses[:, :, j:]
                sd = stim_data[:, :, s_start:s_end]
                if lag > 0:
                    sd = stim_data[:, :-lag, s_start:s_end]
                
                br = br.to(device)
                sd = sd.to(device)

                # binned spikes [N, K, T] @ stim [N, T, S] = [N, K, S]
                epoch_stas = batched_matmul(br, sd)

                # Avg across epochs for [K, S]
                stas[:, j, s_start:s_end] += epoch_stas.mean(axis=0).cpu()

                # Clear memory
                del br, sd, epoch_stas
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
        # Reverse time dim for standard convention
        stas = stas[:, ::-1, :]

    elif method=="conv":
        # The shapes here can get confusing. Key to note is:
        # Using conv2d so that (H, W) of input can be filled by (stim_dims, time+depth-1)
        # convolving that with (K, 1, time) responses gives (K, stim_dims, depth) output.
        # Singleton dims added where needed. Batching over epochs with vmap.

        batched_conv = torch.vmap(torch.nn.functional.conv2d)
        for i in tqdm.tqdm(np.arange(n_splits), desc="STA compute chunk"):
            s_start = i * n_split_sz
            s_end = (i+1) * n_split_sz
            if s_end > n_stim_dims:
                s_end = n_stim_dims
            
            # [N, K, 1, 1, T]
            br = binned_responses.unsqueeze(2).unsqueeze(2)
            # [N, T, S]
            sd = stim_data[:, :, s_start:s_end]
            # permute to [N, S, T]
            sd = sd.permute(0, 2, 1)

            # Pad stim data on left with depth-1 zeros
            # so [N, S, T+depth-1]
            sd = torch.nn.functional.pad(
                sd, (depth-1, 0)
            )

            # [N, 1, 1, S, T+depth-1]
            sd = sd.unsqueeze(1).unsqueeze(1)

            br = br.to(device)
            sd = sd.to(device)
            
            # batch over [N], conv ([1, 1, S, T+D-1], [K, 1, 1, T]) -> [N, 1, K, S, D]
            with torch.no_grad():
                epoch_stas = batched_conv(sd, br, padding='valid')
            
            # Remove singleton and swap last two dims -> [N, K, D, S]
            epoch_stas = epoch_stas.squeeze(1)
            epoch_stas = epoch_stas.transpose(2, 3)

            # Avg across epochs for [K, D, S]
            stas[:, :, s_start:s_end] = epoch_stas.mean(axis=0).cpu()

            del br, sd, epoch_stas
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()


    else:
        raise ValueError("Method must be either matmul or conv!")

    # Reshape back to full stim dims
    stas = stas.reshape(n_cells, depth, *stim_dims)
    stas = stas.numpy()
    
    # Final clean up
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    return stas
    


def compute_stas_for_chunk(
        sg = None,
        rg = None,
        exp_name: str=None, 
        chunk_name: str=None, 
        ss_version: Optional[str] = 'kilosort2.5',
        datafile_name: Optional[list] = None,
        stride: Optional[int] = 2,
        depth: int=60,
        verbose: bool=True 
    ):
    if sg is None or rg is None:
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
            b_load_fd = True, verbose=verbose
        )

    # Collect stim and resp data
    stim_data = []
    resp_data = []
    for i in range(len(sg.ls_blocks)):
        sb = sg.ls_blocks[i]
        rb = rg.ls_blocks[i]

        sb.regenerate_stimulus(n_jobs=10)
        
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
            print(f"Total frames: {n_frames}")
        bs = bs[:, :, t_start:t_end]
        if verbose:
            print(f"Cropped binned spikes shape: {bs.shape}")
        resp_data.append(bs)
    
    # Create tensors
    stim_data = np.concatenate(stim_data, axis=0)
    resp_data = np.concatenate(resp_data, axis=0)    

    # [N epochs, T frames, H, W, C]
    stim_data = torch.tensor(stim_data, dtype=torch.float32)
    # [K cells, N epochs, T frames]
    binned_responses = torch.tensor(resp_data, dtype=torch.float32)
    # Permute to [N epochs, K cells, T frames]
    binned_responses = binned_responses.permute(1, 0, 2)

    stas = compute_stas(stim_data, binned_responses, depth=depth, method='conv', verbose=verbose)

    return stas

