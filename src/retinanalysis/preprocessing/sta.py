import torch
import numpy as np
import tqdm.auto as tqdm
from typing import Optional, List
from retinanalysis import regen
import argparse
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
from vision_utils import STAWriter

def _get_n_splits_memory(
    sd: torch.Tensor, 
    br: torch.Tensor, 
    stride: int,
    device: torch.device, 
    n_max_usage: float=0.8, 
    method: str="matmul",
    verbose: bool=True)->int:
    """
    Get number of splits for STA compute.

    Args:
        sd (torch.Tensor): _description_
        br (torch.Tensor): _description_
        stride (int): Stride for upsampling stimulus data to match binned responses.
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
    total_data_gb = (sd.element_size() * sd.nelement() * stride + br.element_size() * br.nelement()) / 1e9
    n_splits = int(np.ceil(total_data_gb/max_memory_gb))
    if method == 'conv':
        # Conv uses more memory, so adding some buffer splits
        n_splits *= 2
    
    if verbose:
        print(f"Total data size: {total_data_gb:.1f}GB, max available: {max_memory_gb:.1f}GB")
        print(f"Recommending {n_splits} splits of compute.")
    return n_splits

def compute_stas(
    stim_data: np.ndarray, 
    binned_responses: np.ndarray,
    depth: int=60,
    stride: int=2,
    method: str="matmul",
    verbose: bool=True
    )->np.ndarray:
    """
    Workhorse compute method for STA 
    Can use for EI as well, where instead of frames you have raw data samples

    Args:
        stim_data (np.ndarray):
            Stimulus data of shape [N epochs, T frames, *]
            For noise stim, last dims are [H, W, C]
            For EI, last dims are [C] electrodes
        binned_responses (np.ndarray):
            Binned spikerate/spikecount of shape [N epochs, K cells, T frames]
        stride (int): Stride for upsampling stimulus data to match binned responses. If stim_data is already upsampled, set to 1.
        method (str): "matmul" or "conv"

    Returns:
        stas (np.ndarray):
            [K cells, D depth, *]
            For noise stim, [H, W, C]
            For EI, [C]
    """
    # [N epochs, T frames, H, W, C]
    stim_data = torch.tensor(stim_data, dtype=torch.float32)
    # [N epochs, K cells, T frames]
    binned_responses = torch.tensor(binned_responses, dtype=torch.float32)

    stim_dims = stim_data.shape[2:]
    n_stim_dims = np.prod(stim_dims)
    n_epochs, n_cells, n_bins = binned_responses.shape

    n_frames = stim_data.shape[1]
    stim_data = stim_data.reshape(n_epochs, n_frames, -1)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    n_splits = _get_n_splits_memory(
        stim_data, binned_responses, stride,
        device, method=method, verbose=verbose
    )
    n_split_sz = int(np.ceil(n_stim_dims/n_splits))
    stas = torch.zeros(n_cells, depth, n_stim_dims, dtype=torch.float32)

    binned_responses = binned_responses.to(device)

    if method=="matmul":
        batched_matmul = torch.vmap(torch.matmul)
        lags = range(depth)
        for i in tqdm.tqdm(np.arange(n_splits), desc="STA compute chunk"):
            s_start = i * n_split_sz
            s_end = (i+1) * n_split_sz
            if s_end > n_stim_dims:
                s_end = n_stim_dims
            
            # Put stim data chunk on device
            sd = stim_data[:, :, s_start:s_end].to(device)
            # Upsample by stride
            sd = torch.repeat_interleave(sd, stride, dim=1)

            for j, lag in tqdm.tqdm(list(enumerate(lags)), desc="STA depth"):
                br_lag = binned_responses[:, :, j:]
                sd_lag = sd[:, :n_bins-lag, :]

                # binned spikes [N, K, T] @ stim [N, T, S] = [N, K, S]
                epoch_stas = batched_matmul(br_lag, sd_lag)

                # Avg across epochs for [K, S]
                stas[:, j, s_start:s_end] += epoch_stas.mean(axis=0).cpu()

            # Clear memory
            del sd, br_lag, sd_lag, epoch_stas
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
        
        # Reverse time dim for standard convention
        stas = torch.flip(stas, dims=[1])

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
            # Upsample sd by stride
            sd = torch.repeat_interleave(sd, stride, dim=1)
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
    

def get_noise_datafiles(exp_name, chunk_name):
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
    return datafile_name

def compute_stas_for_chunk(
        sg: MEAStimGroup=None,
        rg: MEAResponseGroup = None,
        exp_name: str=None, 
        chunk_name: str=None, 
        ss_version: Optional[str] = 'kilosort2.5',
        datafile_name: Optional[list] = None,
        stride: Optional[int] = 2,
        depth: Optional[int] = 60,
        method: Optional[str] = "conv",
        verbose: bool=True 
    ):
    if sg is None or rg is None:
        # If datafile name(s) not given, get noise datafiles
        if datafile_name is None:
            datafile_name = get_noise_datafiles(exp_name, chunk_name)
            if verbose:
                print(f'Found noise datafile(s): {datafile_name}')

        # Create stim and response groups
        sg = create_mea_stim_group(exp_name, datafile_name, verbose=verbose)
        rg = create_mea_response_group(
            exp_name, datafile_name, ss_version, 
            b_load_fd = True, verbose=verbose
        )

    # STA input gen and calc loop
    stas = None
    # Number of epochs to process in batch
    n_epochs_batch = 4
    n_blocks = len(sg.ls_blocks)
    for i in range(n_blocks):
        sb = sg.ls_blocks[i]
        rb = rg.ls_blocks[i]
        
        # Get number of frames (assuming same across epochs)
        ls_unique_frames, ls_repeat_frames = regen.get_n_frames_spatial_noise(sb.df_epochs)
        total_frames = np.array(ls_unique_frames) + np.array(ls_repeat_frames)
        if len(np.unique(total_frames))!=1:
            raise ValueError(f"Uneven number of frames across epochs! {total_frames}")
        n_frames = total_frames[0]

        # Bin spike times
        rb.bin_spike_times_by_frames(stride=stride)
        # [K, N, T]
        bs = rb.binned_spikes
        # Make [N, K, T]
        bs = bs.transpose(1, 0, 2)

        # Crop out pre frames
        pre_time_s = rb.d_timing['pre_time_ms'] / 1e3

        stage_frame_rate = rb.d_timing['stage_frame_rate']
        # Count n frames where state.time (1/fr steps) is < pre_time_s
        pre_frames = len(np.arange(0, pre_time_s, 1/stage_frame_rate))
        # LCR CORRECTION
        # pre_frames -= 1
        t_start = pre_frames * stride
        t_end = t_start + n_frames * stride
        if verbose:
            print(f"Block {i}: pre_frames={pre_frames}, binned_spikes shape={bs.shape}")
            print(f"Total frames: {n_frames}")
        bs = bs[:, :, t_start:t_end]
        if verbose:
            print(f"Cropped binned spikes shape: {bs.shape}")
        
        # Loop across epochs in batch
        n_epochs = len(sb.df_epochs)
        n_batches = int(np.ceil(n_epochs/n_epochs_batch))
        for j in tqdm.tqdm(np.arange(n_batches), desc="Epoch batch"):
            e_start = j * n_epochs_batch
            e_end = (j+1) * n_epochs_batch
            if e_end > n_epochs:
                e_end = n_epochs
            
            # Regen stim
            sb.regenerate_stimulus(ls_epochs=list(range(e_start, e_end)))
            # [N, T, H, W, C]
            stim_frames = sb.stim_data['frames']
            
            # [N, K, T]
            resp_data = bs[e_start:e_end]

            if stas is None:
                stas = compute_stas(
                    stim_frames, resp_data,
                    depth=depth, stride=stride, method=method, verbose=verbose
                )

            else:
                stas += compute_stas(
                    stim_frames, resp_data,
                    depth=depth, stride=stride, method=method, verbose=verbose
                )
            
            del stim_frames, resp_data, sb.stim_data
            gc.collect()

    # Normalize by abs max for each cell
    stas = stas / np.abs(stas).max(axis=(1,2,3), keepdims=True)

    grid_size = sb.df_epochs.at[0, 'epoch_parameters']['gridSize']

    return stas, rg.cell_ids, grid_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='sta.py',
        description='Compute STAs for given experiment and chunk.'
    )
    parser.add_argument('exp_name', help='Experiment name (e.g. 20260715C)')
    parser.add_argument('chunk_name', help='Chunk name (e.g. chunk1)')
    parser.add_argument('datafile_name', nargs='*', help='Datafile name(s) to process (e.g. data000). If not given, will attempt to find noise datafiles for the given exp and chunk.')
    parser.add_argument('--ss_version', default='kilosort2.5', help='Spike sorting version to load responses from (default: kilosort2.5)')
    parser.add_argument('--stride', type=int, default=2, help='Stride (bins/frame)')
    parser.add_argument('--depth', type=int, default=61, help='STA depth (bins)')
    parser.add_argument('--method', type=str, default='matmul', help='Method for computing STAs (default: matmul)')

    args = parser.parse_args()

    SAVE_DIR = '/home/vyomr/Desktop/data/analysis'
    
    stas, cell_ids, grid_size = compute_stas_for_chunk(
        exp_name=args.exp_name,
        chunk_name=args.chunk_name,
        datafile_name=args.datafile_name if len(args.datafile_name)>0 else None,
        ss_version=args.ss_version,
        stride=args.stride,
        depth=args.depth,
        method=args.method,
        verbose=True
    )
    chunk_save_dir = os.path.join(SAVE_DIR, args.exp_name, args.chunk_name, args.ss_version)
    if not os.path.exists(os.path.join(SAVE_DIR, args.exp_name)):
        os.mkdir(os.path.join(SAVE_DIR, args.exp_name))
    if not os.path.exists(os.path.join(SAVE_DIR, args.exp_name, args.chunk_name)):
        os.mkdir(os.path.join(SAVE_DIR, args.exp_name, args.chunk_name))
    if not os.path.exists(chunk_save_dir):
        os.mkdir(chunk_save_dir)

    save_prefix = os.path.join(chunk_save_dir, args.ss_version)
    np.save(save_prefix + f'_{args.method}_stas.npy', stas)
    print(f"STAs saved to {save_prefix}_{args.method}_stas.npy")

    # print(f"Saving STAs in .sta format for {len(cell_ids)} cells...")
    # with STAWriter(filepath=save_prefix + '.sta') as wr:
    #     wr.write(sta=stas, ste=None, cluster_id=cell_ids, stixel_size=grid_size)
    # print(f"STAs saved to {save_prefix}.sta")