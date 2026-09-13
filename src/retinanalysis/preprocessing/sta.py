import torch
import numpy as np
import tqdm.auto as tqdm
import argparse
from retinanalysis._database import schema
from retinanalysis.classes import qc
from retinanalysis.utils.datajoint_utils import get_noise_name_by_exp
import os
from retinanalysis.classes.response import (
    MEAResponseGroup,
    create_mea_response_group,
)
from retinanalysis.classes.stim import MEAStimGroup, create_mea_stim_group
import gc
from visionwriter import STAWriter, ParamsWriter, GlobalsFileWriter
import visionloader as vl
from retinanalysis._config import config
from retinanalysis.preprocessing import rfs
import psutil
from pathlib import Path

# cudnn and mkldnn may allocate an im2col buffer that's backend-dependent.
# A named module-level constant keeps this visible and tunable. 
# If you start getting torch.OutOfMemoryError, tune this up, which will 
# increase the assumed 'cost per dim'. If conv is slow, you can tune this
# down. 1.2 is a good balance
CONV_WORKSPACE_FACTOR = 1.2


def _get_n_splits_memory(
    stim_data: torch.Tensor,
    binned_response: torch.Tensor,
    device: torch.device,
    stride: int = 2,
    depth: int = 60,
    max_usage_frac: float = 0.6,
    method: str = "matmul",
    verbose: bool = True,
) -> int:
    """
    Get number of splits for STA compute.

    Args:
        stim_data (torch.Tensor): regenerated stimulus, nd array of shape
            (N epochs, T frames, S stim dimensions)
        binned_response (torch.Tensor): binned spike respones, nd array with one row per epoch
        stride (int): Stride for upsampling stimulus data to match binned responses.
        device (torch.device): pytorch device (cpu, cuda, or mps, though mps not yet functional
            for our purposes here)
        depth (int): window size for the sta (i.e. how many time bins we use)
        max_usage_frac (float, optional): Allowed memory fraction. Defaults to 0.6.
        method (str): "matmul" or "conv". Default is matmul because it's approximately 2x faster. 
            Both are GPU optimized but conv computes full cross corr through conv2d with a kernel
            as long as the response, which is a bad shape for cudnn, and its peak allocation is 
            much larger than matmul per stimulus dimension. Thus, it splits more and pays the
            per-split cost many times. On a sample dataset, matmul took 50s to conv's 100s.
        verbose (bool, optional): Print status messages to the console. Defaults to True.

    Returns:
        int: Number of splits.
    """

    n_epochs, n_cells, n_bins = binned_response.shape
    n_frames = stim_data.shape[1]
    n_bins_up = n_frames * stride
    n_stim_dims = np.prod(stim_data.shape[2:])
    bytes_per_float = stim_data.element_size()

    if device.type == 'cuda':
        free_bytes = torch.cuda.mem_get_info()[0] # free, not total memory
    else:
        free_bytes = psutil.virtual_memory().available

    budget = free_bytes * max_usage_frac

    budget = budget - (bytes_per_float * n_epochs * n_cells * n_bins)

    if budget <= 0:
        raise MemoryError(
            f'Insufficient memory budget: {budget/1e9}Gb'
        )

    if method == 'conv':
        stage_upsample = n_epochs * (n_frames + n_bins_up)
        stage_pad = n_epochs * (2 * n_bins_up + depth - 1)
        stage_conv = n_epochs * (n_bins_up + depth - 1 + n_cells * depth)
        stage_mean = (n_epochs + 1) * n_cells * depth
        cost_per_dim = bytes_per_float * max(stage_upsample, stage_pad, 
                                             stage_conv, stage_mean)
        cost_per_dim *= CONV_WORKSPACE_FACTOR
    else:
        stage_upsample = n_epochs * (n_frames + n_bins_up)
        stage_bmm = n_epochs * n_bins_up + (n_epochs + 1) * n_cells
        cost_per_dim = bytes_per_float * max(stage_upsample, stage_bmm)

    max_dims_per_split = int(np.floor(budget/cost_per_dim))
    if max_dims_per_split < 1:
        raise MemoryError(
            'Insufficient memory. Cost per dim is too high for '
            'current memory budget\n'
            f'    -Cost per dim: {cost_per_dim/1e6:.2f}Mb\n'
            f'    -Memory budget: {budget/1e9:.2f}Gb\n'
        )

    n_splits = int(np.ceil(n_stim_dims/max_dims_per_split))

    if verbose:
        print(
            'Memory splitting output:\n'
            f'    - Memory budget: {budget/1e9:.2f}Gb\n'
            f'    - Cost per dim: {cost_per_dim/1e6:.2f}Mb\n'
            f'    - Dims per split: {max_dims_per_split}\n'
            f'    - Splits: {n_splits}\n'
        )

    return n_splits

def compute_stas(
    stim_data_np: np.ndarray,
    binned_responses_np: np.ndarray,
    depth: int = 60,
    stride: int = 2,
    method: str = "matmul",
    verbose: bool = True,
) -> np.ndarray:
    """
    Workhorse compute method for STA
    Can use for EI as well, where instead of frames you have raw data samples

    Args:
        stim_data_np (np.ndarray):
            Stimulus data of shape [N epochs, T frames, *]
            For noise stim, last dims are [H, W, C]
            For EI, last dims are [C] electrodes
        binned_responses_np (np.ndarray):
            Binned spikerate/spikecount of shape [N epochs, K cells, T frames]
        stride (int): Stride for upsampling stimulus data to match binned responses.
            If stim_data is already upsampled, set to 1.
        method (str): "matmul" or "conv". Default is matmul because it's approximately 2x faster. 
            Both are GPU optimized but conv computes full cross corr through conv2d with a kernel
            as long as the response, which is a bad shape for cudnn, and its peak allocation is 
            much larger than matmul per stimulus dimension. Thus, it splits more and pays the
            per-split cost many times. On a sample dataset, matmul took 50s to conv's 100s.

    Returns:
        stas (np.ndarray):
            [K cells, D depth, *]
            For noise stim, [H, W, C]
            For EI, [C]
    """
    # [N epochs, T frames, H, W, C]
    stim_data = torch.from_numpy(stim_data_np).float()
    # [N epochs, K cells, T frames]
    binned_responses = torch.from_numpy(binned_responses_np).float()

    stim_dims = stim_data.shape[2:]
    n_stim_dims = np.prod(stim_dims)
    n_epochs, n_cells, n_bins = binned_responses.shape

    n_frames = stim_data.shape[1]
    stim_data = stim_data.reshape(n_epochs, n_frames, -1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == 'cpu' and method == 'conv':
        if verbose:
            print("Conv method is optimized for GPU. Falling back to matmul for CPU.")
        method = 'matmul'

    n_splits = _get_n_splits_memory(
        stim_data = stim_data,
        binned_response = binned_responses,
        device = device,
        stride = stride,
        method = method,
        depth=depth,
        verbose=verbose,
    )
    n_split_sz = int(np.ceil(n_stim_dims / n_splits))
    stas = torch.zeros(n_cells, depth, n_stim_dims, dtype=torch.float32)

    binned_responses = binned_responses.to(device)

    if method == "matmul":
        lags = np.arange(depth)
        for i in tqdm.tqdm(np.arange(n_splits), desc="STA compute chunk"):
            s_start = i * n_split_sz
            s_end = (i + 1) * n_split_sz
            if s_end > n_stim_dims:
                s_end = n_stim_dims

            # Put stim data chunk on device
            e_stim_data = stim_data[:, :, s_start:s_end].to(device)
            # Upsample by stride
            e_stim_data = torch.repeat_interleave(e_stim_data, stride, dim=1)

            with torch.no_grad():
                for lag in tqdm.tqdm(lags, desc="STA depth"):
                    br_lag = binned_responses[:, :, lag:]
                    sd_lag = e_stim_data[:, : n_bins - lag, :]

                    # binned spikes [N, K, T] @ stim [N, T, S] = [N, K, S]
                    epoch_stas = torch.bmm(br_lag, sd_lag)

                    # Avg across epochs for [K, S]
                    stas[:, lag, s_start:s_end] += epoch_stas.mean(axis=0).cpu()

            # Clear memory
            del e_stim_data, br_lag, sd_lag, epoch_stas
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # Reverse time dim for standard convention
        stas = torch.flip(stas, dims=[1])

    elif method == "conv":
        # The shapes here can get confusing. Key to note is:
        # Using conv2d so that (H, W) of input can be filled by (stim_dims, time+depth-1)
        # convolving that with (K, 1, time) responses gives (K, stim_dims, depth) output.
        # Singleton dims added where needed. Batching over epochs with vmap.

        # [N, K, 1, 1, T]
        br = binned_responses.unsqueeze(2).unsqueeze(2)

        batched_conv = torch.vmap(torch.nn.functional.conv2d)
        for i in tqdm.tqdm(np.arange(n_splits), desc="STA compute chunk"):
            s_start = i * n_split_sz
            s_end = (i + 1) * n_split_sz
            if s_end > n_stim_dims:
                s_end = n_stim_dims

            # [N, T, S]
            e_stim_data = stim_data[:, :, s_start:s_end].to(device)
            # Upsample sd by stride
            e_stim_data = torch.repeat_interleave(e_stim_data, stride, dim=1)
            # permute to [N, S, T]
            e_stim_data = e_stim_data.permute(0, 2, 1)

            # Pad stim data on left with depth-1 zeros
            # so [N, S, T+depth-1]
            e_stim_data = torch.nn.functional.pad(e_stim_data, (depth - 1, 0))

            # [N, 1, 1, S, T+depth-1]
            e_stim_data = e_stim_data.unsqueeze(1).unsqueeze(1)

            # batch over [N], conv ([1, 1, S, T+D-1], [K, 1, 1, T]) -> [N, 1, K, S, D]
            with torch.no_grad():
                epoch_stas = batched_conv(e_stim_data, br, padding="valid")

            # Remove singleton and swap last two dims -> [N, K, D, S]
            epoch_stas = epoch_stas.squeeze(1)
            epoch_stas = epoch_stas.transpose(2, 3)

            # Avg across epochs for [K, D, S]
            stas[:, :, s_start:s_end] = epoch_stas.mean(axis=0).cpu()

            del e_stim_data, epoch_stas
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    else:
        raise ValueError("Method must be either matmul or conv!")

    # Reshape back to full stim dims
    stas = stas.reshape(n_cells, depth, *stim_dims)
    stas = stas.numpy()
    
    # Final clean up
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return stas


def get_noise_datafiles(exp_name: str, chunk_name: str) -> list:
    exp_id = schema.Experiment() & {"exp_name": exp_name}
    if len(exp_id) != 1:
        raise ValueError(f"{len(exp_id)} exps found for given exp_name: {exp_name}")
    exp_id = exp_id.to_arrays("id")[0]
    chunk_id = schema.SortingChunk() & {
        "experiment_id": exp_id,
        "chunk_name": chunk_name,
    }
    chunk_id = chunk_id.to_arrays("id")[0]
    noise_protocol = get_noise_name_by_exp(exp_name)
    protocol_id = schema.Protocol() & {"name": noise_protocol}
    protocol_id = protocol_id.to_arrays("protocol_id")[0]

    epoch_blocks = schema.EpochBlock() & {
        "experiment_id": exp_id,
        "chunk_id": chunk_id,
        "protocol_id": protocol_id,
    }

    noise_data_dirs = epoch_blocks.to_arrays("data_dir")
    datafile_name = [os.path.basename(path) for path in noise_data_dirs]
    print(f'Found noise datafiles: {datafile_name}')
    
    # Filter out zero contrast datafiles
    eb_contrasts = epoch_blocks.proj(contrast="parameters->>'$.contrast'").fetch('contrast').astype(float)
    non_zero_contrast = np.where(eb_contrasts != 0)[0]
    
    if len(non_zero_contrast) == 0:
        raise ValueError("No non-zero contrast datafiles found for given exp and chunk!")
    elif len(non_zero_contrast) < len(datafile_name):
        datafile_name = [datafile_name[i] for i in non_zero_contrast]
        print(f'Filtered out zero contrast datafiles, remaining: {datafile_name}')

    return datafile_name


def get_stim_response_groups(
    exp_name: str,
    chunk_name: str | None = None,
    datafile_name: list | str | np.ndarray | None = None,
    ss_version: str = "kilosort2.5",
    verbose: bool = True,
) -> tuple[MEAStimGroup, MEAResponseGroup]:

    if isinstance(datafile_name, str):
        datafile_name = [datafile_name]
    elif isinstance(datafile_name, np.ndarray):
        datafile_name = list(datafile_name)

    # If datafile name(s) not given, get noise datafiles
    if datafile_name is None:
        if chunk_name is None:
            raise ValueError(
                f'Must provide a datafile_name(s) or chunk_name'
            )
        else:
            datafile_name = get_noise_datafiles(exp_name, chunk_name)
            if verbose:
                print(f"Found noise datafile(s): {datafile_name}")

    sg = create_mea_stim_group(exp_name, datafile_name, verbose=verbose)
    rg = create_mea_response_group(
        exp_name, datafile_name, ss_version, b_load_fd=True, verbose=verbose
    )
    return sg, rg


def get_data_for_chunk(
    exp_name: str | None = None,
    chunk_name: str | None = None,
    datafile_name: list | str | np.ndarray | None = None,
    sg: MEAStimGroup | None = None,
    rg: MEAResponseGroup | None = None,
    ss_version: str = "kilosort2.5",
    verbose: bool = True,
) -> dict:

    if isinstance(datafile_name, str):
        datafile_name = [datafile_name]
    elif isinstance(datafile_name, np.ndarray):
        datafile_name = list(datafile_name)

    if sg is None or rg is None:
        if exp_name is None:
            raise ValueError(
                'Must provide one of the following pairs:\n'
                '    - stim_group + response_group\n'
                '    - exp_name + (chunk_name or datafile_name(s))'
            )
        if chunk_name is None and datafile_name is None:
            raise ValueError(
                'Must provide one of the following pairs:\n'
                '    - stim_group + response_group\n'
                '    - exp_name + (chunk_name or datafile_name(s))'
            )
        else:
            sg, rg = get_stim_response_groups(
                exp_name=exp_name,
                chunk_name=chunk_name,
                datafile_name=datafile_name,
                ss_version=ss_version,
                verbose = verbose,
            )

    # Collect spike counts
    spike_counts = qc.get_nsps(rg, rg.cell_ids)
    spike_counts = np.array(spike_counts)

    # Collect ISIs for saving in .params file
    isi_dt = 0.5  # ms
    isi_bin_edges = np.arange(0, 300, isi_dt)
    d_isi = qc.get_isi(rg, rg.cell_ids, isi_bin_edges)
    # Convert to array
    isi_array = np.zeros((len(rg.cell_ids), len(isi_bin_edges) - 1))
    for i, cell_id in enumerate(rg.cell_ids):
        isi_array[i, :] = d_isi[cell_id]

    d_output = {
        "sg": sg,
        "rg": rg,
        "cell_ids": rg.cell_ids,
        "spike_counts": spike_counts,
        "isi": isi_array,
        "isi_bin_edges": isi_bin_edges,
        "isi_dt": isi_dt,
    }

    return d_output

def compute_stas_for_chunk(
    exp_name: str | None = None,
    chunk_name: str | None = None,
    datafile_name: list | str | np.ndarray | None = None,
    sg: MEAStimGroup | None = None,
    rg: MEAResponseGroup | None= None,
    ss_version: str = "kilosort2.5",
    stride: int = 2,
    depth: int = 60,
    method: str = "matmul",
    max_epochs_per_batch: int = 4,
    verbose: bool = True,
) -> dict:

    if sg is None or rg is None:
        if exp_name is None and (chunk_name is None or datafile_name is None):
            raise ValueError(
                'Must provide one of the following parirs:\n'
                '    - stim_group + response_group\n'
                '    - exp_name + (chunk_name or datafile_name(s))'
            )

        assert exp_name is not None
        sg, rg = get_stim_response_groups(
            exp_name=exp_name,
            chunk_name=chunk_name,
            datafile_name=datafile_name,
            ss_version=ss_version,
            verbose=verbose,
        )

    # STA input gen and calc loop
    stas = None
    total_sps = np.zeros(len(rg.cell_ids))
    n_epochs_total = 0

    n_blocks = len(sg.ls_blocks)
    print(f'Processing {n_blocks} blocks')
    for i in range(n_blocks):
        stim_block = sg.ls_blocks[i]
        response_block = rg.ls_blocks[i]

        # Bin spike times
        response_block.bin_spike_times_by_frames(stride=stride)

        # [Cell, Epoch, TimeBin]
        binned_spikes = response_block.binned_spikes
        if binned_spikes is None:
            raise ValueError(
                f'Could not bin spikes for {exp_name} block {sg.ls_blocks[i].block_id}.'
            )

        row_of = {cid: r for r, cid in enumerate(response_block.df_spike_times['cell_id'])}
        rows = [row_of[cid] for cid in rg.cell_ids]

        n_epochs = len(stim_block.df_epochs)
        # Make [Epoch, Cell, TimeBin]
        if isinstance(binned_spikes, np.ndarray):
            binned_spikes = binned_spikes[rows]
            block_sps = np.sum(binned_spikes, axis = (1,2))

            binned_spikes = binned_spikes.transpose(1, 0, 2)
            n_batches = int(np.ceil(n_epochs / max_epochs_per_batch))
            e_starts = [j * max_epochs_per_batch for j in range(n_batches)]
            e_ends = [min((j+1) * max_epochs_per_batch,n_epochs) for j in range(n_batches)] 

            # When all epochs are the same lengths, we build the batches as contiguous sets
            # of epochs
            chunks = [list(range(e_starts[i],e_ends[i])) for i in range(n_batches)]
            batches = [(idx, binned_spikes[idx]) for idx in chunks]
            
        else:
            binned_spikes = [binned_spikes[r] for r in rows]
            block_sps = np.array([sum(e.sum() for e in cell_rows) for cell_rows in binned_spikes])
            # Get a list of unique Time dimension lengths. By definition all
            # Cells inside an epoch will have the same number of time bins so 
            # we don't need to iterate over every cell.
            unique_lengths = {len(e_spikes) for e_spikes in binned_spikes[0]}
            all_lengths = [len(e_spikes) for e_spikes in binned_spikes[0]]
            time_groups = []
            group_indices = []
            for length in sorted(unique_lengths):
                # Mask by ragged T dim
                mask = [al == length for al in all_lengths]
                group_idx = [idx for idx, val in enumerate(mask) if val]
                group_indices.append(group_idx)
                # Group by time domain and arrange as Epoch, Cell, TimeBin (replaces transpose above)
                group = np.asarray([[c_spikes[j] for c_spikes in binned_spikes] for j in group_idx])
                # append to spike_groups list
                time_groups.append(group)


            binned_spikes = time_groups

            # When all epochs are not the same length, we build the batches as lists
            # of epochs with the same number of time bins. If any of the lists is 
            # greater than max_epochs_per_batch, we split those up
            batches = []
            for g_idx, time_group in enumerate(binned_spikes):
                group_idx = group_indices[g_idx]
                epochs_per_group = time_group.shape[0]
                n_batches = int(np.ceil(epochs_per_group / max_epochs_per_batch))
                for j in range(n_batches):
                    e_start = j * max_epochs_per_batch
                    e_end = min((j+1) * max_epochs_per_batch, epochs_per_group)
                    batch = group_idx[e_start:e_end]
                    spikes = time_group[e_start:e_end]
                    batches.append((batch, spikes))

        total_sps += block_sps

        # Loop across epochs in batch
        for batch, resp_data in tqdm.tqdm(batches, desc="Epoch batch"):
            # Regen stim
            stim_block.regenerate_stimulus(ls_epochs=batch)
            
            # Check that regen worked
            if stim_block.stim_data is None:
                raise ValueError(
                    'Unable to regenerate stimulus for '
                    f'{stim_block.exp_name} block {stim_block.block_id}'
                )

            # [N, T, H, W, C]
            stim_frames = stim_block.stim_data["frames"]

            # Check how many epochs actually in this batch (last batch likely
            # less than max_epochs_per_batch)
            n_epochs_in_batch = resp_data.shape[0]

            if stim_frames.shape[1] * stride != resp_data.shape[2]:
                raise ValueError(
                    "Stimulus and spike array time dims don't match for epochs"
                    f"{batch} in {stim_block.exp_name} block {stim_block.block_id}."
                )

            # Compute batch stas, weighted by the number of epochs in that batch
            # without this, batches with fewer epochs will be overweighted
            batch_stas = compute_stas(
                stim_data_np = stim_frames,
                binned_responses_np = resp_data,
                depth=depth,
                stride=stride,
                method=method,
                verbose=verbose,
            ) * n_epochs_in_batch

            if stas is None:
                stas = batch_stas
            else:
                stas += batch_stas

            n_epochs_total += n_epochs_in_batch

            del stim_frames, resp_data, stim_block.stim_data
            gc.collect()

    if stas is None:
        raise ValueError(
            f'Unable to compute STAs for {sg.exp_name} datafiles {sg.datafile_names}'
        )
    # Strictly not necessary because of the peak normalization that follows
    stas /= n_epochs_total
    # Final normalize by abs max for each cell
    peaks = np.abs(stas).max(axis=(1,2,3,4), keepdims=True)
    # Avoid div by 0
    peaks[peaks==0] = 1
    stas = stas / peaks

    grid_size = sg.ls_blocks[0].df_epochs.at[0, "epoch_parameters"]["gridSize"]

    d_output = {
        "stas": stas,
        "cell_ids": rg.cell_ids,
        "grid_size": grid_size,
        # Total spikes that went into STA calc. Should be <= overall spike counts bc of grey periods.
        "sta_n_sps": total_sps,
    }

    return d_output


def load_stas_from_vcd(vcd: vl.VisionCellDataTable, cell_ids: np.ndarray) -> np.ndarray:
    # Collect STAs for each cell
    stas = []
    for cell_id in cell_ids:
        # [H, W, D] for each channel
        vcd_sta = vcd.get_sta_for_cell(cell_id)
        vcd_sta = [vcd_sta.red, vcd_sta.green, vcd_sta.blue]

        # Make [D, H, W, C]
        vcd_sta = np.stack(vcd_sta, axis=-1).transpose(2, 0, 1, 3)
        stas.append(vcd_sta)

    # Final [K, D, H, W, C]
    stas = np.stack(stas, axis=0)
    print(f"Collected STAs into array of shape {stas.shape} for {len(cell_ids)} cells.")
    return stas


def load_stas_from_vl(
    exp_name: str | None = None,
    chunk_name: str | None = None,
    ss_version: str = "kilosort2.5",
    data_dir: str | None = None,
    data_name: str = "kilosort2.5",
    analysis_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        exp_name (str, optional): _description_. Defaults to None.
        chunk_name (str, optional): _description_. Defaults to None.
        ss_version (str, optional): _description_. Defaults to 'kilosort2.5'.
        data_dir (str, optional): _description_. Defaults to None.
        data_name (str, optional): _description_. Defaults to 'kilosort2.5'.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_

    Raises:
        ValueError: _description_
    """

    if analysis_dir is None:
        analysis_dir = config.ANALYSIS_DIR

    # Either exp_name and chunk_name, or data_dir must be provided
    elif exp_name is not None and chunk_name is not None and data_dir is not None:
        raise ValueError(
            "Provide either exp_name and chunk_name, OR a data_dir, "
            "not all three."
        )


    if (exp_name is None or chunk_name is None):
        if data_dir is None:
            raise ValueError(
                "Provide either exp_name and chunk_name, or a data_dir."
            )
    else:
        data_dir = os.path.join(analysis_dir, exp_name, chunk_name, ss_version)
        data_name = ss_version

    print(
        f"Loading STAs from VisionLoader with data_dir={data_dir} and data_name={data_name}..."
    )
    vcd = vl.load_vision_data(data_dir, data_name, include_sta=True)
    print(f"Loaded, collecting into array.")
    # Get sorted cell IDs
    cell_ids = np.array(vcd.get_cell_ids())
    cell_ids = np.sort(cell_ids)

    stas = load_stas_from_vcd(vcd, cell_ids)

    return stas, cell_ids

def write_sta_file(
    d_stas: dict,
    save_dir: str | Path,
    ss_version: str,
):
    out_file = os.path.join(save_dir, f"{ss_version}.sta")

    print(f"Saving STAs in .sta format for {len(d_stas['cell_ids'])} cells...")
    with STAWriter(filepath=out_file) as wr:
        wr.write(
            # Reverse time to match vision convention
            sta=d_stas['stas'][:, ::-1], 
            ste=None, cluster_id=d_stas['cell_ids'], 
            stixel_size=d_stas['grid_size']
            )
    print(f"STAs saved to {out_file}")

def write_params_file(sta_height, d_data, d_rf_params, save_dir, ss_version):
    out_file = os.path.join(save_dir, f"{ss_version}.params")

    print(f"Saving RF params and ISI data to {out_file}...")
    with ParamsWriter(filepath=out_file, cluster_id=d_data["cell_ids"]) as wr:
        wr.write(
            timecourse_matrix=d_rf_params["timecourses"],
            isi=d_data["isi"],
            spike_count=d_data["spike_counts"],
            x0=d_rf_params["col_coords"],
            # Flip y0 to match vision convention of top left origin
            y0=sta_height - d_rf_params["row_coords"],
            # matlab_style_gauss2D applies sigma_r along rows (y) and sigma_c along cols (x)
            sigma_x=d_rf_params["c_col_sigmas"],
            sigma_y=d_rf_params["c_row_sigmas"],
            theta=d_rf_params["thetas"],
            isi_binning=d_data["isi_dt"],
        )
    print(f"Saved to {out_file}")


def write_globals_file(
    globals_path: str,
    globals_name: str,
    microns_per_pixel: float,
    display_width_pixels: int,
    sta_width: int,
    sta_height: int,
    mean_frame_rate: float,
    stride: int,
    array_id: int = 504,
    num_samples: int = 20000,
):

    pixels_per_stixel = int(round(display_width_pixels / sta_width))
    microns_per_stixel = pixels_per_stixel * microns_per_pixel

    refreshPeriod = (
        1000.0 / mean_frame_rate / float(stride)
    )  # STA refresh period in msec.
    runtime_movie_params = vl.RunTimeMovieParamsReader(
        pixelsPerStixelX=pixels_per_stixel,
        pixelsPerStixelY=pixels_per_stixel,
        width=sta_width,
        height=sta_height,
        micronsPerStixelX=microns_per_stixel,
        micronsPerStixelY=microns_per_stixel,
        xOffset=0.0,
        yOffset=0.0,
        interval=int(stride),  # MM: same as stride, I think
        monitorFrequency=mean_frame_rate,
        framesPerTTL=1,
        refreshPeriod=refreshPeriod,
        nFramesRequired=-1,  # MM: No idea what this means..
        droppedFrames=[],
    )

    with GlobalsFileWriter(globals_path, globals_name) as gfw:
        gfw.write_simplified_litke_array_globals_file(
            array_id=array_id
            & 0xFFF,  # FIXME get rid of this after we figure out what happened with 120um
            base_time=0,
            seconds_time=0,
            comment="Kilosort converted",
            dataset_identifier="",
            dformat=0,
            n_samples=num_samples,
        )
        gfw.write_run_time_movie_params(runtime_movie_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sta.py", description="Compute STAs for given experiment and chunk."
    )
    parser.add_argument("exp_name", help="Experiment name (e.g. 20260715C)")
    parser.add_argument("chunk_name", help="Chunk name (e.g. chunk1)")
    parser.add_argument(
        "datafile_name",
        nargs="*",
        help="Datafile name(s) to process (e.g. data000). If not given, will attempt to find noise datafiles for the given exp and chunk.",
    )
    parser.add_argument("save_dir", help="Directory to save computed STAs")
    parser.add_argument(
        "recompute_stas",
        type=bool,
        default=False,
        help="Whether to recompute STAs if they already exist. If False, will load existing STAs and re-run RF fitting.",
    )
    parser.add_argument(
        "--ss_version",
        default="kilosort2.5",
        help="Spike sorting version to load responses from (default: kilosort2.5)",
    )
    parser.add_argument("--stride", type=int, default=2, help="Stride (bins/frame)")
    parser.add_argument("--depth", type=int, default=61, help="STA depth (bins)")
    parser.add_argument(
        "--method",
        type=str,
        default="matmul",
        help="Method for computing STAs (default: matmul)",
    )

    args = parser.parse_args()

    SAVE_DIR = args.save_dir

    chunk_save_dir = os.path.join(
        SAVE_DIR, args.exp_name, args.chunk_name, args.ss_version
    )

    if not os.path.exists(chunk_save_dir):
        os.makedirs(chunk_save_dir)

    save_prefix = os.path.join(chunk_save_dir, args.ss_version)
    save_np = save_prefix + f"_{args.method}_stas.npy"
    save_vcd_sta = save_prefix + ".sta"

    nas_vcd_sta = os.path.join(
        config.ANALYSIS_DIR,
        args.exp_name,
        args.chunk_name,
        args.ss_version,
        args.ss_version + ".sta",
    )
    print(nas_vcd_sta)

    d_data = get_data_for_chunk(
        exp_name=args.exp_name,
        chunk_name=args.chunk_name,
        ss_version=args.ss_version,
        datafile_name=args.datafile_name if len(args.datafile_name) > 0 else None,
        verbose=True,
    )

    exists_somewhere = (
        os.path.exists(save_np)
        or os.path.exists(save_vcd_sta)
        or os.path.exists(nas_vcd_sta)
    )
    if not args.recompute_stas and exists_somewhere:
        print(f"STA files already exist!")
        print("Will load saved STAs and re-run RF param fitting.")
        if os.path.exists(save_vcd_sta) or os.path.exists(nas_vcd_sta):
            if os.path.exists(save_vcd_sta):
                load_dir = SAVE_DIR
            elif os.path.exists(nas_vcd_sta):
                load_dir = config.ANALYSIS_DIR
            stas, cell_ids = load_stas_from_vl(
                exp_name=args.exp_name,
                chunk_name=args.chunk_name,
                ss_version=args.ss_version,
                analysis_dir=load_dir,
            )
            # Check that cell ids match those in d_data
            if not np.array_equal(cell_ids, d_data["cell_ids"]):
                raise ValueError(
                    "Cell IDs from VisionLoader do not match those in response group!"
                )
        else:
            print(f"Loading STAs from {save_np}...")
            stas = np.load(save_np)
            print(f"STAs loaded from {save_np}!")

    else:
        print(
            f"Computing STAs for {args.exp_name} {args.chunk_name} with method {args.method}..."
        )
        d_stas = compute_stas_for_chunk(
            sg=d_data["sg"],
            rg=d_data["rg"],
            stride=args.stride,
            depth=args.depth,
            method=args.method,
            verbose=True,
        )

        np.save(save_np, d_stas["stas"])
        print(f"STAs saved to {save_np}")

        write_sta_file(
        d_stas=d_stas,
        save_dir=chunk_save_dir,
        ss_version=args.ss_version)

        stas = d_stas["stas"]

    d_rf_params = rfs.rf_fitting_pipeline(stas=stas, str_output_dir=chunk_save_dir)

    # Mark bottom 10% of spike count cells as lowSNR
    spike_counts = d_data["spike_counts"]
    threshold = np.percentile(spike_counts, 10)
    low_idxs = spike_counts <= threshold
    auto_types = d_rf_params["auto_types"]
    auto_types[low_idxs] = "lowSNR"

    # Prepend 'All/'
    auto_types = np.array(["All/" + t for t in auto_types])

    # Save auto_types to .txt file. rows of {cell ID}  {type}
    auto_types_file = os.path.join(chunk_save_dir, f"auto_types.txt")
    type_data = np.array(list(zip(d_data["cell_ids"], auto_types)), dtype=object)
    np.savetxt(auto_types_file, type_data, fmt="%s", delimiter="\t")
    print(f"Auto types saved to {auto_types_file}")

    sta_height, sta_width = stas.shape[2], stas.shape[3]

    # Save RF params with ISIs in .params file
    write_params_file(sta_height, d_data, d_rf_params, chunk_save_dir, args.ss_version)

    # Save .globals file
    d_display = d_data["sg"].ls_blocks[0].d_display
    write_globals_file(
        globals_path=chunk_save_dir,
        globals_name=args.ss_version,
        microns_per_pixel=d_display["mu_per_pixel"],
        display_width_pixels=d_display["n_wt"],
        sta_width=sta_width,
        sta_height=sta_height,
        mean_frame_rate=d_display["mean_frame_rate"],
        stride=args.stride,
    )
