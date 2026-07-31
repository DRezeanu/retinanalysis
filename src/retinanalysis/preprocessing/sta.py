import torch
import numpy as np
import tqdm.auto as tqdm
from typing import Optional, List
from retinanalysis import regen
import argparse
from retinanalysis._database import schema
from retinanalysis.classes import qc
from retinanalysis.utils.datajoint_utils import get_noise_name_by_exp
import os
from retinanalysis.classes.response import (
    MEAResponseBlock,
    MEAResponseGroup,
    create_mea_response_group,
)
from retinanalysis.classes.stim import MEAStimBlock, MEAStimGroup, create_mea_stim_group
import gc
from visionwriter import STAWriter, ParamsWriter, GlobalsFileWriter
import visionloader as vl
from retinanalysis.utils import ANALYSIS_DIR
from retinanalysis.preprocessing import rfs
import psutil


def _get_n_splits_memory(
    sd: torch.Tensor,
    br: torch.Tensor,
    stride: int,
    device: torch.device,
    n_max_usage: float = 0.6,
    method: str = "matmul",
    verbose: bool = True,
) -> int:
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
    if device.type == "cuda":
        # Get max memory GB from available GPU
        max_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    else:
        # cpu
        max_memory_gb = psutil.virtual_memory().available / 1e9

    max_memory_gb *= n_max_usage

    # Estimate total data size
    total_data_gb = (
        sd.element_size() * sd.nelement() * stride + br.element_size() * br.nelement()
    ) / 1e9
    n_splits = int(np.ceil(total_data_gb / max_memory_gb))
    if method == "conv":
        # Conv uses more memory, so adding some buffer splits
        n_splits *= 2
    elif method == 'matmul':
        n_splits += 1
    
    if verbose:
        print(
            f"Total data size: {total_data_gb:.1f}GB, max available: {max_memory_gb:.1f}GB"
        )
        print(f"Recommending {n_splits} splits of compute.")
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
        stride (int): Stride for upsampling stimulus data to match binned responses. If stim_data is already upsampled, set to 1.
        method (str): "matmul" or "conv"

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
        stim_data, binned_responses, stride, device, method=method, verbose=verbose
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
            sd = stim_data[:, :, s_start:s_end].to(device)
            # Upsample by stride
            sd = torch.repeat_interleave(sd, stride, dim=1)

            with torch.no_grad():
                for lag in tqdm.tqdm(lags, desc="STA depth"):
                    br_lag = binned_responses[:, :, lag:]
                    sd_lag = sd[:, : n_bins - lag, :]

                    # binned spikes [N, K, T] @ stim [N, T, S] = [N, K, S]
                    epoch_stas = torch.bmm(br_lag, sd_lag)

                    # Avg across epochs for [K, S]
                    stas[:, lag, s_start:s_end] += epoch_stas.mean(axis=0).cpu()

            # Clear memory
            del sd, br_lag, sd_lag, epoch_stas
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
            sd = stim_data[:, :, s_start:s_end].to(device)
            # Upsample sd by stride
            sd = torch.repeat_interleave(sd, stride, dim=1)
            # permute to [N, S, T]
            sd = sd.permute(0, 2, 1)

            # Pad stim data on left with depth-1 zeros
            # so [N, S, T+depth-1]
            sd = torch.nn.functional.pad(sd, (depth - 1, 0))

            # [N, 1, 1, S, T+depth-1]
            sd = sd.unsqueeze(1).unsqueeze(1)

            # batch over [N], conv ([1, 1, S, T+D-1], [K, 1, 1, T]) -> [N, 1, K, S, D]
            with torch.no_grad():
                epoch_stas = batched_conv(sd, br, padding="valid")

            # Remove singleton and swap last two dims -> [N, K, D, S]
            epoch_stas = epoch_stas.squeeze(1)
            epoch_stas = epoch_stas.transpose(2, 3)

            # Avg across epochs for [K, D, S]
            stas[:, :, s_start:s_end] = epoch_stas.mean(axis=0).cpu()

            del sd, epoch_stas
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
    chunk_name: str,
    ss_version: str = "kilosort2.5",
    datafile_name: Optional[list] = None,
    verbose: bool = True,
) -> tuple[MEAStimGroup, MEAResponseGroup]:
    # If datafile name(s) not given, get noise datafiles
    if datafile_name is None:
        datafile_name = get_noise_datafiles(exp_name, chunk_name)
        if verbose:
            print(f"Found noise datafile(s): {datafile_name}")

    sg = create_mea_stim_group(exp_name, datafile_name, verbose=verbose)
    rg = create_mea_response_group(
        exp_name, datafile_name, ss_version, b_load_fd=True, verbose=verbose
    )
    return sg, rg


def get_data_for_chunk(
    sg: Optional[MEAStimGroup] = None,
    rg: Optional[MEAResponseGroup] = None,
    exp_name: Optional[str] = None,
    chunk_name: Optional[str] = None,
    ss_version: str = "kilosort2.5",
    datafile_name: Optional[list] = None,
    verbose: bool = True,
) -> dict:
    if sg is None or rg is None:
        sg, rg = get_stim_response_groups(
            exp_name, chunk_name, ss_version, datafile_name, verbose
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

def compute_stas_from_scratch(
    exp_name: str,
    datafile_names: str | list[str],
    ss_version: str = 'kilosort2.5',
):
    return

def compute_stas_for_chunk(
    sg: Optional[MEAStimGroup] = None,
    rg: Optional[MEAResponseGroup] = None,
    exp_name: Optional[str] = None,
    chunk_name: Optional[str] = None,
    ss_version: str = "kilosort2.5",
    datafile_name: Optional[list] = None,
    stride: int = 2,
    depth: int = 60,
    method: str = "conv",
    verbose: bool = True,
) -> dict:
    if sg is None or rg is None:
        sg, rg = get_stim_response_groups(
            exp_name, chunk_name, ss_version, datafile_name, verbose
        )

    # STA input gen and calc loop
    # TODO initialize stas with max n_cells across blocks, and keep track of cell idx to add for each block.
    stas = None
    total_sps = None
    # Number of epochs to process in batch
    n_epochs_batch = 4
    n_blocks = len(sg.ls_blocks)
    for i in range(n_blocks):
        sb = sg.ls_blocks[i]
        rb = rg.ls_blocks[i]

        # Get number of frames (assuming same across epochs)
        ls_unique_frames, ls_repeat_frames = regen.get_n_frames_spatial_noise(
            sb.df_epochs
        )
        total_frames = np.array(ls_unique_frames) + np.array(ls_repeat_frames)
        if len(np.unique(total_frames)) != 1:
            raise ValueError(f"Uneven number of frames across epochs! {total_frames}")
        n_frames = total_frames[0]

        # Bin spike times
        rb.bin_spike_times_by_frames(stride=stride)
        # [K, N, T]
        bs = rb.binned_spikes
        # Make [N, K, T]
        bs = bs.transpose(1, 0, 2)

        # Crop out pre frames
        pre_time_s = rb.d_timing["pre_time_ms"] / 1e3

        stage_frame_rate = rb.d_timing["stage_frame_rate"]
        # Count n frames where state.time (1/fr steps) is < pre_time_s
        # pre_frames = len(np.arange(0, pre_time_s, 1 / stage_frame_rate))

        # MM sets visibility using state.time, but the actual checkerboard is shown using
        # state.frame and a pre_frame offset (preF) calculated by taking floor(pre_time/1000 * 60)
        pre_frames = np.floor(pre_time_s * 60).astype(int)

        # LCR CORRECTION
        pre_frames -= 1
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
        n_batches = int(np.ceil(n_epochs / n_epochs_batch))
        for j in tqdm.tqdm(np.arange(n_batches), desc="Epoch batch"):
            e_start = j * n_epochs_batch
            e_end = (j + 1) * n_epochs_batch
            if e_end > n_epochs:
                e_end = n_epochs

            # Regen stim
            sb.regenerate_stimulus(ls_epochs=list(range(e_start, e_end)))
            # [N, T, H, W, C]
            stim_frames = sb.stim_data["frames"]

            # [N, K, T]
            resp_data = bs[e_start:e_end]

            if stas is None:
                stas = compute_stas(
                    stim_frames,
                    resp_data,
                    depth=depth,
                    stride=stride,
                    method=method,
                    verbose=verbose,
                )

            else:
                stas += compute_stas(
                    stim_frames,
                    resp_data,
                    depth=depth,
                    stride=stride,
                    method=method,
                    verbose=verbose,
                )

            del stim_frames, resp_data, sb.stim_data
            gc.collect()

    # Final normalize by abs max for each cell
    peaks = np.abs(stas).max(axis=(1,2,3,4), keepdims=True)
    # Avoid div by 0
    peaks[peaks==0] = 1
    stas = stas / peaks

    grid_size = sb.df_epochs.at[0, "epoch_parameters"]["gridSize"]

    d_output = {
        "stas": stas,
        "cell_ids": rg.cell_ids,
        "grid_size": grid_size,
        # Total spikes that went into STA calc. Should be <= overall spike counts bc of grey periods.
        "sta_n_sps": np.squeeze(total_sps),
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
    exp_name: Optional[str] = None,
    chunk_name: Optional[str] = None,
    ss_version: str = "kilosort2.5",
    data_dir: Optional[str] = None,
    data_name: str = "kilosort2.5",
    analysis_dir: str = ANALYSIS_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        exp_name (str, optional): _description_. Defaults to None.
        chunk_name (str, optional): _description_. Defaults to None.
        ss_version (str, optional): _description_. Defaults to 'kilosort2.5'.
        data_dir (str, optional): _description_. Defaults to None.
        data_name (str, optional): _description_. Defaults to 'kilosort2.5'.

    Raises:
        ValueError: _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    # Either exp_name and chunk_name, or data_dir must be provided
    if exp_name is not None and chunk_name is not None:
        if data_dir is not None:
            raise ValueError("Provide either exp_name and chunk_name, or data_dir!")

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
):
    # TODO get array ID and nsamples from raw .bin files or schema?
    array_id = 504
    num_samples = 20000  # placeholder

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
            array_id
            & 0xFFF,  # FIXME get rid of this after we figure out what happened with 120um
            0,
            0,
            "Kilosort converted",
            "",
            0,
            num_samples,
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

    # if not os.path.exists(os.path.join(SAVE_DIR, args.exp_name)):
    #     os.mkdir(os.path.join(SAVE_DIR, args.exp_name))
    # if not os.path.exists(os.path.join(SAVE_DIR, args.exp_name, args.chunk_name)):
    #     os.mkdir(os.path.join(SAVE_DIR, args.exp_name, args.chunk_name))
    # if not os.path.exists(chunk_save_dir):
    #     os.mkdir(chunk_save_dir)

    if not os.path.exists(chunk_save_dir):
        os.makedirs(chunk_save_dir)

    save_prefix = os.path.join(chunk_save_dir, args.ss_version)
    save_np = save_prefix + f"_{args.method}_stas.npy"
    save_vcd_sta = save_prefix + ".sta"

    nas_vcd_sta = os.path.join(
        ANALYSIS_DIR,
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
                load_dir = ANALYSIS_DIR
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

        print(f"Saving STAs in .sta format for {len(d_stas['cell_ids'])} cells...")
        with STAWriter(filepath=save_vcd_sta) as wr:
            wr.write(
                # Reverse time to match vision convention
                sta=d_stas['stas'][:, ::-1], 
                ste=None, cluster_id=d_stas['cell_ids'], 
                stixel_size=d_stas['grid_size']
                )
        print(f"STAs saved to {save_vcd_sta}")

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
