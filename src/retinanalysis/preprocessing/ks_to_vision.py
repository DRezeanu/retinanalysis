from __future__ import annotations
import bin2py as b2p
import xarray as xr
import visionwriter as vw
import visionloader as vl
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
from retinanalysis.classes.stim import StimBlock
from retinanalysis._config import config
from retinanalysis._database import schema
from retinanalysis.utils.datajoint_utils import (
    get_block_id_from_datafile,
    get_exp_summary,
)
from retinanalysis.utils.regen import (
    get_spatial_noise_frames,
    get_n_frames_spatial_noise
)
from subprocess import run as sp_run
from multiprocessing import cpu_count
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from warnings import warn

NOISE_PROTOCOLS = [
    'manookinlab.protocols.SpatialNoise',
    'manookinlab.protocols.FastNoise',
]
NUM_SAMPLES = 20000
SAMPLES_PER_MS = NUM_SAMPLES / 1e3

@dataclass(slots=True, frozen=True)
class RawDataContainer:
    array_id: int
    n_electrodes: int
    n_points: int
    electrode_data: np.ndarray | None
    ttl_data: np.ndarray
    epoch_starts: np.ndarray
    epoch_ends: np.ndarray


def load_raw_data(
    rawfile_location: str,
    chunk_samples: int = 100000,
    ttl_only: bool = False,
) -> RawDataContainer:

    with b2p.PyBinFileReader(rawfile_location, chunk_samples=chunk_samples) as pbfr:
        n_points = pbfr.length
        n_electrodes = pbfr.num_electrodes
        array_id = pbfr.array_id
        if ttl_only:
            ttl_data = pbfr.get_data_for_electrode(0, 0, n_points)
            electrode_data = None
        else:
            data = pbfr.get_data(0, n_points)
            ttl_data = data[:, 0]
            electrode_data = data[:, 1:]

    epoch_starts = np.array(
        [idx for idx, i in enumerate(np.diff(ttl_data)) if i<0]
    )
    epoch_ends = np.array(
        [idx for idx, i in enumerate(np.diff(ttl_data)) if i>0]
    )

    return RawDataContainer(
        array_id = array_id,
        n_electrodes=n_electrodes,
        n_points=n_points,
        electrode_data=electrode_data,
        ttl_data=ttl_data,
        epoch_starts=epoch_starts,
        epoch_ends=epoch_ends,
    )

def load_ks_data(ks_location: str, include_mua: bool = True):
    """
    Helper function that loads kilosort spike times and associates them with their cluster IDs.
    For now filters out all IDs listed as 'MUA' by Kilosort's 'cluster_group.tsv' output.

    Parameters:
        ks_location (str): path to kilosort output directory (e.g. '.../sorted/data000/kilosort2.5/')

        include_mua (bool): if true, will keep units that kilosort labeled as 'multi unit activity'
        instead of filtering them out.

    Returns:
        spike_dict (Dict[int, ndarray]): dictionary of cell_id : spike_time_array
    """
    spike_times = np.load(os.path.join(ks_location, "spike_times.npy"))
    cluster_ids = np.load(os.path.join(ks_location, "spike_clusters.npy"))

    if include_mua:
        good_ids = np.unique(cluster_ids)
    else:
        cluster_group = pd.read_csv(
            os.path.join(ks_location, "cluster_group.tsv"), delimiter="\t"
        )

        good_ids = cluster_group.query('KSLabel == "good"')["cluster_id"].to_list()

    spike_dict = {int(id + 1): spike_times[cluster_ids == id] for id in good_ids}

    return spike_dict


def get_chunk_datafiles(exp_name: str, chunk_name: str) -> list:
    exp_id = schema.Experiment() & {"exp_name": exp_name}
    if len(exp_id) != 1:
        raise ValueError(f"{len(exp_id)} exps found for given exp_name: {exp_name}")
    exp_id = exp_id.to_arrays("id")[0]
    chunk_id = schema.SortingChunk() & {
        "experiment_id": exp_id,
        "chunk_name": chunk_name,
    }
    chunk_id = chunk_id.to_arrays("id")[0]

    epoch_blocks = schema.EpochBlock() & {
        "experiment_id": exp_id,
        "chunk_id": chunk_id,
    }

    data_dirs = epoch_blocks.to_arrays("data_dir")
    datafile_names = [os.path.basename(path) for path in data_dirs]
    print(f'Found {chunk_name} datafiles: {datafile_names}')
    
    return datafile_names

def ks_datafile_to_vision(
    exp_name: str,
    datafile_name: str,
    output_dir: str,
    vision_path: str,
    raw_data_path: str | None = None,
    ks_data_path: str | None = None,
    ks_version: str = 'kilosort2.5',
    include_mua: bool = True,
    verbose: bool = True,
):
    """Function for generating vision files from the kilosort sorter outputs for an individual datafile.

    The function uses Vision.jar and vision_writer utilities to write .globals, .neurons, and .ei
    file types by default, with additional .params and .sta file types writen only for those datafiles
    that arise from spatial noise.

    Args:
        exp_name: name of the experiment (e.g. '20260506C')
        datafiles_name: name of datafile sorted by kilosort (e.g. 'data001')
        output_dir: directory where the output folders and files will be created
        vision_path: path to compiled Vision.jar program for EI calculation
        raw_data_path: path to raw data directory, default retinanalysis config.RAW_DIR
        ks_data_path: path to kilosort output directory where sorted data files live.
            Default is retinanalysis config.DATA_DIR
        ks_version: kilosort version used to do the sorting, default is 'kilosort2.5'
        include_mua: boolean value, when true will include units labeled 'multi-unit activity'
            by kilosort.
        verbose: when true, will print status messages to console.
    Returns:
        None. Vision files will be exported to the directory of interest.
    """
    if raw_data_path is None:
        raw_data_path = config.RAW_DIR
    if ks_data_path is None:
        ks_data_path = config.DATA_DIR

    raw_file_path = Path(raw_data_path) / exp_name / datafile_name
    ks_file_path = Path(ks_data_path) / exp_name / datafile_name / ks_version
    output_path = Path(output_dir) / exp_name / datafile_name

    raw_file_path.mkdir(parents=True, exist_ok=True)
    ks_file_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Using raw files from: {raw_file_path}")
        print(f"Using kilosort data from: {ks_file_path}")
        print(f"Path to Vision.jar: {vision_path}.")
        print(f"Vision files will be written to: {output_path}.")


    spike_dict = load_ks_data(str(ks_file_path), include_mua)

    # Load raw data
    raw_data = load_raw_data(str(raw_file_path), ttl_only=True)

    # Write neurons file
    with vw.NeuronsFileWriter(str(output_path), datafile_name) as nfw:
        nfw.write_neuron_file(spike_dict, raw_data.epoch_starts, raw_data.n_points)

    if verbose:
        print(f"\nNeurons file written to {output_path}")

    # Write ei
    n_cpus = cpu_count()

    # Use 60% of available CPUs
    available_cpus = int(n_cpus * 0.6)

    if verbose:
        print(f"\nUsing {available_cpus} CPUs for EI computation\n")

    # sp_run(
    #     f'java -Xmx8G -cp {vision_path} edu.ucsc.neurobiology.vision.calculations.CalculationManager "Electrophysiological Imaging Fast" {output_path} {raw_file_path} 0.01 67 133 1000000 {available_cpus}',
    #     shell=True,
    # )

    if _is_noise_datafile(exp_name, datafile_name):

        block_id = get_block_id_from_datafile(exp_name, datafile_name)
        stim_block = StimBlock(exp_name, block_id)
        n_epochs = len(stim_block.df_epochs)

        d_display = stim_block.d_display
        epoch_params = stim_block.df_epochs.loc[0, 'epoch_parameters']
        
        sta_width, sta_height = epoch_params['numXChecks'], epoch_params['numYChecks']
        microns_per_pixel = d_display['mu_per_pixel']
        
        pixels_per_stixel = int(round(d_display['n_wt'] / sta_width))
        microns_per_stixel = pixels_per_stixel * microns_per_pixel
        mean_frame_rate = d_display['mean_frame_rate']
        stage_frame_rate = d_display['stage_frame_rate']

        pre_time_s = stim_block.d_epoch_block_params['preTime']*1e-3
        pre_frames_1 = np.floor(pre_time_s*60)
        pre_frames_1 -= 1

        st_by_epoch = _align_spikes_by_epoch(
            spike_dict=spike_dict,
            raw_data=raw_data,
        )

        psth_xarr = _bin_spikes_by_frame(
            spike_dict = st_by_epoch,
            raw_data=raw_data,
            mean_frame_rate=mean_frame_rate,
        )

        for epoch in tqdm(range(n_epochs)):
            d_e_params = stim_block.df_epochs.loc[epoch, 'epoch_parameters']
            ls_unique_frames, ls_repeat_frames = get_n_frames_spatial_noise(stim_block.df_epochs) 

            d_meta = {
                "numXStixels": d_e_params["numXStixels"],
                "numYStixels": d_e_params["numYStixels"],
                "numXChecks": d_e_params["numXChecks"],
                "numYChecks": d_e_params["numYChecks"],
                "gridSizeUm": d_e_params["gridSize"],
                "chromaticClass": d_e_params["chromaticClass"],
                "unique_frames": ls_unique_frames[epoch],
                "repeat_frames": ls_repeat_frames[epoch],
                "stepsPerStixel": d_e_params["stepsPerStixel"],
                "seed": int(d_e_params["seed"]),
                "frameDwell": d_e_params["frameDwell"],
            }
            if "canvasSize" in d_e_params:
                d_meta["canvasSize"] = d_e_params["canvasSize"]
            else:
                canvas_size = (1140, 1824)
                print(
                    f"canvasSize not in epoch params and not provided, defaulting to {canvas_size}"
                )
                d_meta["canvasSize"] = canvas_size

            if "gaussianFilter" in d_e_params:
                d_meta["gaussianFilter"] = d_e_params["gaussianFilter"]
            if "filterSdStixels" in d_e_params:
                d_meta["filterSdStixels"] = d_e_params["filterSdStixels"]
            if "canvasSize" in d_e_params:
                # (x, y) to (rows, cols)
                d_meta["canvasSize"] = tuple(d_e_params["canvasSize"][::-1])
            if "micronsPerPixel" in d_e_params:
                d_meta["micronsPerPixel"] = d_e_params["micronsPerPixel"]
            if "repeating_seed" in d_e_params:
                d_meta["repeating_seed"] = int(d_e_params["repeating_seed"])

            e_frames, e_steps = get_spatial_noise_frames(**d_meta)


        print(e_frames.shape)

        stride = 2

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

        with vw.GlobalsFileWriter(str(output_path), datafile_name) as gfw:
            gfw.write_simplified_litke_array_globals_file(
                raw_data.array_id
                & 0xFFF,  # FIXME get rid of this after we figure out what happened with 120um
                0,
                0,
                "Kilosort converted",
                "",
                0,
                NUM_SAMPLES,
            )

            gfw.write_run_time_movie_params(runtime_movie_params)


    else:

        with vw.GlobalsFileWriter(str(output_path), datafile_name) as gfw:
            gfw.write_simplified_litke_array_globals_file(
                raw_data.array_id
                & 0xFFF,  # FIXME get rid of this after we figure out what happened with 120um
                0,
                0,
                "Kilosort converted",
                "",
                0,
                NUM_SAMPLES,
            )


def _align_spikes_by_epoch(
    spike_dict: dict[int, np.ndarray],
    raw_data: RawDataContainer,
) -> dict[int, np.ndarray]:

    raw_data = _validate_epoch_timing(raw_data)
    epoch_starts = raw_data.epoch_starts
    epoch_ends = raw_data.epoch_ends

    st_by_epoch = dict()
    for id in spike_dict:
        spike_times = spike_dict[id]
        times_by_epoch = []
        for idx, start in enumerate(epoch_starts):
            mask = (spike_times > start) & (spike_times < epoch_ends[idx])
            aligned = (spike_times[mask]-start)/SAMPLES_PER_MS
            times_by_epoch.append(np.round(aligned,2))

        st_by_epoch[id] = times_by_epoch

    return st_by_epoch

def ks_chunk_to_vision(
):
    return

def generate_vision_files(
    exp_name: str,
    output_dir: str,
    vision_path: str,
    chunk_name: str | None = None,
    datafile_names: list[str] | None = None,
    raw_path: str | None = None,
    sorted_path: str | None = None,
    ks_version: str = "kilosort2.5",
    include_mua: bool = True,
    verbose: bool = True,
):
    """
    Function for converting spike times and raw data for an MEA datafile into .neurons,
    .globals, and .ei files for vision.

    Parameters:
        exp_name (str): name of experiment (e.g. '20250713C')

        datafile_name (str): name of the datafile of interest (e.g. 'data005')

        output_path (str): path to output directory for the .neurons, .globals and .ei files

        vision_path (str): path to Vision.jar file for EI computation

        raw_path (str) Optional: path to raw data directory. Default is retinanalysis config.RAW_DIR

        sorted_path (str) Optional: path to sorted data directory. Default is retinanalysis config.DATA_DIR

        ks_version (str) Optional: kilosort version used for sorting. Default is 'kilosort2.5'.

        include_mua (bool) Optional: if true, will include units marked as 'MUA' or 'multi-unit activity'
        by kilosort. Default True

        verbose (bool) Optional: When true, will print status messages to console. Default True.

    Returns:
        None: No return values. The .neurons, .globals, and .ei files will be written to the
        specified output_path
    """
    if raw_path is None:
        raw_path = config.RAW_DIR

    if sorted_path is None:
        sorted_path = config.DATA_DIR

    if chunk_name is not None:
        if datafile_names is not None:
            raise ValueError('Cannot provide both chunk name and list of datafiles names')

        datafile_names = get_chunk_datafiles(
            exp_name=exp_name,
            chunk_name=chunk_name
        )
        chunk_output_path = os.path.join(output_dir, chunk_name)
    else:
        if datafile_names is None:
            raise ValueError('Must provide either chunk name or datafile names, got None for both')
        chunk_output_path = None


    rawfile_dir = os.path.join(raw_path, exp_name)
    ks_location = os.path.join(sorted_path, exp_name, datafile_name, ks_version)

    if verbose:
        print(f"\nRaw file location: {rawfile_dir}")
        print(f"Kilosort file location: {ks_location}")
        print(f"Path to Vision.jar: {vision_path}")
        print(f"Writing output to: {output_dir}")

    # Load spike times and units from Kilosort output
    spike_dict = load_ks_data(ks_location, include_mua)

    # Load raw data
    num_pts = 0
    ttl_triggers = []
    for datafile in datafile_names:
        rawfile_location = os.path.join(rawfile_dir, datafile)
        raw_data = load_raw_data(rawfile_location, ttl_only=True)
        
        num_pts += raw_data.n_points
        ttl_triggers += raw_data.epoch_starts

    ttl_triggers=np.array(ttl_triggers)

    # Write neurons file
    if chunk_output_path is not None:
        with vw.NeuronsFileWriter(chunk_output_path, datafile_name) as nfw:
            nfw.write_neuron_file(spike_dict, ttl_triggers, num_pts)

    if verbose:
        print(f"\nNeurons file written to {output_path}")

    # Write ei
    n_cpus = cpu_count()

    # Use 60% of available CPUs
    available_cpus = int(n_cpus * 0.6)

    if verbose:
        print(f"\nUsing {available_cpus} CPUs for EI computation\n")

    for datafile in datafile_names:
        rawfile_location = os.path.join(rawfile_dir, datafile)
        sp_run(
            f'java -Xmx8G -cp {vision_path} edu.ucsc.neurobiology.vision.calculations.CalculationManager "Electrophysiological Imaging Fast" {output_path} {rawfile_location} 0.01 67 133 1000000 {available_cpus}',
            shell=True,
        )


def _is_noise_datafile(
    exp_name: str,
    datafile_name: str,
) -> bool:

    exp_summary = get_exp_summary(exp_name)
    if exp_summary is None:
        raise ValueError(
            f"No summary found for experiment {exp_name}"
        )

    protocol_name = exp_summary.query('datafile_name == @datafile_name')['protocol_name'].item()
    if protocol_name in NOISE_PROTOCOLS:
        return True
    else:
        return False

def _bin_spikes_by_frame(
    spike_dict: dict[int, np.ndarray],
    raw_data: RawDataContainer,
    mean_frame_rate: float = 59.941548817817917,
) -> xr.DataArray:
    n_epochs = len(list(spike_dict.values())[0])
    epoch_starts = raw_data.epoch_starts
    epoch_ends = raw_data.epoch_ends
    epoch_length_ms = np.mean(epoch_ends - epoch_starts)/SAMPLES_PER_MS


    spike_time_arr = [sts for _, sts in spike_dict.items()]
    spike_time_arr = np.array(spike_time_arr, dtype=object)

    dims = ["cell_id", "epoch"]

    coords = {
        "cell_id": sorted(list(spike_dict.keys())),
        "epoch": np.arange(n_epochs),
    }


    spike_time_xarr = xr.DataArray(spike_time_arr, dims=dims, coords=coords)

    ms_per_frame = 1/mean_frame_rate*1e3
    bin_edges = np.arange(0, epoch_length_ms+ms_per_frame, ms_per_frame)
    n_bins = len(bin_edges)-1


    def apply_hist(arr, bin_edges):
        output, _ = np.histogram(arr, bin_edges)
        return output

    psth_xarr = xr.apply_ufunc(
        apply_hist,
        spike_time_xarr,
        kwargs={"bin_edges": bin_edges},
        input_core_dims=[[]],
        output_core_dims=[["bin"]],
        vectorize=True,
    )
    psth_xarr = psth_xarr.assign_coords({"bin": np.arange(0, n_bins)})
    psth_xarr = psth_xarr.assign_coords({"bin_edges": ("bin", bin_edges[:-1])})

    return psth_xarr

def _validate_epoch_timing(
    raw_data: RawDataContainer,
) -> RawDataContainer:
    epoch_starts = raw_data.epoch_starts
    epoch_ends = raw_data.epoch_ends

    if len(epoch_starts) == len(epoch_ends)+1:
        warn(
            "Data contains one extra epoch start, throwing away final epoch",
            category=UserWarning,
            stacklevel=2,
        )
        epoch_starts = epoch_starts[:-1]

        return RawDataContainer(
            array_id=raw_data.array_id,
            n_electrodes=raw_data.n_electrodes,
            n_points=raw_data.n_points,
            electrode_data=raw_data.electrode_data,
            ttl_data=raw_data.ttl_data[:epoch_ends[-1]],
            epoch_ends=epoch_ends,
            epoch_starts=epoch_starts[:-1],
        )

    if len(epoch_ends) > len(epoch_starts):
        raise ValueError(
            "More epoch ends than epoch starts, inspect ttl trace"
        )

    if len(epoch_starts) > len(epoch_ends)+1:
        raise ValueError(
            f"{len(epoch_starts)} epoch starts but only "
            f"{len(epoch_ends)} epoch ends. "
            "Inspect TTL trace"
        )

    return raw_data


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="ks_to_vision.py",
        description="Convert spike times and TTL data from Kilosort output to .neurons, .globals, and .ei Vision files",
    )

    # Positional arguments
    parser.add_argument("exp_name", help="experiment name, (e.g. 20260715C)", type=str)
    parser.add_argument(
        "datafile_name",
        help="name of datafile to convert to vision (e.g. data000)",
        type=str,
    )
    parser.add_argument(
        "output_path",
        help="path for output files (e.g. /Volumes/SSD/vision_output)",
        type=str,
    )
    parser.add_argument(
        "vision_path",
        help="path to Vision.jar file (e.g. .../MEA/src/Vision7_for_2015DAQ/Vision.jar)",
        type=str,
    )

    # Optional arguments with default values.
    # Note: If the flag is included without an argument, 'const' will be used. If the flag is
    # not included at all, 'default' will be used. Just an argparse quirk.
    parser.add_argument(
        "-r",
        "--raw",
        help="Path to raw data directory (e.g. .../Volumes/data/raw/)",
        nargs="?",
        default=config.RAW_DIR,
        const=config.RAW_DIR,
        type=str,
    )

    parser.add_argument(
        "-s",
        "--sorted",
        help="Path to KS sorter output directory (e.g. .../Volumes/data/sorted)",
        nargs="?",
        default=config.DATA_DIR,
        const=config.DATA_DIR,
        type=str,
    )

    parser.add_argument(
        "-k",
        "--ks_version",
        help="kilosort version used for spike sorting (e.g. kilosort2.5)",
        nargs="?",
        default="kilosort2.5",
        const="kilosort2.5",
        type=str,
    )

    # Boolean optionals.
    # Note: 'store_true' means that including the flag will set the value to true, but it's false by default
    # and 'store_false' means that including the flag will set the value to false, but it's true by default
    parser.add_argument(
        "-v", "--verbose", help="print status messages to console", action="store_true"
    )
    parser.add_argument(
        "-m",
        "--no_mua",
        help="exclude cells marked as MUA by kilosort",
        action="store_false",
    )

    args = parser.parse_args()

    exp_name = args.exp_name
    datafile_name = args.datafile_name
    output_path = args.output_path
    vision_path = args.vision_path
    raw_path = args.raw
    sorted_path = args.sorted
    ks_version = args.ks_version
    verbose = args.verbose
    include_mua = args.no_mua

    # Generate vision files
    generate_vision_files(
        exp_name=exp_name,
        output_dir=output_path,
        raw_path=raw_path,
        sorted_path=sorted_path,
        ks_version=ks_version,
        vision_path=vision_path,
        include_mua=include_mua,
        verbose=verbose,
    )
