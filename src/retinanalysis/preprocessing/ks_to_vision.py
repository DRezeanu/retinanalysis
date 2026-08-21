from __future__ import annotations
import xarray as xr
import visionwriter as vw
import numpy as np
import argparse
import os
from retinanalysis._config import config
from retinanalysis._database import schema
from retinanalysis.utils.datajoint_utils import get_exp_summary
from subprocess import run as sp_run
from multiprocessing import cpu_count
import pandas as pd
from pathlib import Path
from warnings import warn
from .raw_data_loader import load_raw_data, RawDataContainer
from .sta import (
    get_data_for_chunk,
    compute_stas_for_chunk,
    write_sta_file,
    write_globals_file,
    write_params_file,
)
from .rfs import rf_fitting_pipeline
from .ei_merge import merge_eis

NOISE_PROTOCOLS = [
    'manookinlab.protocols.SpatialNoise',
    'manookinlab.protocols.FastNoise',
]
NUM_SAMPLES = 20000
SAMPLES_PER_MS = NUM_SAMPLES / 1e3

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
        raise ValueError(f"{len(exp_id)} experimentss found for given exp_name: {exp_name}")
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


def ks_chunk_to_vision(
    exp_name: str,
    chunk_name: str,
    output_dir: str,
    vision_path: str | None = None,
    raw_data_dir: str | None = None,
    ks_data_dir: str | None = None,
    ks_version: str = 'kilosort2.5',
    include_mua: bool = True,
    compute_datafile_stas: bool = False,
    overwrite_existing: bool = False,
    verbose: bool = True,
):
    """Function for generating vision files from the kilosort sorter outputs for a sorting chunk.

    The function uses Vision.jar and vision_writer utilities to write .globals, .neurons, and .ei
    file types by default, with additional .params and .sta file types writen only if the chunk
    arises from spatial noise.

    Args:
        exp_name: name of the experiment (e.g. '20260506C')
        chunk_name: name of chunk that contains sorted kilosort files (e.g. 'chunk1')
        output_dir: directory where the output folders and files will be created
        vision_path: path to compiled Vision.jar program for EI calculation
        raw_data_dir: path to raw data directory, default retinanalysis config.RAW_DIR
        ks_data_dir: path to kilosort output directory where sorted data files live.
            Default is retinanalysis config.DATA_DIR
        ks_version: kilosort version used to do the sorting, default is 'kilosort2.5'
        include_mua: boolean value, when true will include units labeled 'multi-unit activity'
            by kilosort.
        compute_datafile_stas: boolean value, when true will compute .sta and .params files
            for individual datafiles in addition to the chunk as a whole.
        overwrite_existing: boolean value, if true will overwrite existing chunk .ei, .sta,
            and .params file. Default false.
        verbose: when true, will print status messages to console.

    Returns:
        None. Vision files will be exported to the directory of interest.
    """
    experiments = schema.Experiment() & {"exp_name": exp_name}
    if len(experiments) != 1:
        raise ValueError(f"{len(experiments)} experiments found for given exp_name: {exp_name}")

    if raw_data_dir is None:
        raw_data_dir = config.RAW_DIR
    if ks_data_dir is None:
        ks_data_dir = config.DATA_DIR
    if vision_path is None:
        vision_path = config.VISION_PATH

    chunk_datafiles = get_chunk_datafiles(
                exp_name = exp_name,
                chunk_name = chunk_name,
    )

    raw_file_paths = [Path(raw_data_dir)/exp_name/datafile for datafile in chunk_datafiles]
    ks_chunk_path = Path(ks_data_dir) / exp_name / chunk_name / ks_version
    chunk_output_path = Path(output_dir) / exp_name / chunk_name / ks_version

    chunk_output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"***Creating Vision Files for {exp_name} {chunk_name}...***")
        print(f"Using kilosort chunk data from: {ks_chunk_path}\n")
        print(f"Path to Vision.jar: {vision_path}\n")
        print(f"Chunk's Vision files will be written to: {chunk_output_path}\n")


    chunk_spike_dict = load_ks_data(str(ks_chunk_path), include_mua)
    ls_raw_data = [load_raw_data(raw_file, ttl_only=True) for raw_file in raw_file_paths]

    
    all_epoch_starts = []
    n_samples = 0
    for raw_data in ls_raw_data:
        all_epoch_starts += (raw_data.epoch_starts + n_samples).tolist()
        n_samples += raw_data.n_samples

    all_epoch_starts = np.array(all_epoch_starts)

    # Write .globals, .neurons and .ei files for each datafile in turn
    # If compute_datafile_stas = True, compute .sta and .params files
    # for those datafiles that were for noise runs

    # for datafile in chunk_datafiles:
    #     if verbose:
    #         print(f'***Creating Vision Files for {exp_name} {datafile}...***')
    #     ks_datafile_to_vision(
    #         exp_name=exp_name,
    #         datafile_name=datafile,
    #         output_dir = output_dir,
    #         vision_path=vision_path,
    #         raw_data_dir=raw_data_dir,
    #         ks_data_dir = ks_data_dir,
    #         ks_version = ks_version,
    #         include_mua = include_mua,
    #         compute_sta = compute_datafile_stas,
    #         verbose=verbose,
    #     )

    # Write neurons file for chunk
    with vw.NeuronsFileWriter(str(chunk_output_path), ks_version) as nfw:
        nfw.write_neuron_file(chunk_spike_dict, all_epoch_starts, n_samples)

    if verbose:
        print(f"Neurons file written to {chunk_output_path}\n")

    # Merge datafile EIs into chunk EI
    merge_eis(
        exp_name=exp_name,
        chunk_name=chunk_name,
        datafiles=chunk_datafiles,
        sorted_dir = output_dir,
        output_dir = output_dir,
        ss_version=ks_version,
        overwrite=overwrite_existing,
        verbose=verbose)

    # If this is a noise chunk, write sta, params, globals, and runtime movie params
    if _is_noise_data(
        exp_name=exp_name,
        data_folder=chunk_name,
    ):
        try:
            # Create a temporary preprocessing config profile
            # So stim_group and response_group are made from 
            # the newly created neurons file.
            config.create_profile(
                name='preprocessing',
                profile_paths={
                    'analysis': str(output_dir),
                    'data': str(output_dir),
                    'h5' : config.H5_DIR,
                    'raw' : config.RAW_DIR,
                    'meta' : config.META_DIR,
                    'tags' : config.TAGS_DIR,
                    'vision': config.VISION_PATH,
                    'user': config.USER,
                },
                overwrite=True,
            )
            config.set_profile('preprocessing')


            d_data = get_data_for_chunk(
                exp_name=exp_name,
                ss_version=ks_version,
                chunk_name=chunk_name,
                verbose=True,
            )

            sta_dict = compute_stas_for_chunk(
                sg = d_data['sg'],
                rg=d_data['rg'],
                ss_version=ks_version,
            )

            stas = sta_dict['stas']
            rf_dict = rf_fitting_pipeline(stas, str(chunk_output_path))

            sta_height, sta_width = stas.shape[2], stas.shape[3]

            write_sta_file(
                d_stas=sta_dict,
                save_dir=str(chunk_output_path),
                ss_version=ks_version,
            )

            # Save RF params with ISIs in .params file
            write_params_file(
                sta_height=sta_height,
                d_data=d_data,
                d_rf_params=rf_dict,
                save_dir=str(chunk_output_path),
                ss_version=ks_version,
            )

            # Save .globals file
            d_display = d_data["sg"].ls_blocks[0].d_display
            write_globals_file(
                globals_path=str(chunk_output_path),
                globals_name=ks_version,
                microns_per_pixel=d_display["mu_per_pixel"],
                display_width_pixels=d_display["n_wt"],
                sta_width=sta_width,
                sta_height=sta_height,
                mean_frame_rate=d_display["mean_frame_rate"],
                stride=2,
                array_id=ls_raw_data[0].array_id,
                num_samples=NUM_SAMPLES,
            )

            config.reset()
            config.remove_profile('preprocessing')
            if verbose:
                print(f'Wrote .neurons, .globals, .ei, .sta, and .params files to {chunk_output_path}\n')

        except Exception as e:
            config.reset()
            config.remove_profile('preprocessing')
            print(
                f"Unable to create .sta, .params, and .globals file for {chunk_name}.\n"
                f"Error: {e}"
            )

    else:
        with vw.GlobalsFileWriter(str(chunk_output_path), ks_version) as gfw:
            gfw.write_simplified_litke_array_globals_file(
                array_id=ls_raw_data[0].array_id
                & 0xFFF,  # FIXME get rid of this after we figure out what happened with 120um
                base_time=0,
                seconds_time=0,
                comment="Kilosort converted",
                dataset_identifier="",
                dformat=0,
                n_samples=NUM_SAMPLES,
            )

        if verbose:
            print(f'Wrote .neurons, .globals, and .ei files to {chunk_output_path}\n')

def ks_datafile_to_vision(
    exp_name: str,
    datafile_name: str,
    output_dir: str,
    vision_path: str | None = None,
    raw_data_dir: str | None = None,
    ks_data_dir: str | None = None,
    ks_version: str = 'kilosort2.5',
    include_mua: bool = True,
    compute_sta: bool = False,
    verbose: bool = True,
):
    """Function for generating vision files from the kilosort sorter outputs for an individual datafile.

    The function uses Vision.jar and vision_writer utilities to write .globals, .neurons, and .ei
    file types by default, with additional .params and .sta file types writen only for those datafiles
    that arise from spatial noise.

    Args:
        exp_name: name of the experiment (e.g. '20260506C')
        datafile_name: name of datafile folder that contains sorted kilosort files (e.g. 'data001')
        output_dir: directory where the output folders and files will be created
        vision_path: path to compiled Vision.jar program for EI calculation
        raw_data_dir: path to raw data directory, default retinanalysis config.RAW_DIR
        ks_data_dir: path to kilosort output directory where sorted data files live.
            Default is retinanalysis config.DATA_DIR
        ks_version: kilosort version used to do the sorting, default is 'kilosort2.5'
        include_mua: boolean value, when true will include units labeled 'multi-unit activity'
            by kilosort.
        compute_sta: boolean value, when true, STA and Params will be computed and written to
            disk. Default False.
        verbose: when true, will print status messages to console.

    Returns:
        None. Vision files will be exported to the directory of interest.
    """

    exp_summary = get_exp_summary(exp_name)
    if exp_summary is None:
        raise ValueError(
            f"Experiment {exp_name} is not yet in the database. Parse the h5 and "
            "run retinanalysis.populate_database() before creating vision files."
        )

    if raw_data_dir is None:
        raw_data_dir = config.RAW_DIR
    if ks_data_dir is None:
        ks_data_dir = config.DATA_DIR
    if vision_path is None:
        vision_path = config.VISION_PATH

    raw_file_path = Path(raw_data_dir) / exp_name / datafile_name
    ks_file_path = Path(ks_data_dir) / exp_name / datafile_name / ks_version
    output_path = Path(output_dir) / exp_name / datafile_name / ks_version

    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Using raw files from: {raw_file_path}\n")
        print(f"Using kilosort data from: {ks_file_path}\n")
        print(f"Path to Vision.jar: {vision_path}\n")
        print(f"Vision files will be written to: {output_path}\n")

    spike_dict = load_ks_data(str(ks_file_path), include_mua)

    # Load raw data
    raw_data = load_raw_data(str(raw_file_path), ttl_only=True)

    # Write neurons file
    with vw.NeuronsFileWriter(str(output_path), ks_version) as nfw:
        nfw.write_neuron_file(spike_dict, raw_data.epoch_starts, raw_data.n_samples)

    if verbose:
        print(f"Neurons file written to {output_path}\n")

    # Write ei
    n_cpus = cpu_count()

    # Use 60% of available CPUs
    available_cpus = int(n_cpus * 0.8)

    if verbose:
        print(f"Using {available_cpus} of {n_cpus} CPUs for EI computation\n")

    sp_run(
        f'java -Xmx8G -cp {vision_path} edu.ucsc.neurobiology.vision.calculations.CalculationManager "Electrophysiological Imaging Fast" {output_path} {raw_file_path} 0.01 67 133 1000000 {available_cpus}',
        shell=True,
    )

    with vw.GlobalsFileWriter(str(output_path), datafile_name) as gfw:
        gfw.write_simplified_litke_array_globals_file(
            array_id=raw_data.array_id
            & 0xFFF,  # FIXME get rid of this after we figure out what happened with 120um
            base_time=0,
            seconds_time=0,
            comment="Kilosort converted",
            dataset_identifier="",
            dformat=0,
            n_samples=NUM_SAMPLES,
        )

    neurons_filepath = output_path / f'{ks_version}.neurons'
    ei_filepath = output_path / f'{ks_version}.ei'
    os.rename(neurons_filepath, Path(output_path)/f'{datafile_name}.neurons')
    os.rename(ei_filepath, Path(output_path)/f'{datafile_name}.ei')

    if _is_noise_data(exp_name, datafile_name) and compute_sta:

        try:
            # Create a temporary preprocessing config profile
            # So stim_group and response_group are made from 
            # the newly created neurons file.
            config.create_profile(
                name='preprocessing',
                profile_paths={
                    'analysis': str(output_dir),
                    'data': str(output_dir),
                    'h5' : config.H5_DIR,
                    'raw' : config.RAW_DIR,
                    'meta' : config.META_DIR,
                    'tags' : config.TAGS_DIR,
                    'vision': config.VISION_PATH,
                    'user': config.USER,
                },
                overwrite=True,
            )
            config.set_profile('preprocessing')


            d_data = get_data_for_chunk(
                exp_name=exp_name,
                ss_version=ks_version,
                datafile_name=datafile_name,
                verbose=True,
            )

            sta_dict = compute_stas_for_chunk(
                sg = d_data['sg'],
                rg=d_data['rg'],
                ss_version=ks_version,
            )

            stas = sta_dict['stas']
            rf_dict = rf_fitting_pipeline(stas, str(output_path))

            sta_height, sta_width = stas.shape[2], stas.shape[3]

            write_sta_file(
                d_stas=sta_dict,
                save_dir=str(output_path),
                ss_version=ks_version,
            )

            # Save RF params with ISIs in .params file
            write_params_file(
                sta_height=sta_height,
                d_data=d_data,
                d_rf_params=rf_dict,
                save_dir=str(output_path),
                ss_version=ks_version,
            )

            # Save .globals file
            d_display = d_data["sg"].ls_blocks[0].d_display
            write_globals_file(
                globals_path=str(output_path),
                globals_name=ks_version,
                microns_per_pixel=d_display["mu_per_pixel"],
                display_width_pixels=d_display["n_wt"],
                sta_width=sta_width,
                sta_height=sta_height,
                mean_frame_rate=d_display["mean_frame_rate"],
                stride=2,
                array_id=raw_data.array_id,
                num_samples=NUM_SAMPLES,
            )

            # remove the preprocessing profile
            config.reset()
            config.remove_profile('preprocessing')

            globals_filepath = output_path / f'{ks_version}.globals'
            params_filepath = output_path / f'{ks_version}.params'
            sta_filepath = output_path / f'{ks_version}.sta'

            os.rename(globals_filepath, Path(output_path)/f'{datafile_name}.globals')
            os.rename(params_filepath, Path(output_path)/f'{datafile_name}.params')
            os.rename(sta_filepath, Path(output_path)/f'{datafile_name}.sta')

            if verbose:
                print(f'Wrote .neurons, .globals, .ei, .sta, and .params files to {output_path}\n')

        except Exception as e:
            # remove the preprocessing profile
            config.reset()
            config.remove_profile('preprocessing')
            
            print(
                f"Unable to create .sta, .params, and .globals file for {datafile_name}.\n"
                f"Error: {e}"
            )

    else:

        if verbose:
            print(f'Wrote .neurons, .globals, and .ei files to {output_path}')

def _is_noise_data(
    exp_name: str,
    data_folder: str,
) -> bool:

    exp_summary = get_exp_summary(exp_name)
    if exp_summary is None:
        raise ValueError(
            f"No summary found for experiment {exp_name}"
        )

    if data_folder.startswith('data'):
        protocol_name = exp_summary.query('datafile_name == @data_folder')['protocol_name'].item()
        if protocol_name in NOISE_PROTOCOLS:
            return True
        else:
            return False
    else:
        protocol_names = exp_summary.query('chunk_name == @data_folder')['protocol_name'].to_list()
        return any(p for p in protocol_names if p in NOISE_PROTOCOLS)

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


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="ks_to_vision.py",
        description="Convert spike times and TTL data from Kilosort output to "
        ".neurons, .globals, and .ei Vision files. If noise data, can also create "
        ".sta and .params files.",
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


