import bin2py as b2p
import visionwriter as vw
import numpy as np
import argparse
import os
from retinanalysis import (RAW_DIR,
                           DATA_DIR)
from subprocess import run as sp_run
from multiprocessing import cpu_count
import pandas as pd

def load_raw_data(rawfile_location: str, chunk_samples: int = 100000):
    """
    Helper function that loads raw data from a file location using bin2py.

    Parameters:
        rawfile_location (str): path to folder containing all raw data (e.g. '.../data/raw/data000')

        chunk_samples (int): number of samples to load per chunk

    Returns:
        num_pts (int): number of datapoints in the full concatenated raw trace

        ttl_triggers (List[int]): A list of ttl trigger times pulled from electrode 0 of raw litke data.
    """
    with b2p.PyBinFileReader(rawfile_location, chunk_samples = chunk_samples) as pbfr:
        num_pts = pbfr.length
        data = pbfr.get_data(0, num_pts)
        trigger_data = data[:, 0]

    ttl_triggers = np.array([idx for idx, i in enumerate(np.diff(trigger_data)) if i < 0])

    return num_pts, ttl_triggers 

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
    spike_times = np.load(os.path.join(ks_location, 'spike_times.npy'))
    cluster_ids = np.load(os.path.join(ks_location, 'spike_clusters.npy'))
    cluster_group = pd.read_csv(os.path.join(ks_location, 'cluster_group.tsv'), delimiter ='\t')

    if include_mua:
        good_ids = np.unique(cluster_ids)
    else:
        good_ids = cluster_group.query('KSLabel == "good"')['cluster_id'].to_list()

    spike_dict = {int(id+1) : spike_times[cluster_ids == id] for id in good_ids}
    
    return spike_dict

def generate_vision_files(exp_name: str, datafile_name: str, output_path: str,
                          vision_path: str, raw_path: str = RAW_DIR,
                          sorted_path: str = DATA_DIR, ks_version: str = 'kilosort2.5', 
                          include_mua: bool = True, verbose: bool = True):
    """
    Function for converting spike times and raw data for an MEA datafile into .neurons,
    .globals, and .ei files for vision.

    Parameters:
        exp_name (str): name of experiment (e.g. '20250713C')

        datafile_name (str): name of the datafile of interest (e.g. 'data005')

        output_path (str): path to output directory for the .neurons, .globals and .ei files

        vision_path (str): path to Vision.jar file for EI computation

        raw_path (str) Optional: path to raw data directory. Default is retinanalysis RAW_DIR

        sorted_path (str) Optional: path to sorted data directory. Default is retinanalysis DATA_DIR

        ks_version (str) Optional: kilosort version used for sorting. Default is 'kilosort2.5'.

        include_mua (bool) Optional: if true, will include units marked as 'MUA' or 'multi-unit activity'
        by kilosort. Default True

        verbose (bool) Optional: When true, will print status messages to console. Default True.

    Returns:
        None: No return values. The .neurons, .globals, and .ei files will be written to the 
        specified output_path
    """

    rawfile_location = os.path.join(raw_path, exp_name, datafile_name)
    ks_location = os.path.join(sorted_path, exp_name, datafile_name, ks_version)
    output_path = os.path.join(output_path, datafile_name)

    if verbose:
        print(f'\nRaw file location: {rawfile_location}')
        print(f'Kilosort file location: {ks_location}')
        print(f'Path to Vision.jar: {vision_path}')
        print(f'Writing output to: {output_path}')

    # Load spike times and units from Kilosort output
    spike_dict = load_ks_data(ks_location, include_mua)   

    # Load raw data
    num_pts, ttl_triggers = load_raw_data(rawfile_location)

    # Write neurons file:
    with vw.NeuronsFileWriter(output_path, datafile_name) as nfw:
        nfw.write_neuron_file(spike_dict, ttl_triggers, num_pts)

    if verbose:
        print(f'\nNeurons file written to {output_path}')

    # Write globals file
    with vw.GlobalsFileWriter(output_path, datafile_name) as gfw:
        gfw.write_simplified_litke_array_globals_file(504, 0, 0, 'no comment', 'no identifier', 0, num_pts)

    if verbose:
        print(f'\nGlobals file written to {output_path}')

    # Write ei
    n_cpus = cpu_count()

    # Use 60% of available CPUs
    available_cpus = int(n_cpus*0.6)

    if verbose:
        print(f'\nUsing {available_cpus} CPUs for EI computation\n')

    sp_run(f'java -Xmx8G -cp {vision_path} edu.ucsc.neurobiology.vision.calculations.CalculationManager "Electrophysiological Imaging Fast" {output_path} {rawfile_location} 0.01 67 133 1000000 {available_cpus}', shell = True)


if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(prog='ks_to_vision.py',
                                     description='Convert spike times and TTL data from Kilosort output to .neurons, .globals, and .ei Vision files')

    # Positional arguments
    parser.add_argument('exp_name', help='experiment name, (e.g. 20260715C)', type=str)
    parser.add_argument('datafile_name', help='name of datafile to convert to vision (e.g. data000)', type=str)
    parser.add_argument('output_path', help='path for output files (e.g. /Volumes/SSD/vision_output)', type=str)
    parser.add_argument('vision_path', help='path to Vision.jar file (e.g. .../MEA/src/Vision7_for_2015DAQ/Vision.jar)', type=str)

    # Optional arguments with default values.
    # Note: If the flag is included without an argument, 'const' will be used. If the flag is
    # not included at all, 'default' will be used. Just an argparse quirk.
    parser.add_argument('-r', '--raw', help='Path to raw data directory (e.g. .../Volumes/data/raw/)',
                        nargs = '?', default = RAW_DIR, const = RAW_DIR, type=str)

    parser.add_argument('-s', '--sorted', help='Path to KS sorter output directory (e.g. .../Volumes/data/sorted)',
                        nargs = '?', default = DATA_DIR, const = DATA_DIR, type=str)

    parser.add_argument('-k', '--ks_version', help='kilosort version used for spike sorting (e.g. kilosort2.5)',
                        nargs = '?', default = 'kilosort2.5', const = 'kilosort2.5', type=str)

    # Boolean optionals.
    # Note: 'store_true' means that including the flag will set the value to true, but it's false by default
    # and 'store_false' means that including the flag will set the value to false, but it's true by default
    parser.add_argument('-v', '--verbose', help='print status messages to console', action = 'store_true')
    parser.add_argument('-m', '--no_mua', help='exclude cells marked as MUA by kilosort', action = 'store_false')

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
    generate_vision_files(exp_name = exp_name, datafile_name = datafile_name,
                          output_path = output_path, raw_path = raw_path,
                          sorted_path = sorted_path, ks_version = ks_version,
                          vision_path = vision_path, include_mua = include_mua,
                          verbose = verbose)
