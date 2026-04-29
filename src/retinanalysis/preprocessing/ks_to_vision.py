import bin2py as b2p
import visionwriter as vw
import numpy as np
import argparse
import os
from retinanalysis import (RAW_DIR,
                           DATA_DIR)
from subprocess import run as sp_run
from multiprocessing import cpu_count

def load_raw_data(rawfile_location, chunk_samples = 100000):
    with b2p.PyBinFileReader(rawfile_location, chunk_samples = chunk_samples) as pbfr:
        num_pts = pbfr.length
        data = pbfr.get_data(0, num_pts)
        trigger_data = data[:, 0]

    ttl_triggers = np.array([idx for idx, i in enumerate(np.diff(trigger_data)) if i < 0])

    return num_pts, ttl_triggers 

def load_ks_data(ks_location):
    spike_times = np.load(os.path.join(ks_location, 'spike_times.npy'))
    cluster_ids = np.load(os.path.join(ks_location, 'spike_clusters.npy'))
    unique_ids = np.unique(cluster_ids)

    spike_dict = {int(id+1) : spike_times[cluster_ids == id] for id in unique_ids}
    
    return spike_dict

def generate_vision_files(exp_name: str, datafile_name: str, output_path: str,
                          vision_path: str, raw_path: str = RAW_DIR,
                          sorted_path: str = DATA_DIR, ks_version: str = 'kilosort2.5', 
                          verbose: bool = True):

    rawfile_location = os.path.join(raw_path, exp_name, datafile_name)
    ks_location = os.path.join(sorted_path, exp_name, datafile_name, ks_version)
    output_path = os.path.join(output_path, datafile_name)

    if verbose:
        print(f'\nRaw file location: {rawfile_location}')
        print(f'Kilosort file location: {ks_location}')
        print(f'Writing output to: {output_path}')

    # Load spike times and units from Kilosort output
    spike_dict = load_ks_data(ks_location)   

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

    # User 60% of available CPUs
    available_cpus = int(n_cpus*0.6)

    if verbose:
        print(f'\nUsing {available_cpus} CPUs for EI computation\n')

    sp_run(f'java -Xmx8G -cp {vision_path} edu.ucsc.neurobiology.vision.calculations.CalculationManager "Electrophysiological Imaging Fast" {output_path} {rawfile_location} 0.01 67 133 1000000 {available_cpus}', shell = True)


if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(prog='ks_to_vision',
                                     description='Convert spike times and TTL data from Kilosort output to vision files')

    parser.add_argument('exp_name', help='experiment name, (e.g. 20260715)')
    parser.add_argument('datafile_name', help='name of datafile to convert to vision')
    parser.add_argument('output_path', help='path for output files')
    parser.add_argument('vision_path', help='path to Vision.jar file')
    parser.add_argument('-r', '--raw', help='Path to raw data directory', default = RAW_DIR)
    parser.add_argument('-s', '--sorted', help='Path to KS sorter output directory', default = DATA_DIR)
    parser.add_argument('-k', '--ks_version', help='kilosort version used for spike sorting', default = 'kilosort2.5')
    parser.add_argument('-v', '--verbose', help='if true, print status messages to console', default = True)

    args = parser.parse_args()

    exp_name = args.exp_name
    datafile_name = args.datafile_name
    output_path = args.output_path
    vision_path = args.vision_path
    raw_path = args.raw
    sorted_path = args.sorted
    ks_version = args.ks_version
    verbose = args.verbose

    generate_vision_files(exp_name, datafile_name, output_path, raw_path, sorted_path,
                          ks_version, vision_path, verbose = True)









