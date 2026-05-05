import argparse
import visionloader as vl
from visionwriter import (EIWriter,
                          WriteableEIData)
from retinanalysis import (DATA_DIR)
import os
import numpy as np
from typing import List


def merge_error(mean_ei, total_spikes, n_spikes, ei_error, eis):
    """
    Calculate combined standard deviation of multiple electrical images to create one
    merged EI from several datafile EIs. This is a helper function, it is not meant
    to be called on its own.

    The ei_error property of EIContainer holds a standard deviation. To combine these
    standard devs, we use the law of total variance.
    """

    num_datasets = len(n_spikes)
    n_spikes = n_spikes[:, None, None]
    sos_within = np.sum((n_spikes - 1) * (ei_error ** 2), axis=0)
    sos_between = np.sum(n_spikes * ((eis - mean_ei) ** 2), axis=0)
    combined_var = (sos_within + sos_between) / (total_spikes - num_datasets)
    avg_error = np.sqrt(combined_var)

    return avg_error

def merge_eis(exp_name: str, chunk_name: str, datafiles: List[str], sorted_dir: str = DATA_DIR,
                output_dir: str = DATA_DIR, ss_version: str = 'kilosort2.5', verbose: bool = True):

    """
        Combine the EIs from multiple datafiles into a single chunk EI, and write
        this file to disk. This function pulls the EI data, takes the weighted
        average of the EIs using the number of spikes as 'weights', and calculates the
        average ei_error as a standard deviation.

        The output is an EIContainer object with 'ei', 'ei_error', 'n_spikes', 'nl_points',
        and 'nr_points' fields.

        Parameters:
            exp_name (str): Name of experiment (e.g. '20260224C')

            chunk_name (str): Name of chunk that all datafiles belong to (e.g. 'chunk1')

            datafiles (List[str]): List of datafiles to merge for this chunk (e.g. ['data000', 'data001'])

            sorted_dir (str) Optional: Path to sorted data directory where datafile EIs will be found in the
            subdirectories: sorted_dir/exp_name/datafile_name/ss_version
            Default is the configured retinanalysis DATA_DIR

            output_dir (str) Optional: Path to output directory where merged EI file will be stored in the
            subdirectories: output_dir/exp_name/chunk_name/ss_version/.
            Default is the configured retinanalysis DATA_DIR

            ss_version (str) Optional: The spike sorter version used to sort the data. Default is 'kilosort2.5'

            verbose (bool) Optional: If true will print status messages to the console. Default is True.

        Returns:
            None: No return values. The function will write a ss_version.ei file in the directory:
            output_dir/exp_name/chunk_name/ss_version/
    """
    if verbose:
        print(f'Merging EIs from {len(datafiles)} datafiles into {chunk_name} EI')

    # Pull sample ei for shape and default values in case a cell is missing its EI
    vision_dir = os.path.join(sorted_dir, exp_name, datafiles[0], ss_version)
    vcd = vl.load_vision_data(vision_dir, datafiles[0], include_ei = True)
    cell_ids = vcd.get_cell_ids()
    sample_ei = vcd.get_ei_for_cell(cell_ids[0])

    sample_nl_points = sample_ei.nl_points
    sample_nr_points = sample_ei.nr_points
    ei_shape = sample_ei.ei.shape

    all_eis = dict()
    all_ids = []
    for df in datafiles:
        vision_dir = os.path.join(sorted_dir, exp_name, df, ss_version)
        vcd = vl.load_vision_data(vision_dir, df, include_ei=True)
        cell_ids = vcd.get_cell_ids()
        all_ids += cell_ids
        all_eis[df] = {id:vcd.get_ei_for_cell(id) for id in cell_ids}

    cell_ids = np.unique(all_ids)

    if verbose:
        print(f'Found {len(cell_ids)} unique cell ids across all datafiles')

    combined_eis = dict()
    for id in cell_ids:
        eis = np.array([all_eis[df][id].ei if id in all_eis[df] else np.zeros(ei_shape) for df in datafiles])
        ei_error = np.array([all_eis[df][id].ei_error if id in all_eis[df] else np.zeros(ei_shape) for df in datafiles])
        n_spikes = np.array([all_eis[df][id].n_spikes if id in all_eis[df] else 0 for df in datafiles])
        nl_points = np.array([all_eis[df][id].nl_points if id in all_eis[df] else sample_nl_points for df in datafiles])
        nr_points = np.array([all_eis[df][id].nr_points if id in all_eis[df] else sample_nr_points for df in datafiles])
        
        # Make sure all EIs used the same number of time points
        assert np.all(nl_points==sample_nl_points), 'Not all eis have same number of left pts'
        assert np.all(nr_points==sample_nr_points), 'Not all eis have same number of right pts'
        
        # Calculate weighted average EI
        avg_ei = np.average(eis, axis=0, weights=n_spikes)
        total_spikes = np.sum(n_spikes)

        # Calculate avg error
        avg_error = merge_error(avg_ei, total_spikes, n_spikes,
                                     ei_error, eis)
        
        combined_eis[int(id)] = WriteableEIData(ei_matrix = avg_ei,
                                           ei_error = avg_error,
                                           n_spikes = int(total_spikes))


    output_dir = os.path.join(output_dir, exp_name, chunk_name, ss_version)

    if verbose:
        print(f"Writing EI File...")

    with EIWriter(output_dir, ss_version, sample_nl_points, sample_nr_points, array_id=504) as eir:
        eir.write_eis_by_cell_id(combined_eis)

    if verbose:
        print(f"EI file written to {output_dir}/{ss_version}.ei")


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='ei_merge',
                                     description='Script for merging EIs from several individual datafiles\
                                     into a single ei for the chunk.')

    # Positional arguments
    parser.add_argument('exp_name', help='Name of experiment, (e.g. 20260224C)',
                        type=str)
    parser.add_argument('chunk_name', help='name of sorting chunk (e.g. chunk1)',
                        type=str)
    parser.add_argument('datafiles', help='datafiles to include in chunk',
                        nargs='*')

    # Optional arguments
    parser.add_argument('--ss_version', help='version of kilosort used for spike sorting (e.g. kilosort2.5)',
                        type=str, nargs='?', default='kilosort2.5', const='kilosort2.5')

    parser.add_argument('--sorted_dir', help='path to directory containing datafile eis (e.g. .../Volumes/data/sorted/)',
                        type=str, nargs='?', default=DATA_DIR, const=DATA_DIR)

    parser.add_argument('--output_dir', help='path to directory containing datafile eis (e.g. .../Volumes/data/sorted/)',
                        type=str, nargs='?', default=DATA_DIR, const=DATA_DIR)

    parser.add_argument('-v', '--verbose', help='print status messages to console', action='store_true')

    args = parser.parse_args()

    chunk_name = args.chunk_name
    datafiles = args.datafiles
    sorted_dir = args.sorted_dir
    output_dir = args.output_dir
    ss_version = args.ss_version
    exp_name = args.exp_name
    verbose = args.verbose

    merge_eis(exp_name = exp_name, chunk_name = chunk_name, datafiles = datafiles, sorted_dir = sorted_dir,
              output_dir = output_dir, ss_version = ss_version, verbose = verbose)
