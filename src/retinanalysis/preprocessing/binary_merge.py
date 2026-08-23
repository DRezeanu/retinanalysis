import numpy as np
import bin2py as b2p
import os
import argparse
from retinanalysis._config import config
from typing import List
from tqdm.auto import tqdm
import csv
from pathlib import Path

def merge_binary(
    exp_name: str,
    chunk_name: str,
    datafiles: List[str],
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    ss_version: str = 'kilosort2.5',
    overwrite: bool = False,
    verbose: bool = True,
):
    """
        Function for merging binary data from a list of datafiles into a concatenated\
        bin file named chunk_name.bin. The output file will be saved in the directory\
        output_dir/exp_name/chunk_name/ss_version/ along with a chunk_name.csv file\
        that contains the start index, stop index, and total number of points for\
        each included datafile.

        Args:
            exp_name (str): The name of the experiment (e.g. 20260424C)
            
            chunk_name (str): The name of the output chunk (e.g. chunk1)

            datafiles (List[str]): List of datafiles to concatenate (e.g. ['data000',\
            'data001']

            raw_dir (str) Optional: Path to raw directory. Default is retinanalysis RAW_DIR

            output_dir (str) Optional: Path to output_directory. Default is retinanalysis \
            DATA_DIR

            ss_version (str) Optional: Spike sorter used. Default is kilosort2.5

            overwrite (bool) Optional: If true, will overwrite the 'chunk_name.bin' binary \
            file if one already exists in the output directory. Default is False.

            verbose (bool) Optional: If true, will print status messages to console. \
            Default is True.

        Returns:
            None: This function does not return anything. It will write a chunk_name.bin \
            and chunk_name.csv file to the given output_directory as described above.
    """
    if output_dir is None:
        output_dir = config.DATA_DIR

    if raw_dir is None:
        raw_dir = config.RAW_DIR

    chunk_size=100000
    
    out_path = Path(output_dir) / exp_name / chunk_name / ss_version

    if not out_path.exists():
        print(f'\nWARNING: output directory {out_path} does not exist, creating it now...\n')
        out_path.mkdir(parents=True, exist_ok=False)

    output_binary = out_path / f'{chunk_name}.bin'
    output_csv = out_path / f'{chunk_name}.csv'
    partial_path = out_path / f'{chunk_name}.bin.partial'

    if overwrite:
        if output_binary.is_file():
            os.remove(output_binary)
    else:
        if output_binary.is_file():
            raise FileExistsError(
                'File Already Exists and overwrite is set to False. '
                'To overwrite, set overwrite = True.'
            )

    pbar_total = 0
    datafile_pts = []
    datafile_chunks = []

    for datafile in datafiles:
        raw_file_location=os.path.join(raw_dir, exp_name, datafile)
        with b2p.PyBinFileReader(raw_file_location) as pbfr:
            num_pts = pbfr.length
            n_chunks = np.ceil(num_pts/chunk_size).astype(int)
            datafile_pts.append(num_pts)
            datafile_chunks.append(n_chunks)
            pbar_total += n_chunks

    if verbose:
        print(f'\nConcatenating {datafiles} binary files into {chunk_name}.bin...\n')

    total_pts = 0
    datafile_boundaries = []

    succeeded =  False
    try:
        with open(partial_path, 'wb') as f:
            with tqdm(total=pbar_total, desc=f'Creating {chunk_name}.bin', disable = not verbose) as pbar:
                for d_idx, datafile in enumerate(datafiles):

                    start_point = total_pts
                    raw_file_location=os.path.join(raw_dir, exp_name, datafile)

                    with b2p.PyBinFileReader(raw_file_location, chunk_samples = chunk_size) as pbfr:
                        for chunk in range(datafile_chunks[d_idx]):

                            start_idx=chunk*chunk_size
                            stop_idx = min(start_idx+chunk_size,datafile_pts[d_idx])

                            try:
                                data = pbfr.get_data(start_idx,stop_idx-start_idx)
                            except Exception as e:
                                raise RuntimeError(
                                        f'Could not retrieve binary data for {datafile}... '
                                        f'Start idx: {start_idx}\nStop idx: {stop_idx}'
                                ) from e

                            binary_data = np.ascontiguousarray(data[:, 1:])

                            binary_data.tofile(f)
                            pbar.update(1)

                    end_point = total_pts+datafile_pts[d_idx]-1
                    total_pts += datafile_pts[d_idx]

                    datafile_boundaries.append([datafile, start_point, end_point, datafile_pts[d_idx]])
        succeeded = True
    finally:
        if not succeeded:
            partial_path.unlink(missing_ok=True)

    os.replace(partial_path, output_binary)


    headers = ["Datafile", "Start Idx", "Stop Idx", "Num Points"]

    with open(output_csv, 'w',
              newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(datafile_boundaries)

    if verbose:
        print(f'\nWrote binary file to: {out_path}/{chunk_name}.bin')
        print(f'Wrote boundaries file to: {out_path}/{chunk_name}.csv')


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='binary_merge',
                                     description='Script for parsing litke binary data from\
                                                    Santa Cruz arrays into concatenated binary\
                                                    files for spike sorting.')

    parser.add_argument('exp_name',
                        help='Name of experiment for datafiles',
                        type=str)
    parser.add_argument('chunk_name',
                      help='Name of chunk, used for naming merged binary file',
                      type=str)
    parser.add_argument('datafiles',
                        help='Datafiles to include in corresponding chunk',
                        type=str,
                        nargs='*')
    # Optional arguments
    parser.add_argument('--raw_dir',
                        help='directory containing raw files from datafiles of interest',
                        type=str,
                        nargs='?',
                        default=config.RAW_DIR,
                        const=config.RAW_DIR)
    parser.add_argument('--output_dir',
                        help='path to output directory where bin file and csv file will be\
                        saved',
                        type=str,
                        nargs='?',
                        default=config.DATA_DIR,
                        const=config.DATA_DIR)
    parser.add_argument('--ss_version',
                        help='Spike sorter used, this will be used as a folder in which the\
                        data will be saved, (e.g. output_dir/exp_name/chunk_name\
                        /ss_version/chunk_name.bin',
                        type=str,
                        nargs='?',
                        default='kilosort2.5',
                        const='kilosort2.5')
    parser.add_argument('-o', '--overwrite',
                        help='Overwrite binary file if it already exists',
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        help='Print status messages to console',
                        action='store_true')
    
    args = parser.parse_args()

    exp_name=args.exp_name
    chunk_name=args.chunk_name
    datafiles=args.datafiles
    raw_dir=args.raw_dir
    output_dir = args.output_dir
    ss_version = args.ss_version
    overwrite = args.overwrite
    verbose=args.verbose


    merge_binary(exp_name, chunk_name, datafiles, raw_dir,
                 output_dir, ss_version, overwrite, verbose)
