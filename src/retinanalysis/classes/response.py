from retinanalysis.utils.datajoint_utils import (get_epochblock_amp_data,
                                                 get_epochblock_frame_data,
                                                 get_epochblock_timing,
                                                 get_block_id_from_datafile,
                                                 get_exp_summary)

from retinanalysis.config.settings import ANALYSIS_DIR
from retinanalysis.utils.vision_utils import (get_protocol_vcd,
                                              cluster_match)

from retinanalysis.utils.spike_detector import detector
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle
from typing import (Optional,
                    List)
import matplotlib.pyplot as plt
import os

SAMPLE_RATE = 20000 # MEA DAQ sample rate in Hz


class ResponseBlock:
    """
    Generic class for single cell or MEA response blocks. A response block contains spike time
    data for an MEA datafile (e.g. data000 or data001) or a Single Cell epoch block.

    Parameters:
        exp_name (str) Optional: String giving the name of the experiment (e.g. '20250713C').
        Optional and Default None, but None is only valid if pkl_file is provided.

        block_id (int) Optional: Block_id for this particular epoch block in the database.
        Optional and Default None, but None is only valid if pkl_file is provided.

        h5_file (str) Optional: Path to an h5 file containing metadata for this experiment and
        epoch block. Default None but required in order to get frame monitor data.

        pkl_file (str | dict) Optional: Path to pkl_file or dict generated from pkl_file
        containing a ResponseBlock object.

        b_load_fd (bool): Boolean value, if True will load frame monitor data.

        b_LED (bool): Boolean value, set to True if stimulus was delivered by an LED.
        This automatically overwrites b_load_fd to False since LED stimuli have no frame data.

        verbose (bool): Boolean value, if True will print status messages to console. Default True.
    """
    def __init__(self, exp_name: Optional[str]=None, block_id: Optional[int]=None,
                 h5_file: Optional[str]=None, pkl_file: Optional[str | dict]=None, 
                 b_load_fd: bool=True, b_LED: bool=False,
                 verbose: bool = True):

        self.verbose = verbose
        self.b_LED = b_LED

        if self.b_LED:
            # No frame data for LED blocks
            b_load_fd = False

        if pkl_file is None:
            if self.verbose:
                print(f"Initializing ResponseBlock for {exp_name} block {block_id}")
            if exp_name is None or block_id is None:
                raise ValueError("Either exp_name and block_id or pkl_file must be provided.")
        else:
            if self.verbose:
                print(f"Initializing ResponseBlock from pickle file.")
            # Load from pickle file if string, otherwise must be a dict
            if isinstance(pkl_file, str):
                if self.verbose:
                    print(f"  pkl_file: {pkl_file}")
                with open(pkl_file, 'rb') as f:
                    d_out = pickle.load(f)
            else:
                d_out = pkl_file
                pkl_file = "input dict."
            self.__dict__.update(d_out)
            if self.verbose:
                print(f"ResponseBlock loaded from {pkl_file}")
            return

        self.exp_name = exp_name
        self.block_id = block_id    
        self.h5_file = h5_file
        self.d_timing = get_epochblock_timing(self.exp_name, self.block_id, b_LED=self.b_LED)

        if b_load_fd:
            frame_data, frame_sample_rate = get_epochblock_frame_data(self.exp_name, self.block_id, str_h5=self.h5_file, verbose = self.verbose)
        else:
            frame_data = np.array([])
            frame_sample_rate = None

        self.frame_data = frame_data
        self.frame_sample_rate = frame_sample_rate
        exp_summary = get_exp_summary(self.exp_name)

        assert exp_summary is not None, f'Experiment summary for {self.exp_name} came back empty'
        self.d_block_summary = exp_summary.query('block_id == @block_id').iloc[0].to_dict()

    def plot_frame_monitor(self, e_idx=0, xlim=(0,1)):
        f, ax = plt.subplots(figsize=(10,4))
        fd = self.frame_data[e_idx]
        time = np.arange(fd.shape[0]) / self.frame_sample_rate
        ax.plot(time, fd)
        for ft in self.d_timing['frameTimesMs'][e_idx]:
            ax.axvline(x=ft * 1e-3, color='r', linestyle='--')
        ax.set_xlim(*xlim)

    def export_to_pkl(self, file_path: str):
        d_out = self.__dict__.copy()
        with open(file_path, 'wb') as f:
            pickle.dump(d_out, f)
        print(f"ResponseBlock exported to {file_path}")

    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  block_id: {self.block_id}\n"
        str_self += f"  b_LED: {self.b_LED} (whether response to LED stimulus)\n"
        str_self += f"  protocol_name: {self.d_block_summary['protocol_name']}"
        str_self += f"  d_timing with keys: {list(self.d_timing.keys())}\n"
        str_self += f"  frame_sample_rate: {self.frame_sample_rate} Hz\n"
        str_self += f"  frame_data shape: {self.frame_data.shape}\n"
        if self.h5_file is not None:
            str_self += f"  h5_file: {self.h5_file}\n"

        return str_self


class SCResponseBlock(ResponseBlock):
    """
    Single Cell subclass of ResponseBlock. Can load spiking and non-spiking data
    from amp output contained in h5 file. Automatically uses a spike detector to
    get spikes if b_spiking is set to True.
    
    Parameters:
        exp_name (str) Optional: String giving the name of the experiment (e.g. '20250713C').
        Optional and Default None, but None is only valid if pkl_file is provided.

        block_id (int) Optional: Block_id for this particular epoch block in the database.
        Optional and Default None, but None is only valid if pkl_file is provided.

        h5_file (str) Optional: Path to an h5 file containing metadata for this experiment and
        epoch block. Optional and Default None, but None is only valid if pkl_file is provided.

        pkl_file (str | dict) Optional: Path to pkl_file or dict generated from pkl_file
        containing a ResponseBlock object.
        
        b_spiking (bool): Boolean value, if True this is a recording from a spiking cell and 
        spike detection will be run. Default False.

        b_load_fd (bool): Boolean value, if True will load frame monitor data.

        b_LED (bool): Boolean value, set to True if stimulus was delivered by an LED.
        This automatically overwrites b_load_fd to False since LED stimuli have no frame data.

        verbose (bool): Boolean value, if True will print status messages to console. Default True.

        **detector_kwargs: kwargs to be given to the detector method in get_spike_times()
    """

    def __init__(self, exp_name: Optional[str]=None, block_id: Optional[int]=None,
                 h5_file: Optional[str]=None, pkl_file: Optional[str]=None,
                 b_spiking: bool=False, b_load_fd: bool=True, b_LED: bool=False,
                 verbose: bool = True, **detector_kwargs):

        super().__init__(
            exp_name=exp_name, block_id=block_id, h5_file=h5_file, b_LED=b_LED,
            pkl_file=pkl_file, b_load_fd=b_load_fd, verbose = verbose)

        if pkl_file is not None:
            return
        
        self.b_spiking = b_spiking
        amp_data, sample_rate = get_epochblock_amp_data(self.exp_name, self.block_id, str_h5=self.h5_file, verbose = self.verbose)
        self.amp_data = amp_data
        self.amp_sample_rate = sample_rate
        if b_spiking:
            self.get_spike_times(**detector_kwargs)

    def get_spike_times(self, **detector_kwargs):
        spike_times, amps, refs = detector(self.amp_data, sample_rate=self.amp_sample_rate, **detector_kwargs)
        self.spike_times = spike_times
        self.spike_amps = amps
        self.spike_refs = refs

    def __repr__(self):
        str_self = super().__repr__()
        str_self += f"  b_spiking: {self.b_spiking}\n"
        str_self += f"  amp_data shape: {self.amp_data.shape}\n"
        str_self += f"  amp_sample_rate: {self.amp_sample_rate} Hz\n"
        if self.b_spiking:
            str_self += f"  spike_times length: {len(self.spike_times)}\n"
            str_self += f"  spike_amps length: {len(self.spike_amps)}\n"
            str_self += f"  spike_refs length: {len(self.spike_refs)}\n"
        return str_self


class MEAResponseBlock(ResponseBlock):
    """
    Subclass of ResponseBlock, specifically for MEA spike time data. This pulls 
    data from the vision files in a datafile folder (e.g. data000) specified by
    datafile_name. The class also optionally loads frame monitor data and EI data.

    Parameters:
        exp_name (str) Optional: String specifying the name of the experiment (e.g. 20250715C).
        Optional and default of None, but None is only valid if a pkl_file is provided.

        datafile_name (str) Optional: String specifying the name of the datafile to query (e.g. dat000).
        Optional and default of None, but None is only valid if a pkl_file is provided.

        ss_version (str): String specifying the spike sorter used. Default 'kilosort2.5'.

        pkl_file (str) Optional: Path to a pkl_file containing an MEAResponseBlock object. Default None.

        b_load_fd (bool): Boolean value, if True will load frame monitor data. Default is False.

        b_LED (bool): Boolean value, if True the stimulus used for this datafile was delivered using an
        LED rather than a microdisplay or lightcrafter. This automatically overwrites b_load_fd to False
        since LED stimuli have no frame monitor data.

        verbose (bool): Boolean value, if True all status messages will be printed to the console as
        the response block is created. Default is True.
    """

    def __init__(self, exp_name: Optional[str]=None, datafile_name: Optional[str]=None,
                 ss_version: str = 'kilosort2.5', pkl_file: Optional[str]=None, 
                 h5_file: Optional[str]=None, include_ei: bool=True, 
                 b_load_fd: bool=False, b_LED: bool=False,
                 verbose: bool = True):

        if pkl_file is None:
            # If pkl_file is None, exp_name and datafile_name must be provided
            if exp_name is None or datafile_name is None:
                raise ValueError("Either exp_name and datafile_name or pkl_file must be provided.")
            else:
                # If exp_name and datafile_name are provided, get block_id from datafile_name
                block_id = get_block_id_from_datafile(exp_name, datafile_name)
                # Set the ss_version and datafile_name for loading VCD.
                self.ss_version = ss_version
                self.datafile_name = datafile_name
        else:
            # If pkl_file is provided, block_id can be None.
            block_id = None

        # Initialize superclass
        super().__init__(
            exp_name=exp_name, block_id=block_id, pkl_file=pkl_file, 
            h5_file=h5_file, b_LED=b_LED, b_load_fd=b_load_fd, verbose = verbose)

        self.amp_sample_rate = SAMPLE_RATE # MEA DAQ sample rate in Hz, analogous variable in SCResponseBlock
        self.vcd = get_protocol_vcd(self.exp_name, self.datafile_name, self.ss_version, include_ei=include_ei, verbose = self.verbose)

        # If pkl_file is provided, everything else is already loaded in parent init.
        if pkl_file is not None:
            return

        
        self.protocol_name = self.d_block_summary['protocol_name']
        self.cell_ids = np.array(self.vcd.get_cell_ids(), dtype=int)

        if include_ei:
            self.d_EIs = dict()
            self.d_EI_error = dict()
            bad_ids = []
            for id in self.cell_ids:
                try:
                    self.d_EIs[id] = self.vcd.get_ei_for_cell(id).ei
                    self.d_EI_error[id] = self.vcd.get_ei_for_cell(id).ei_error
                except:
                    print(f'WARNING: No ei for ref cell id {id}, removing from {self.datafile_name} ResponseBlock')
                    bad_ids.append(id)

            mask = ~np.isin(self.cell_ids, bad_ids)
            self.cell_ids = self.cell_ids[mask]
        
        self.get_spike_times()

    # Pull spike times from Vision Cell Data Table, split by cell and epoch,
    # and add them to df_spike_times dataframe for each cell and epoch
    def get_spike_times(self):
        d_spike_times = {'cell_id': [], 'spike_times': []}

        epoch_starts = self.d_timing['epochStarts']
        epoch_ends = self.d_timing['epochEnds']

        self.n_epochs = self.d_timing['n_epochs']
        for cell_id in self.cell_ids:
            all_spike_times = []
            # STs in samples
            try:
                cell_sts = self.vcd.get_spike_times_for_cell(cell_id)
                for i in range(self.n_epochs):
                    # Set epoch start as zero
                    e_sts = cell_sts - epoch_starts[i]
                    n_epoch_samples = epoch_ends[i] - epoch_starts[i]

                    # Filter spike times to be within the epoch
                    e_sts = e_sts[(e_sts >= 0) & (e_sts <= n_epoch_samples)]

                    # From samples to ms
                    e_sts = e_sts / SAMPLE_RATE * 1000

                    all_spike_times.append(e_sts)

                d_spike_times['cell_id'].append(cell_id)
                d_spike_times['spike_times'].append(all_spike_times)
            except Exception as e:
                print(f"WARNING: No spike times found for cell {cell_id}.\n Error: {e}")
        self.df_spike_times: pd.DataFrame = pd.DataFrame(d_spike_times)

    def get_max_bins_for_rate(self, bin_rate: float):
        # bin_rate: float, in Hz
        # Returns the maximum number of bins for the given bin rate across all epochs.
        epoch_starts = self.d_timing['epochStarts']
        epoch_ends = self.d_timing['epochEnds']
        ls_bins = []
        for i in range(self.n_epochs):
            n_epoch_samples = epoch_ends[i] - epoch_starts[i]
            n_bins = np.ceil(n_epoch_samples / (SAMPLE_RATE / bin_rate))
            ls_bins.append(n_bins)
        n_max_bins = int(np.max(ls_bins))
        return n_max_bins
    
    def bin_spike_times_by_frames(self): # , stride: int=1
        if self.b_LED:
            raise ValueError("Cannot bin spike times by frames for LED blocks, no frame data.")

        stride = 1
        # TODO implement stride > 1 with interpolating bw frame times.
        frame_times_ms = self.d_timing['frameTimesMs']
        if int(self.exp_name[:8]) < 20230926:
            marginal_frame_rate = 60.31807657 # Upper bound on the frame rate to make sure that we don't miss any frames.
        else:
            marginal_frame_rate = 59.941548817817917 # Upper bound on the frame rate to make sure that we don't miss any frames.
        bin_rate = marginal_frame_rate * stride # in Hz
        
        n_max_bins = self.get_max_bins_for_rate(bin_rate)
        n_cells = len(self.cell_ids)
        
        binned_spikes = np.zeros((n_cells, self.n_epochs, n_max_bins))
        ls_diff_frames = []
        for i_cell in tqdm(self.df_spike_times.index, desc='Binning spikes for cells'):
            sts = self.df_spike_times.at[i_cell, 'spike_times']
            for j_epoch in range(self.n_epochs):
                e_sts = sts[j_epoch]
                
                fts = frame_times_ms[j_epoch]
                fts, _ = check_frame_times(fts, frame_rate=marginal_frame_rate)
                ls_diff_frames.append(np.diff(fts))

                bs = np.histogram(e_sts, bins=fts)[0]
                if len(bs) > n_max_bins:
                    bs = bs[:n_max_bins]
                binned_spikes[i_cell, j_epoch, :len(bs)] = bs
        self.df_spike_times['binned_spikes'] = [binned_spikes[i_cell, :, :] for i_cell in range(n_cells)]
        
        self.binned_spikes = binned_spikes

        # Taken from SD. Compute the mean frame rate.
        ls_diff_frames = np.concatenate(ls_diff_frames)
        ls_diff_frames = ls_diff_frames[ls_diff_frames < 20.0]
        mean_frame_rate = 1000.0 / np.mean(ls_diff_frames)
        print(f'Mean frame rate: {mean_frame_rate:.2f} Hz\n')
        self.mean_frame_rate = mean_frame_rate
        self.bin_rate = bin_rate
        self.time_bins_ms = np.arange(0, n_max_bins) / self.bin_rate * 1000 # in ms

    def bin_spike_times_at_rate(self, bin_rate: float, b_count: bool=True):
        n_bins = self.get_max_bins_for_rate(bin_rate)
        time_bins = np.arange(n_bins + 1) / bin_rate * 1000 # in ms
        n_cells = len(self.cell_ids)
        
        binned_spikes = np.zeros((n_cells, self.n_epochs, n_bins))
        for i_cell in tqdm(self.df_spike_times.index, desc='Binning spikes for cells'):
            sts = self.df_spike_times.at[i_cell, 'spike_times']
            for j_epoch in range(self.n_epochs):
                e_sts = sts[j_epoch]

                bs = np.histogram(e_sts, bins=time_bins)[0]
                binned_spikes[i_cell, j_epoch, :] = bs
        
        if not b_count:
            binned_spikes *= bin_rate
        
        self.df_spike_times['binned_spikes'] = [binned_spikes[i_cell, :, :] for i_cell in range(n_cells)]
        
        self.binned_spikes = binned_spikes
        self.bin_rate = bin_rate
        self.time_bins_ms = time_bins[:-1]

    # Add cell types column to df_spike_times dataframe, given a specific noise chunk and typing file.
    # Cell types are added after cluster matching with the given noise_chunk.
    def add_cell_types(self, noise_chunk: Optional[str] = None, typing_file: Optional[str] = None):

        import retinanalysis
        from retinanalysis.classes.analysis_chunk import AnalysisChunk
        try:
            import importlib.resources as ir
        except:
            import importlib_resources as ir #type: ignore

        cell_types_list_path = str(ir.files(retinanalysis) / "assets/cell_types.csv")
        cell_types_list = pd.read_csv(cell_types_list_path)
        cell_types = cell_types_list['cell_types'].values

        if noise_chunk is None:
            from retinanalysis.classes.stim import MEAStimBlock
            stim_block = MEAStimBlock(self.exp_name, self.datafile_name, verbose = False) 
            noise_chunk = stim_block.nearest_noise_chunk
            analysis_chunk = AnalysisChunk(self.exp_name, noise_chunk, self.ss_version,
                                           b_load_spatial_maps = False, verbose = False)

            if typing_file is None:
                if len(analysis_chunk.typing_files) == 0:
                    raise ValueError(f"No typing files for nearest noise {noise_chunk}")
                else:
                    print(f"No typing file provided, using {analysis_chunk.typing_files[0]}")
                    typing_file = analysis_chunk.typing_files[0]
            else:
                assert typing_file in analysis_chunk.typing_files, f"Typing file {typing_file} not found in noise chunk {noise_chunk}"

        else:
            analysis_chunk = AnalysisChunk(self.exp_name, noise_chunk, self.ss_version,
                                           b_load_spatial_maps = False, verbose = False)
            if typing_file is None:
                if len(analysis_chunk.typing_files) == 0:
                    raise ValueError(f"No typing files for nearest noise {noise_chunk}")
                else:
                    print(f"No typing file provided, using {analysis_chunk.typing_files[0]}")
                    typing_file = analysis_chunk.typing_files[0]

            else:
                assert typing_file in analysis_chunk.typing_files, f"Typing file {typing_file} not found in noise chunk {noise_chunk}"


        assert noise_chunk is not None and typing_file is not None

        # TODO: Check if responseblock is part of analysis chunk, and if so skip cluster matching.
        # Cluster Match
        self.match_dict, _ = cluster_match(analysis_chunk, self, verbose = self.verbose) 
        inverse_match_dict = {value: key for key, value in self.match_dict.items()}
        noise_ids = [inverse_match_dict[id] if id in inverse_match_dict.keys() else 0 for id in self.cell_ids]


        # Pull types from classification file
        file_path = os.path.join(ANALYSIS_DIR, self.exp_name, noise_chunk, self.ss_version, typing_file)

        d_result = dict()
        with open(file_path, 'r') as file:
           for line in file:
                # Split each line into key and value using the specified delimiter
                key, value = map(str.strip, line.split(' ', 1))
                sub_values = value.split('/')
                
                # Add key-value pair to the dictionary
                if int(key) in self.match_dict.keys():
                    d_result[self.match_dict[int(key)]] = sub_values[:-1]

        for idx, id in enumerate(self.cell_ids):
            if id in d_result.keys():

                for ctype in cell_types:
                    if ctype in d_result[id]:
                        d_result[id] = ctype
                        break
            else:
                d_result[id] = 'Unknown'
            
            if 'All' in d_result[id]:
                d_result[id] = 'Unknown'

            if noise_ids[idx] == 0:
                d_result[id] = 'Ummatched'
            
        
        classification = [d_result[id] for id in self.cell_ids]
        self.df_spike_times['noise_id'] = noise_ids
        self.df_spike_times['cell_type'] = classification


    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  datafile_name: {self.datafile_name}\n"
        str_self += f"  protocol_name: {self.protocol_name}\n"
        str_self += f"  b_LED: {self.b_LED} (whether response to LED stimulus)\n"
        str_self += f"  ss_version: {self.ss_version}\n"
        str_self += f"  n_epochs: {self.n_epochs}\n"
        str_self += f"  cell_ids of length: {len(self.cell_ids)}\n"
        str_self += f"  df_spike_times with times for {self.df_spike_times.shape[0]} cells\n"
        str_self += f"  block_id: {self.block_id}\n"
        str_self += f"  d_EIs dictionary containing EIs for {len(self.cell_ids)} cell IDs\n"
        str_self += f"  d_EI_error dictionary containing EI SDs for {len(self.cell_ids)} cell IDs\n"
        str_self += f"  d_timing with keys: {list(self.d_timing.keys())}\n"
        str_self += f"  frame_sample_rate: {self.frame_sample_rate}\n"
        str_self += f"  frame_data shape: {self.frame_data.shape}\n"
        return str_self

    def export_to_pkl(self, file_path: str):
        d_out = self.__dict__.copy()
        # Pop out vcd
        d_out.pop('vcd', None)
        with open(file_path, 'wb') as f:
            pickle.dump(d_out, f)
        print(f"MEAResponseBlock exported to {file_path}")

class MEAResponseGroup:
    """
    A group of MEAResponseBlocks concatenated into a single object. This works as long as all the
    included MEAResponseBlocks are (1) part of the same experiment, (2) used the same protocol (e.g. movingbar),
    (3) part of the same sorting chunk, and (4) each come from a different datafile.

    Parameters:
        ls_blocks (List[MEAResponseBlock]): A list of MEAResponseBlock objects to concatenate into a ResponseGroup

        b_load_fd (bool): Boolean value, if True will pull and include frame monitor data for all response blocks

        verbose (bool): Boolean value, if True will print all status messages to the console. Because of the number
        of response blocks created, default value is False.
    """

    def __init__(self, ls_blocks: List[MEAResponseBlock], b_load_fd: bool = False, verbose: bool = False):

        self.verbose = verbose
        self.b_LED = ls_blocks[0].b_LED

        if not all(block.exp_name == ls_blocks[0].exp_name for block in ls_blocks):
            raise ValueError("All ResponseBlocks must have the same exp_name")

        if not all(block.protocol_name == ls_blocks[0].protocol_name for block in ls_blocks):
            raise ValueError("All ResponseBlocks must have the same protocol_name")

        if not all(block.d_block_summary['chunk_name'] == ls_blocks[0].d_block_summary['chunk_name'] for block in ls_blocks):
            raise ValueError("All ResponseBlocks must be from the same chunk")

        datafile_names = [block.datafile_name for block in ls_blocks]
        if len(set(datafile_names)) != len(datafile_names):
            raise ValueError(f"ResponseBlocks must have unique datafile_names, but found {datafile_names}")

        if self.verbose:
            print(f"\nGenerating MEA Response Block from {ls_blocks[0].protocol_name} datafiles")

        # Pull only cell ids that are common to all blocks in this group
        all_ids = [set(block.cell_ids) for block in ls_blocks]
        self.cell_ids = list(set.intersection(*all_ids))
        n_cells_lost = len(set.union(*all_ids)) - len(self.cell_ids)
        if self.verbose:
            print(f'\nLost {n_cells_lost} cells when concatenating response blocks')

        # Concatenate spike times and recreate df_spike_times
        ls_spike_times = []
        for id in self.cell_ids:
            concat_spike_times = []
            for block in ls_blocks:
                block_times = block.df_spike_times.query('cell_id == @id')['spike_times'].item()
                concat_spike_times += block_times
            ls_spike_times.append(concat_spike_times)

        df_spike_times: pd.DataFrame = pd.DataFrame({'cell_id' : self.cell_ids,
                                       'spike_times' : ls_spike_times})

        # Recreate d_timing dictionary by concatenating values that need to be joined
        d_timing = {'exp_name' : ls_blocks[0].d_block_summary['exp_name'],
                    'block_ids' : [block.block_id for block in ls_blocks],
                    'epochStarts' : [starts for block in ls_blocks for starts in block.d_timing['epochStarts']],
                    'epochEnds' : [ends for block in ls_blocks for ends in block.d_timing['epochEnds']],
                    'n_samples' : [block.d_timing['n_samples'] for block in ls_blocks],
                    'n_epochs' : [block.d_timing['n_epochs'] for block in ls_blocks],
                    'pre_time_ms' : [block.d_timing['pre_time_ms'] for block in ls_blocks],
                    'stim_time_ms' : [block.d_timing['stim_time_ms'] for block in ls_blocks],
                    'tail_time_ms' : [block.d_timing['tail_time_ms'] for block in ls_blocks]
        }
        if not self.b_LED:
            d_timing.update({
                'frameTimesMs': [times for block in ls_blocks for times in block.d_timing['frameTimesMs']],
                'stage_frame_rate' : ls_blocks[0].d_timing['stage_frame_rate'],
                'actual_onset_times_ms' : [onsets for block in ls_blocks for onsets in block.d_timing['actual_onset_times_ms']],
                'actual_offset_times_ms' : [offsets for block in ls_blocks for offsets in block.d_timing['actual_offset_times_ms']]
                })


        self.ls_blocks = ls_blocks
        self.block_ids = [block.block_id for block in ls_blocks]
        self.exp_name = ls_blocks[0].exp_name
        self.ss_version = ls_blocks[0].ss_version
        self.n_epochs = np.sum([block.n_epochs for block in ls_blocks])
        self.protocol_name = ls_blocks[0].protocol_name
        self.datafile_names = datafile_names
        self.df_spike_times = df_spike_times
        self.d_timing = d_timing
        self.d_EIs, self.d_EI_error = self.merge_eis()


        if b_load_fd:
            
            if all(block.frame_sample_rate is None for block in ls_blocks) or all(block.frame_sample_rate is not None for block in ls_blocks):

                if self.verbose:
                    print("Loading and concatenating frame monitor data for all datafiles...\n")

                if ls_blocks[0].frame_sample_rate is None:
                    frame_monitor_data = [get_epochblock_frame_data(block.exp_name, block.block_id, str_h5=block.h5_file) for block in ls_blocks]    
                    frame_sample_rates = [data for _, data in frame_monitor_data]
                    self.frame_sample_rates = frame_sample_rates
                    try:
                        frame_data = np.array([data for data, _ in frame_monitor_data])
                        self.frame_data = np.reshape(frame_data, (-1, frame_data.shape[2]))
                    except Exception as e:
                        print('Could not convert fame data to numpy array, trying object array')
                        self.frame_data = np.array([data for _, data in frame_monitor_data], dtype = object)
                else:
                    self.frame_sample_rates = [block.frame_sample_rate for block in ls_blocks]
                    try:
                        frame_data = np.array([block.frame_data for block in ls_blocks])
                        self.frame_data = np.reshape(frame_data, (-1, frame_data.shape[2]))
                    except Exception as e:
                        print('Could not convert fame data to numpy array, trying object array')
                        self.frame_data = np.array([block.frame_data for block in ls_blocks], dtype = object)

                
            else:
                raise ValueError("Some ResponseBlocks have frame data and some don't, they must all be uniform")

            if np.all(np.array(self.frame_sample_rates) == self.frame_sample_rates[0]):
                self.frame_stample_rates = self.frame_sample_rates[0]

        else:
            self.frame_sample_rates = None 
            self.frame_data = np.array([])

    def merge_eis(self):

        if self.verbose:
            print("\nGenerating average EI for all cells")

        # Take the weighted average of each cell's EI across datasets
        all_eis = []
        all_ei_error = []
        for id in self.cell_ids:
            ls_eis = [block.vcd.get_ei_for_cell(id).ei for block in self.ls_blocks]
            ls_spikes = [block.vcd.get_ei_for_cell(id).n_spikes for block in self.ls_blocks]
            ls_error = [block.vcd.get_ei_for_cell(id).ei_error for block in self.ls_blocks]

            ls_eis = np.stack(ls_eis)
            ls_error = np.stack(ls_error)
            ls_spikes = np.asarray(ls_spikes)

            # Weighted average of EIs
            average_ei = np.average(ls_eis, axis = 0, weights = ls_spikes)

            # Computed pooled variance, then take square root to get pooled SD
            within_var = np.sum(
                (ls_spikes[:, None, None] - 1) * (ls_error ** 2),
                axis=0
            )

            between_var = np.sum(
                ls_spikes[:, None, None] * (ls_eis - average_ei) ** 2,
                axis=0
            )

            total_spikes = np.sum(ls_spikes)

            pooled_var = (within_var + between_var) / (total_spikes - 1)
            average_error = np.sqrt(pooled_var)

            all_eis.append(average_ei)
            all_ei_error.append(average_error)

        all_eis = np.stack(all_eis)
        all_ei_error = np.stack(all_ei_error)
        mean_eis = {id : all_eis[idx] for idx, id in enumerate(self.cell_ids)}
        mean_ei_error = {id : all_ei_error[idx] for idx, id, in enumerate(self.cell_ids)}
        
        if self.verbose:
            print(f"Merged EIs and EI_Error for {len(mean_eis)} cells across {len(self.ls_blocks)} response blocks")
        
        return mean_eis, mean_ei_error

    def add_cell_types(self, noise_chunk: Optional[str] = None, typing_file: Optional[str] = None):

        import retinanalysis
        from retinanalysis.classes.analysis_chunk import AnalysisChunk
        try:
            import importlib.resources as ir
        except:
            import importlib_resources as ir #type: ignore

        cell_types_list_path = str(ir.files(retinanalysis) / "assets/cell_types.csv")
        cell_types_list = pd.read_csv(cell_types_list_path)
        cell_types = cell_types_list['cell_types'].values

        if noise_chunk is None:
            from retinanalysis.classes.stim import create_mea_stim_group
            stim_group = create_mea_stim_group(self.exp_name, self.datafile_names, verbose = False)
            noise_chunk = stim_group.nearest_noise_chunk
            analysis_chunk = AnalysisChunk(self.exp_name, noise_chunk, self.ss_version,
                                           b_load_spatial_maps = False, verbose = False)

            if typing_file is None:
                if len(analysis_chunk.typing_files) == 0:
                    raise ValueError(f"No typing files for nearest noise {noise_chunk}")
                else:
                    print(f"No typing file provided, using {analysis_chunk.typing_files[0]}")
                    typing_file = analysis_chunk.typing_files[0]
            else:
                assert typing_file in analysis_chunk.typing_files, f"Typing file {typing_file} not found in noise chunk {noise_chunk}"

        else:
            analysis_chunk = AnalysisChunk(self.exp_name, noise_chunk, self.ss_version,
                                           b_load_spatial_maps = False, verbose = False)
            if typing_file is None:
                if len(analysis_chunk.typing_files) == 0:
                    raise ValueError(f"No typing files for nearest noise {noise_chunk}")
                else:
                    print(f"No typing file provided, using {analysis_chunk.typing_files[0]}")
                    typing_file = analysis_chunk.typing_files[0]

            else:
                assert typing_file in analysis_chunk.typing_files, f"Typing file {typing_file} not found in noise chunk {noise_chunk}"


        assert noise_chunk is not None and typing_file is not None

        # Cluster Match
        self.match_dict, _ = cluster_match(analysis_chunk, self, verbose = self.verbose) 
        inverse_match_dict = {value: key for key, value in self.match_dict.items()}
        noise_ids = [inverse_match_dict[id] if id in inverse_match_dict.keys() else 0 for id in self.cell_ids]


        # Pull types from classification file
        file_path = os.path.join(ANALYSIS_DIR, self.exp_name, noise_chunk, self.ss_version, typing_file)

        d_result = dict()
        with open(file_path, 'r') as file:
           for line in file:
                # Split each line into key and value using the specified delimiter
                key, value = map(str.strip, line.split(' ', 1))
                sub_values = value.split('/')
                
                # Add key-value pair to the dictionary
                if int(key) in self.match_dict.keys():
                    d_result[self.match_dict[int(key)]] = sub_values[:-1]

        for idx, id in enumerate(self.cell_ids):
            if id in d_result.keys():

                for ctype in cell_types:
                    if ctype in d_result[id]:
                        d_result[id] = ctype
                        break
            else:
                d_result[id] = 'Unknown'
            
            if 'All' in d_result[id]:
                d_result[id] = 'Unknown'

            if noise_ids[idx] == 0:
                d_result[id] = 'Ummatched'
            
        
        classification = [d_result[id] for id in self.cell_ids]
        self.df_spike_times['noise_id'] = noise_ids
        self.df_spike_times['cell_type'] = classification

    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  ls_blocks containg {len(self.ls_blocks)} MEAResponseBlocks\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  datafile_names: {self.datafile_names}\n"
        str_self += f"  protocol_name: {self.protocol_name}\n"
        str_self += f"  b_LED: {self.b_LED} (whether response to LED stimulus)\n"
        str_self += f"  ss_version: {self.ss_version}\n"
        str_self += f"  n_epochs: {self.n_epochs}\n"
        str_self += f"  cell_ids of length: {len(self.cell_ids)}\n"
        str_self += f"  df_spike_times with times for {self.df_spike_times.shape[0]} cells\n"
        str_self += f"  block_ids: {self.block_ids}\n"
        str_self += f"  d_EIs dictionary containing EIs for {len(self.cell_ids)} cell IDs\n"
        str_self += f"  d_EI_error dictionary containing EI SDs for {len(self.cell_ids)} cell IDs\n"
        str_self += f"  d_timing with keys: {list(self.d_timing.keys())}\n"
        str_self += f"  frame_sample_rates: {self.frame_sample_rates}\n"
        str_self += f"  frame_data shape: {self.frame_data.shape}\n"
        return str_self


def create_mea_response_group(
        exp_name: str, ls_datafile_names: List[str], 
        ss_version: str='kilosort2.5', b_load_fd: bool = False, 
        b_LED: bool=False, verbose: bool = False
    ):
    """
    Helper function for creating an MEA Response Group from a list of datafiles. The function creates all of the
    MEAResponseBlock objects from the list of datafiles, and uses those MEAResponseBlock objects to create the
    MEAResponseGroup.

    Parameters:
        exp_name (str): String of experiment to use (e.g. '20250713C')

        ls_datafile_names (List[str]): List of datafiles names as strings (e.g. ['data000', 'data001'])

        ss_version (str): Spike sorter used to sort those datafiles, default 'kilosort2.5'.

        b_load_fd (bool): Boolean value, if True will load and included frame monitor data. Default False.

        b_LED (bool): Boolean value, if True assume stimulus for tehse datafiles was deliverd by an LED.
        This automatically sets b_load_fd to False since LED stimuli have no frame data. Default False.

        verbose (bool): Boolean value, if True will print all outputs to the console. Since multiple
        MEAResponseBlocks must be created, with many status messages, Default is False.

    Returns:
        response_group (MEAReponseGroup): A response group object made up of the MEAResponseBlocks
        corresponding to all the datafiles in ls_datafile_names.
    """

    response_blocks = [
        MEAResponseBlock(
            exp_name, datafile_name, ss_version = ss_version, b_LED = b_LED, 
            b_load_fd = b_load_fd, verbose = verbose
            ) for datafile_name in ls_datafile_names
            ]

    return MEAResponseGroup(ls_blocks = response_blocks, b_load_fd = b_load_fd, verbose = verbose)


def check_frame_times(frame_times: np.ndarray | list, frame_rate: float=60.0): 
    """
    Check the frame times for dropped frames.

    Parameters:
        frame_times (ndarray): 1D array of frame times.

        frame_rate (float): frame rate of the stimulus in Hz. Default is 60.

    Returns:
        frame_times (ndarray): 1D array of frame times with dropped frames fixed.
    """
    # check that frame_times is an array not a list. Conver to array if not.
    if not isinstance(frame_times, np.ndarray):
        frame_times = np.array(frame_times)

    # Get the frame durations in milliseconds.
    frame_interval = 1000/frame_rate
    d_frames = np.diff(frame_times)
    # Get the number of frames between transitions/check for drops.
    transition_frames = np.round( d_frames / frame_interval ).astype(np.int32)   # this was backwards... frame_interval/d_frames
                                                                                # prints 1 wherever there is a missing frame and
                                                                                # a zero everywhere else...

    # Check for frame drops.
    if np.amax(transition_frames) > 1:
        n_frames = np.sum(transition_frames)+1
        # print(f'Number of frames: {n_frames}')
        # print(list(transition_frames))

        f_times = np.zeros((n_frames,), dtype=np.float64)
        frame_count = 0
        for idx in range(len(frame_times)-1):
            if transition_frames[idx] > 1:
                this_frame = frame_times[idx]
                next_frame = frame_times[idx+1]
                new_times = np.linspace(this_frame, next_frame, transition_frames[idx], endpoint=False)
                for new_t in new_times:
                    f_times[frame_count] = new_t
                    frame_count += 1
            else:
                f_times[frame_count] = frame_times[idx]
                frame_count += 1
            # Add in the last frame time.
            f_times[-1] = frame_times[-1]
        return f_times, transition_frames
    else: 
        return frame_times, transition_frames
