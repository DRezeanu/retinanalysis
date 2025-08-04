import retinanalysis.utils.datajoint_utils as dju
import retinanalysis.utils.vision_utils as vu
from retinanalysis.utils.spike_detector import detector
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle

SAMPLE_RATE = 20000 # MEA DAQ sample rate in Hz

def check_frame_times(frame_times: np.ndarray, frame_rate: float=60.0): 
    """
    Check the frame times for dropped frames.

    Parameters:
        frame_times: 1D array of frame times.
        frame_rate: frame rate of the stimulus.

    Returns:
        frame_times: 1D array of frame times with dropped frames fixed.
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

class ResponseBlock:
    """
    Generic class for single cell or MEA response blocks. 
    """
    def __init__(self, exp_name: str=None, block_id: int=None, h5_file: str=None,
                 pkl_file: str=None):
        if pkl_file is None:
            print(f"Initializing ResponseBlock for {exp_name} block {block_id}")
            if exp_name is None or block_id is None:
                raise ValueError("Either exp_name and block_id or pkl_file must be provided.")
        else:
            print(f"Initializing ResponseBlock for {exp_name} block {block_id} from pickle file.")
            # Load from pickle file if string, otherwise must be a dict
            if isinstance(pkl_file, str):
                with open(pkl_file, 'rb') as f:
                    d_out = pickle.load(f)
            else:
                d_out = pkl_file
                pkl_file = "input dict."
            self.__dict__.update(d_out)
            print(f"ResponseBlock loaded from {pkl_file}")
            return

        self.exp_name = exp_name
        self.block_id = block_id    
        self.h5_file = h5_file
        self.d_timing = dju.get_epochblock_timing(self.exp_name, self.block_id)
        frame_data, frame_sample_rate = dju.get_epochblock_frame_data(self.exp_name, self.block_id, str_h5=self.h5_file)    
        self.frame_data = frame_data
        self.frame_sample_rate = frame_sample_rate
    
    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  block_id: {self.block_id}\n"
        str_self += f"  d_timing with keys: {list(self.d_timing.keys())}\n"
        str_self += f"  frame_sample_rate: {self.frame_sample_rate} Hz\n"
        str_self += f"  frame_data shape: {self.frame_data.shape}\n"
        if self.h5_file is not None:
            str_self += f"  h5_file: {self.h5_file}\n"
        return str_self

    def export_to_pkl(self, file_path: str):
        d_out = self.__dict__.copy()
        with open(file_path, 'wb') as f:
            pickle.dump(d_out, f)
        print(f"ResponseBlock exported to {file_path}")


class SCResponseBlock(ResponseBlock):
    def __init__(self, exp_name: str=None, block_id: int=None, h5_file: str=None,
                 pkl_file: str=None, b_spiking: bool=False, **detector_kwargs):
        super().__init__(exp_name=exp_name, block_id=block_id, h5_file=h5_file, pkl_file=pkl_file)
        if pkl_file is not None:
            return

        self.b_spiking = b_spiking
        amp_data, sample_rate = dju.get_epochblock_amp_data(self.exp_name, self.block_id, str_h5=self.h5_file)
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
    def __init__(self, exp_name: str=None, datafile_name: str=None, ss_version: str = 'kilosort2.5', 
                 pkl_file: str=None, h5_file: str=None):
        # If pkl_file is provided, block_id can be None.
        block_id = None
        if pkl_file is None:
            # Either pkl_file or exp_name and datafile_name must be provided
            if exp_name is None or datafile_name is None:
                raise ValueError("Either exp_name and datafile_name or pkl_file must be provided.")
            else:
                # If exp_name and datafile_name are provided, get block_id from datafile_name
                block_id = dju.get_block_id_from_datafile(exp_name, datafile_name)
                # Set the ss_version and datafile_name for loading VCD.
                self.ss_version = ss_version
                self.datafile_name = datafile_name
        
        super().__init__(exp_name=exp_name, block_id=block_id, pkl_file=pkl_file, h5_file=h5_file)
        self.vcd = vu.get_protocol_vcd(self.exp_name, self.datafile_name, self.ss_version)

        # If pkl_file is provided, everything else is already loaded in parent init.
        if pkl_file is not None:
            return
        
        self.protocol_name = vu.get_protocol_from_datafile(self.exp_name, self.datafile_name)
        self.cell_ids = self.vcd.get_cell_ids()
        self.get_spike_times()

    def get_spike_times(self):
        d_spike_times = {'cell_id': [], 'spike_times': []}

        epoch_starts = self.d_timing['epochStarts']
        epoch_ends = self.d_timing['epochEnds']

        #i think if symphony crashed during recording, there might be more 1 more start than end
        #this ignores the partial epoch
        if len(epoch_ends) == len(epoch_starts)-1:
            epoch_starts = epoch_starts[:len(epoch_ends)]
        elif len(epoch_ends) != len(epoch_starts):
            raise ValueError("Mismatch in number of epoch starts and ends.")

        # n_samples = self.d_timing['n_samples']
        # frame_times_ms = self.d_timing['frame_times_ms']

        self.n_epochs = len(epoch_starts)
        for cell_id in self.cell_ids:
            all_spike_times = []
            # STs in samples
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
        self.df_spike_times = pd.DataFrame(d_spike_times)

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
    
    def bin_spike_times_by_frames(self, stride: int=1):
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
        self.binned_time = np.arange(0, n_max_bins) / self.bin_rate * 1000 # in ms


    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  datafile_name: {self.datafile_name}\n"
        str_self += f"  protocol_name: {self.protocol_name}\n"
        str_self += f"  ss_version: {self.ss_version}\n"
        str_self += f"  n_epochs: {self.n_epochs}\n"
        str_self += f"  cell_ids of length: {len(self.cell_ids)}\n"
        str_self += f"  df_spike_times with shape: {self.df_spike_times.shape}\n"
        str_self += f"  block_id: {self.block_id}\n"
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