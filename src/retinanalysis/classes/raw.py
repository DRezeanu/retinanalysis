import bin2py
import numpy as np
import os
import matplotlib.pyplot as plt
from retinanalysis.classes.response import MEAResponseBlock
from retinanalysis.utils import RAW_DIR
from retinanalysis import ei_utils as eiu

# Constants
RW_BLOCKSIZE = 100000  # Block size for reading data
TTL_THRESHOLD = 1000
SAMPLE_RATE = 20000 # Hz
class RawTraces:
    def __init__(self, rb: MEAResponseBlock):
        self.binpath = os.path.join(RAW_DIR, rb.exp_name, rb.datafile_name)
        self.d_timing = rb.d_timing
        self.sorted_electrodes = eiu.sort_electrode_map(rb.vcd.get_electrode_map())
        self.data = None
        self.ttl_times = None
        self.ttl_samples = None
        self.sample_rate = SAMPLE_RATE  # Hz
        self.epoch_idx = None

    def load_ttl_data(self, start_sample, end_sample, verbose=False):
        with bin2py.PyBinFileReader(self.binpath, chunk_samples=RW_BLOCKSIZE, is_row_major=True) as pbfr:
            array_id = pbfr.header.array_id
            total_samples = pbfr.length

            # Set end_sample to the total length if not specified
            if end_sample is None:
                end_sample = total_samples

            # Validate sample range
            if start_sample < 0 or end_sample > total_samples or start_sample >= end_sample:
                raise ValueError("Invalid start_sample or end_sample range.")

            query_samples = end_sample - start_sample
            if verbose:
                print(f"Querying {query_samples} samples from {start_sample} to {end_sample} for array {array_id}.")
                print(f'Queried time: {query_samples / SAMPLE_RATE} seconds')
                print(f'From {start_sample / SAMPLE_RATE} to {end_sample / SAMPLE_RATE} seconds')

            data = np.zeros(query_samples, dtype=np.float32)
            for start_idx in range(start_sample, end_sample, RW_BLOCKSIZE):
                n_samples_to_get = min(RW_BLOCKSIZE, end_sample - start_idx)
                samples = pbfr.get_data_for_electrode(0, start_idx, n_samples_to_get)
                data[start_idx - start_sample:start_idx - start_sample + n_samples_to_get] = samples

        self.ttl_samples = data


    def load_bin_data(self, start_sample=0, end_sample=None, verbose=False):
        """
        Load raw .bin data into a NumPy array.

        Parameters:
            binpath (str): Path to the .bin file.
            start_sample (int): Starting sample index (default: 0).
            end_sample (int): Ending sample index (default: None, reads till the end).

        Returns:
            np.ndarray: Loaded data as a NumPy array of shape [electrodes, samples].
        """
        with bin2py.PyBinFileReader(self.binpath, chunk_samples=RW_BLOCKSIZE, is_row_major=True) as pbfr:
            # Determine the number of electrodes and total samples
            n_channels = pbfr.num_electrodes
            total_samples = pbfr.length
            
            if verbose:
                print(f"Number of electrodes: {n_channels}, Total samples: {total_samples}.")
                print(f"Total time: {total_samples / SAMPLE_RATE} seconds")
                print(f"Sample rate: {SAMPLE_RATE} Hz")

            # Set end_sample to the total length if not specified
            if end_sample is None:
                end_sample = total_samples

            # Validate sample range
            if start_sample < 0 or end_sample > total_samples or start_sample >= end_sample:
                raise ValueError("Invalid start_sample or end_sample range.")

            query_samples = end_sample - start_sample
            if verbose:
                print(f"Querying {query_samples} samples from {start_sample} to {end_sample}.")
                print(f'Queried time: {query_samples / SAMPLE_RATE} seconds')
                print(f'From {start_sample / SAMPLE_RATE} to {end_sample / SAMPLE_RATE} seconds')
                
            # Preallocate array for the data
            data = np.zeros((n_channels, query_samples), dtype=np.float32)

            ttl_times_buffer = []
            ttl_samples = np.zeros((query_samples,), dtype=np.float32)
            # Read data in chunks
            for start_idx in range(start_sample, end_sample, RW_BLOCKSIZE):
                n_samples_to_get = min(RW_BLOCKSIZE, end_sample - start_idx)
                chunk = pbfr.get_data(start_idx, n_samples_to_get)

                # Extract TTL data (channel 0) and compute TTL times
                ttl_samples = chunk[0, :]
                below_threshold = (ttl_samples < -TTL_THRESHOLD)
                above_threshold = np.logical_not(below_threshold)
                below_to_above = np.logical_and.reduce([
                    below_threshold[:-1],
                    above_threshold[1:]
                ])
                trigger_indices = np.argwhere(below_to_above) + start_idx
                ttl_times_buffer.append(trigger_indices[:, 0])

                # Populate the data matrix (exclude channel 0)
                data[:, start_idx - start_sample:start_idx - start_sample + n_samples_to_get] = chunk[1:, :]
                # ttl_samples[start_idx - start_sample:start_idx - start_sample + n_samples_to_get] = chunk[0, :]

            # Concatenate TTL times
            ttl_times = np.concatenate(ttl_times_buffer, axis=0)
        
        if verbose:
            print(f'Data shape: {data.shape}')
        # print(f'TTL times shape: {ttl_times.shape}')
        self.data = data
        self.ttl_times = ttl_times
        self.ttl_samples = ttl_samples
        
    def load_epoch_index(self, epoch_idx, verbose=True):
        epoch_start = self.d_timing['epochStarts'][epoch_idx]
        epoch_end = self.d_timing['epochEnds'][epoch_idx]
        self.load_bin_data(start_sample=epoch_start, end_sample=epoch_end, verbose=verbose)
        self.epoch_idx = epoch_idx

def plot_sts_over_trace(rt: RawTraces, rb: MEAResponseBlock, 
                        cell_id, epoch_idx, start_time=0, end_time=None,
                        n_highlight_width=18, channel_idx:int=None, ax=None):
    # Load epoch_idx if needed
    if rt.epoch_idx != epoch_idx:
        rt.load_epoch_index(epoch_idx, verbose=True)
    
    # Get max amplitude channel for this cell_id
    if channel_idx is None:
        top_idx = eiu.get_top_electrodes(cell_id, rb.vcd, n_markers=1, b_sort=False)[0]
        channel_idx = rt.sorted_electrodes[top_idx]

    raw_ts = rt.data[channel_idx, :]
    time = np.arange(len(raw_ts)) / rt.sample_rate # in seconds
    
    if end_time is None:
        end_time = time[-1]
    
    mask = np.where((time >= start_time) & (time <= end_time))[0]
    time = time[mask]
    raw_ts = raw_ts[mask]
    # print(f'Raw trace shape: {raw_ts.shape}, Time shape: {time.shape}')
    # print(f'Channel index: {channel_idx}, Cell ID: {cell_id}, Epoch index: {epoch_idx}')
    # print(f'Time range: {start_time} to {end_time} seconds, Mask shape: {mask.shape}')
    # return
    if ax is None:
        f, ax = plt.subplots(figsize=(12,6))
    ax.plot(time, raw_ts)
    ax.set_title(f'Cell {cell_id}, Channel {channel_idx}, Epoch {epoch_idx}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Raw Signal')
    
    df_st = rb.df_spike_times
    cell_idx = np.where(df_st['cell_id'] == cell_id)[0]
    if len(cell_idx) == 0:
        raise ValueError(f'Cell ID {cell_id} not found in response block {rb.exp_name}.')
    cell_idx = cell_idx[0]
    sts = df_st.at[cell_idx, 'spike_times'][epoch_idx]
    if len(sts) == 0:
        print(f'No spikes found for cell {cell_id} in epoch {epoch_idx}.')
        return
    # Convert from ms back to samples
    sts = np.round(sts * rt.sample_rate / 1000).astype(int)
    # Keep only spike times in time range given by sample index
    sts = sts[(sts>=mask[0]) & (sts<=mask[-1])]
    sts -= mask[0]  # Adjust spike times to the new time range
    ax.scatter(time[sts], raw_ts[sts], color='red', zorder=10)
    for st in sts:
        ax.plot(time[st:st+n_highlight_width],
                raw_ts[st:st+n_highlight_width], color='red', lw=2, alpha=0.5)
    
    



