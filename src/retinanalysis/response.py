import visionloader as vl
import retinanalysis.datajoint_utils as dju
import retinanalysis.vision_utils as vu
import retinanalysis.schema as schema
from retinanalysis.settings import NAS_DATA_DIR
import os
import numpy as np
import pandas as pd

SAMPLE_RATE = 20000 # MEA DAQ sample rate in Hz

class ResponseBlock:
    def __init__(self, exp_name, datafile_name, ss_version: str = 'kilosort2.5'):
        self.exp_name = exp_name
        self.datafile_name = datafile_name
        self.ss_version = ss_version
        self.protocol_name = vu.get_protocol_from_datafile(self.exp_name, self.datafile_name)
        self.vcd = vu.get_protocol_vcd(self.exp_name, self.datafile_name, self.ss_version)
        self.cell_ids = self.vcd.get_cell_ids()
        self.get_spike_times()

    def get_spike_times(self):
        d_timing = dju.get_mea_epochblock_timing(self.exp_name, self.datafile_name)
        d_spike_times = {'cell_id': [], 'spike_times': []}
        epoch_starts = d_timing['epoch_starts']
        epoch_ends = d_timing['epoch_ends']
        n_samples = d_timing['n_samples']
        frame_times_ms = d_timing['frame_times_ms']

        self.n_epochs = len(epoch_starts)
        for cell_id in self.cell_ids:
            all_spike_times = np.zeros(self.n_epochs, dtype=object)
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

                all_spike_times[i] = e_sts
            d_spike_times['cell_id'].append(cell_id)
            d_spike_times['spike_times'].append(all_spike_times)
        self.df_spike_times = pd.DataFrame(d_spike_times)
        self.df_spike_times.set_index('cell_id', inplace=True)

    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  datafile_name: {self.datafile_name}\n"
        str_self += f"  protocol_name: {self.protocol_name}\n"
        str_self += f"  ss_version: {self.ss_version}\n"
        str_self += f"  n_epochs: {self.n_epochs}\n"
        str_self += f"  cell_ids of length: {len(self.cell_ids)}\n"
        str_self += f"  df_spike_times with shape: {self.df_spike_times.shape}\n"
        return str_self