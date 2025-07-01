import numpy as np
import pandas as pd
from retinanalysis.response import ResponseBlock
from retinanalysis.analysis_chunk import AnalysisChunk
import visionloader as vl

def get_nsps(vcd: vl.VisionCellDataTable, cell_ids: list):
    ls_nsps = []
    for n_ID in cell_ids:
        if 'SpikeTimes' in vcd.main_datatable[n_ID].keys():
            ls_nsps.append(len(vcd.get_spike_times_for_cell(n_ID)))
        else:
            ls_nsps.append(0)
            print(f'No SpikeTimes for {n_ID}.')
    return ls_nsps

def get_isi(vcd: vl.VisionCellDataTable, cell_ids: list, bin_edges: np.array):
    isi_dict = dict()
    for n_ID in cell_ids:
        try:
            spike_times = vcd.get_spike_times_for_cell(n_ID) / 20000 * 1000 # ms
        except Exception as e:
            print(f'Error for cell {n_ID}: {e}')
            spike_times = []
        
        # Compute the interspike interval
        if len(spike_times) > 1:
            isi_tmp = np.diff(spike_times)
            isi_dict[n_ID] = np.histogram(isi_tmp,bins=bin_edges)[0].astype(float)
            # Normalize by sum
            isi_dict[n_ID] /= np.sum(isi_dict[n_ID])
        else:
            isi_dict[n_ID] = np.zeros((len(bin_edges)-1,)).astype(int)
    
    return isi_dict

def get_pct_refractory(isi_dict, n_bin_max):
    # Make array of [cells, bins]
    isi = np.array(list(isi_dict.values()))
    pct_refractory = np.sum(isi[:,:n_bin_max], axis=1) * 100
    return pct_refractory

def get_ei_corr(vcd1: vl.VisionCellDataTable, vcd2: vl.VisionCellDataTable, 
                match_dict: dict):
    ei_corrs = []
    for id1 in match_dict.keys():
        ei1 = vcd1.get_ei_for_cell(id1).ei.flatten()
        id2 = match_dict[id1]
        ei2 = vcd2.get_ei_for_cell(id2).ei.flatten()
        r = np.corrcoef(ei1, ei2)[0,1]
        ei_corrs.append(r)
    return ei_corrs

class MEAQC():
    def __init__(self, rb: ResponseBlock, ac: AnalysisChunk, match_dict: dict,
                 refractory_period_ms: float=1.5):
        self.rb = rb
        self.ac = ac
        self.match_dict = match_dict
        # Assuming different sorting chunks for now
        # And assuming rb is not for noise protocol
        self.refractory_period_ms = refractory_period_ms
        
        isi_bin_edges = np.linspace(0,300,601)
        isi_bins = np.array([(isi_bin_edges[i], isi_bin_edges[i+1]) for i in range(len(isi_bin_edges)-1)])
        isi_bin_max = np.argwhere(isi_bins[:,1] <= refractory_period_ms)[-1][0] + 1
        print(f'Using {refractory_period_ms} ms refractory period.')
        print(f'Using first {isi_bin_max} bins for refractory period calculation.')
        self.isi_bin_edges = isi_bin_edges
        self.isi_bin_max = isi_bin_max
        
        self.df_qc = self.get_df_qc()

    def get_df_qc(self):
        ls_cols = ['cell_id', 'cell_type', 'noise_spikes',
                    'noise_isi_violations', 'crf_f1', 'ei_corr',
                    'protocol_spikes', 'protocol_isi_violations',
                    'analysis_chunk_cell_id']
        df_qc = pd.DataFrame(columns=ls_cols)
        df_qc['cell_id'] = self.match_dict.values()
        df_qc['analysis_chunk_cell_id'] = self.match_dict.keys()
        df_qc['cell_type'] = self.rb.df_spike_times.loc[df_qc['cell_id'], 'cell_type']
        
        df_qc['protocol_spikes'] = get_nsps(self.rb.vcd, df_qc['cell_id'].values)
        df_qc['noise_spikes'] = get_nsps(self.ac.vcd, df_qc['analysis_chunk_cell_id'].values)

        self.protocol_isi = get_isi(self.rb.vcd, df_qc['cell_id'].values, self.isi_bin_edges)
        self.noise_isi = get_isi(self.ac.vcd, df_qc['analysis_chunk_cell_id'].values, self.isi_bin_edges)
        df_qc['noise_isi_violations'] = get_pct_refractory(self.noise_isi, self.isi_bin_max)
        df_qc['protocol_isi_violations'] = get_pct_refractory(self.protocol_isi, self.isi_bin_max)

        df_qc['ei_corr'] = get_ei_corr(self.ac.vcd, self.rb.vcd, self.match_dict)

        df_qc = df_qc.set_index('cell_id')

        return df_qc

