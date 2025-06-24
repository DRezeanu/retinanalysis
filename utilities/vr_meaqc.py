"""
Functions and classes for QC of MEA data.
Main methods:
ISI rejection
NumSpikes rejection
CRF rejection
EI correlation
"""

import numpy as np
from utilities import vr_spikeoutputs as so
from utilities import vr_spikeplots as sp
from utilities import vr_celltype_io as ctio
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML
import visionloader as vl

def get_acf_from_vcd(vcd: vl.VisionCellDataTable, cell_ids: np.ndarray, 
                     bin_edges=np.linspace(0,300,301)):
    # Modified from symphony_data.py for convenience
    isi = dict()
    for cell in cell_ids:
        try:
            spike_times = vcd.get_spike_times_for_cell(cell) / 20000 * 1000 # ms
        except Exception as e:
            print(f'Error for cell {cell}: {e}')
            spike_times = []
        
        # Compute the interspike interval
        if len(spike_times) > 1:
            isi_tmp = np.diff(spike_times)
            isi[cell] = np.histogram(isi_tmp,bins=bin_edges)[0]
        else:
            isi[cell] = np.zeros((len(bin_edges)-1,)).astype(int)
    
    ids_noise_isi = np.array(list(isi.keys()))
    acf = np.zeros((len(ids_noise_isi), len(bin_edges)-1))
    for idx, n_ID in enumerate(list(isi.keys())):
        if np.sum(isi[n_ID]) > 0:
            acf[idx] = isi[n_ID] / np.sum(isi[n_ID])
        else:
            acf[idx] = isi[n_ID]
    
    return acf

def print_type_summary(df_keep: pd.DataFrame, str_param: str, ls_RGC_labels: list,
                       d_cutoffs: dict=None):
    # Print summary by main cell types
    df_print = df_keep.groupby('cell_type').agg({str_param: 'sum'})
    # Add column for total cells of each type
    df_print['n_total'] = df_keep['cell_type'].value_counts()
    df_print['pct'] = df_print[str_param] / df_print['n_total']
    # pct to string with 2 decimal places
    df_print['pct'] = df_print['pct'].apply(lambda x: f'{x:.2f}')
    df_print = df_print.loc[ls_RGC_labels]
    # Rename str_param column to n_good str_param for clarity
    df_print.rename(columns={str_param: 'n_good'}, inplace=True)

    # If d_cutoffs is provided, add cutoffs to df_print
    if d_cutoffs is not None:
        for str_type in ls_RGC_labels:
            df_print.loc[str_type, 'cutoff'] = d_cutoffs[str_type]
        # If any cutoffs are 0, print warning
        if np.any(df_print['cutoff'] == 0):
            print('Warning: Some cutoffs are 0. Pct targets may not be met.')
    display(df_print)

def filter_low_crf_f1(data: so.SpikeOutputs, crf_datafile, n_percentile=10, contrast=0.05):
    # Filter cells with low F1 amplitude for specified contrast
    print('Filtering low CRF F1 cells...')
    # Load crf datafile
    with open(crf_datafile, 'rb') as f:
        d_crf = pickle.load(f)
    crf_ids = d_crf['cluster_id']
    contrast_vals = d_crf['unique_params']['contrast']
    idx_contrast = np.argwhere(contrast_vals == contrast)[0][0]

    # Get F1 amplitudes for each cell type
    for idx_t, str_type in enumerate(data.ls_RGC_labels):
        type_IDs = data.types.d_main_IDs[str_type]
        
        # arr_nSpikes = np.array([self.vcd.main_datatable[str_ID]['EI'].n_spikes for str_ID in type_IDs])
        type_idx = ctio.map_ids_to_idx(type_IDs, crf_ids)
        arr_f1 = d_crf['4Hz_amp'][type_idx, idx_contrast]
        
        n_cutoff = np.percentile(arr_f1, n_percentile)
        arr_keep = arr_f1 > n_cutoff

        # Update type IDs
        data.types.d_main_IDs[str_type] = type_IDs[arr_keep]
        print(f'{arr_keep.sum()} / {len(type_IDs)} {str_type} cells are kept.')

        # Update good cell IDs by deleting type_IDs not kept
        data.GOOD_CELL_IDS = np.delete(data.GOOD_CELL_IDS, 
                                        np.argwhere(np.isin(data.GOOD_CELL_IDS, type_IDs[~arr_keep])))
    
    data.N_GOOD_CELLS = len(data.GOOD_CELL_IDS)
    data.d_crf = d_crf
    print('Done.')

def remove_dups(data: so.SpikeOutputs, thresh, str_type, b_plot=True,
                sd_mult=1, b_verbose=False):
    type_IDs = data.types.d_main_IDs[str_type].astype(int)
    n_type_cells = len(type_IDs)

    rfs = [(data.d_sta[str_ID]['x0']*data.NOISE_GRID_SIZE, 
            data.d_sta[str_ID]['y0']*data.NOISE_GRID_SIZE) for str_ID in type_IDs]

    # Compute pairwise distance between all RFs
    arr_dist = np.zeros((n_type_cells, n_type_cells))
    arr_dist[:] = np.inf

    dist_idx = (range(n_type_cells), range(n_type_cells))
    for i in range(n_type_cells):
        for j in range(i+1, n_type_cells):
            if i != j:
                arr_dist[i, j] = np.sqrt((rfs[i][0] - rfs[j][0])**2 + (rfs[i][1] - rfs[j][1])**2)

    # Copy of RFs and dist
    dedup_cidx = np.arange(n_type_cells)
    dedup_dist = arr_dist.copy()
    
    # While loop through copy. If distance < thresh, remove from copy.
    ls_already_removed = []
    removed_ids = np.array([])
    while np.any(dedup_dist < thresh):
        # Find min dist
        min_idx = np.unravel_index(np.argmin(dedup_dist), dedup_dist.shape)
        # print(min_idx)
        if b_verbose:
            print(f'Min dist bw {type_IDs[min_idx[0]]} and {type_IDs[dedup_cidx[min_idx[1]]]}: {dedup_dist[min_idx]:.2f}')
        
        # Remove col from copy
        dedup_dist = np.delete(dedup_dist, min_idx[1], axis=1)

        
        # Remove from idx
        # dedup_idx = np.delete(dedup_idx, min_idx[0])
        # mask = np.ones(len(dedup_idx), dtype=bool)
        # mask[min_idx[0]] = False
        # dedup_idx = dedup_idx[mask]
        # dedup_ridx = np.delete(dedup_ridx, min_idx[0])
        dedup_cidx = np.delete(dedup_cidx, min_idx[1])
        dedup_id = np.array([type_IDs[i] for i in dedup_cidx])
        removed_ids = np.setdiff1d(type_IDs, dedup_id)
        new_removed = np.setdiff1d(removed_ids, ls_already_removed)
        ls_already_removed += list(removed_ids)
        if b_verbose:
            print(f'Removed {new_removed}')

    # If removed_ids is empty, print warning
    if len(removed_ids) == 0:
        print(f'Warning: No cells removed for {str_type} at threshold {thresh}.')
    
    # Save deduped IDs
    dedup_id = np.array([type_IDs[i] for i in dedup_cidx])
    data.types.d_main_IDs[str_type+'_dd'] = dedup_id

    # Plot
    if b_plot:
        import utilities.vr_spikeplots as sp
        f, axs = plt.subplots(ncols=2, figsize=(10,5))
        sp.plot_type_rfs(data, [str_type, str_type+'_dd'], axs=axs,
        b_zoom=True, sd_mult=sd_mult)
        data.types.d_main_IDs.pop(str_type+'_dd')

    # Update data
    # if b_update:
        # data.types.d_main_IDs[str_type+'_all'] = type_IDs
        # data.types.d_main_IDs[str_type] = dedup_id

    
    return removed_ids, dedup_id

def find_dup_thresh(data: so.SpikeOutputs, str_type: str, arr_thresh=np.arange(20,100,5)):
    # For each threshold, plot number of cells remaining
    ls_n_cells = []
    for thresh in arr_thresh:
        _, dedup_id = remove_dups(data, thresh, str_type, b_plot=False)
        ls_n_cells.append(len(dedup_id))
    f, ax = plt.subplots()
    ax.plot(arr_thresh, ls_n_cells)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Number of cells remaining')
    # Get current yticks
    yticks = ax.get_yticks()
    # Set to only the integer values
    yticks = np.unique(yticks.astype(int))
    ax.set_yticks(yticks)



class QC(object):
    def __init__(self, data: so.SpikeOutputs, b_noise_only: bool=False,
                 n_refractory_period: float= 1.5 # ms
                 ):
        self.data = data
        self.ls_RGC_labels = data.ls_RGC_labels
        self.N_CELLS = data.N_CELLS
        self.ARR_CELL_IDS = data.ARR_CELL_IDS
        self.GOOD_CELL_IDS = data.GOOD_CELL_IDS
        self.N_GOOD_CELLS = data.N_GOOD_CELLS

        ls_cols = ['cell_id', 'cell_type', 'noise_spikes',
                    'noise_isi_violations', 'crf_f1', 'ei_corr']
        if not b_noise_only:
            ls_cols += ['protocol_spikes', 'protocol_isi_violations']
        self.df_qc = pd.DataFrame(columns=ls_cols)
        
        # Populate df_qc cell_id and cell_type
        self.df_qc['cell_id'] = self.ARR_CELL_IDS

        # Set cell_id as index
        self.df_qc.set_index('cell_id', inplace=True)

        # Get IDs for noise and protocol
        self.noise_ids = np.array(list(data.vcd.main_datatable.keys()))

        # Populate total spike counts
        ls_noisespikes = []
        ls_protocolspikes = []
        ls_celltypes = []
        for n_ID in self.df_qc.index:
            # Check if n_ID in vcd
            if n_ID in data.vcd.main_datatable.keys():
                if 'SpikeTimes' in data.vcd.main_datatable[n_ID].keys():
                    ls_noisespikes.append(len(data.vcd.main_datatable[n_ID]['SpikeTimes']))
                else:
                    ls_noisespikes.append(0)
                    print(f'No SpikeTimes for {n_ID}.')
            else:
                ls_noisespikes.append(0)
            # Check if n_ID in data.types.d_types
            if n_ID in data.types.d_types.keys():
                ls_celltypes.append(data.types.d_types[n_ID])
            else:
                ls_celltypes.append('not in typing.txt')
        
        self.df_qc['cell_type'] = ls_celltypes
        self.df_qc['noise_spikes'] = ls_noisespikes
        self.df_qc['noise_spikes'] = self.df_qc['noise_spikes'].astype(int)
        
        # populate noise ISI violations
        self.n_refractory_period = n_refractory_period
        isi_bin_edges = data.isi[data.str_noise_protocol]['isi_bin_edges']
        isi_bins = np.array([(isi_bin_edges[i], isi_bin_edges[i+1]) for i in range(len(isi_bin_edges)-1)])
        n_bin_max = np.argwhere(isi_bins[:,1] <= n_refractory_period)[-1][0] + 1
        print(f'Using first {n_bin_max} bins for refractory period calculation.')
        print(f'isi_bins: {isi_bins[:n_bin_max]}')

        # Get noise ISI violations
        # noise_isi = data.isi[data.str_noise_protocol]['acf']
        
        # TODO: some cases where the data isi pulled from noise data files 
        # does not match the chunk vcd. Need to investigate.
        # For now use chunk vcd.
        noise_ids = np.array(list(data.d_sta.keys())) # Using d_sta as that has keys with RF fit data.
        noise_isi = get_acf_from_vcd(data.vcd,noise_ids, isi_bin_edges)
        pct_refractory = np.sum(noise_isi[:,:n_bin_max], axis=1) * 100
        self.pct_refractory = pct_refractory
        
        # ids_noise_isi = data.isi[data.str_noise_protocol]['isi_cluster_id']
        self.df_qc.loc[noise_ids, 'noise_isi_violations'] = pct_refractory

        # Get protocol data
        if not b_noise_only:
            self.protocol_ids = np.array(data.spikes['cluster_id'])
            for n_ID in self.df_qc.index:
                # Check if n_ID in data.spikes['cluster_id']
                if n_ID in self.protocol_ids:
                    n_idx = np.argwhere(data.spikes['cluster_id'] == n_ID)[0][0]
                    ls_protocolspikes.append(data.spikes['total_spike_counts'][n_idx])
                else:
                    ls_protocolspikes.append(0)
            self.df_qc['protocol_spikes'] = ls_protocolspikes
            self.df_qc['protocol_spikes'] = self.df_qc['protocol_spikes'].astype(int)
            
            # populate protocol ISI violations
            protocol_isi = data.isi[data.str_protocol]['acf']
            pct_refractory = np.sum(protocol_isi[:,:n_bin_max], axis=1) * 100
            self.df_qc.loc[self.protocol_ids, 'protocol_isi_violations'] = pct_refractory

            # print summary of cells with > 0 spikes in both noise and protocol
            ids_both = self.df_qc[(self.df_qc['noise_spikes'] > 0) & (self.df_qc['protocol_spikes'] > 0)].index.values
            n_both = len(ids_both)

            print(f'{len(noise_ids)} noise cells, {len(self.protocol_ids)} protocol cells, {n_both} cells with >0 sps in both.')
            for str_type in self.ls_RGC_labels:
                n_noise = np.sum(self.df_qc.loc[noise_ids, 'cell_type'] == str_type)
                n_protocol = np.sum(self.df_qc.loc[self.protocol_ids, 'cell_type'] == str_type)
                n_both = np.sum(self.df_qc.loc[ids_both, 'cell_type'] == str_type)
                print(f'{str_type}: {n_noise} noise, {n_protocol} protocol, {n_both} both.')
            
            # Keep only cells with > 0 protocol and noise spikes
            self.df_qc = self.df_qc[(self.df_qc['noise_spikes'] > 0) & (self.df_qc['protocol_spikes'] > 0)]

            self.data.update_ids(good_ids=ids_both)

        # Dict for threshold sets
        df_keep = self.df_qc.copy()
        for str_col in self.df_qc.columns[1:]:
            df_keep[str_col] = False
        self.d_thresh = {'set1': {'df_keep': df_keep}}

    def set_abs_thresh(self, str_set, str_param, n_thresh, b_keep_below=True):
        # Set threshold for str_param in str_set
        d_thresh_vals = {'n_thresh': n_thresh, 'b_keep_below': b_keep_below, 'b_by_type': False}

        # Apply absolute threshold to df_keep
        str_compare = '<' if b_keep_below else '>'
        print(f'Setting {str_param} threshold at {str_compare} {n_thresh}.')
        df_qc = self.df_qc
        df_keep = self.d_thresh[str_set]['df_keep']
        if b_keep_below:
            df_keep[str_param] = df_qc[str_param] < n_thresh
        else:
            df_keep[str_param] = df_qc[str_param] > n_thresh
        d_thresh_vals['n_cells'] = df_keep[str_param].sum()
        print(f'{d_thresh_vals["n_cells"]}/{self.N_CELLS} total cells kept.')

        for str_type in self.data.ls_RGC_labels:
            n_total = df_keep[df_keep['cell_type'] == str_type].shape[0]
            d_thresh_vals[f'n_{str_type}'] = df_keep[df_keep['cell_type'] == str_type][str_param].sum()
            d_thresh_vals[f'pct_{str_type}'] = d_thresh_vals[f'n_{str_type}'] / n_total
            #print(f'{str_type}: {d_thresh_vals[f"n_{str_type}"]} / {n_total} = {d_thresh_vals[f"pct_{str_type}"]:.2f}')

        self.d_thresh[str_set][str_param] = d_thresh_vals

        # Print summary by main cell types
        print_type_summary(df_keep, str_param, self.ls_RGC_labels)

    
    def set_pct_thresh_by_type(self, str_set, str_param, n_top_pct):
        # Set threshold for str_param in str_set
        d_thresh_vals = {'n_top_pct': n_top_pct, 'b_by_type': True}

        # Apply percentile threshold to df_keep
        df_qc = self.df_qc
        df_keep = self.d_thresh[str_set]['df_keep']

        print(f'Setting {str_param} top {n_top_pct} percentile threshold.')
        for str_type in self.data.ls_RGC_labels:
            df_qc_slice = df_qc[df_qc['cell_type'] == str_type]
            n_total = df_qc_slice.shape[0]
            slice_idx = df_qc_slice.index
            arr_param = df_qc_slice[str_param]
            n_cutoff = np.percentile(arr_param, 100-n_top_pct)

            df_keep.loc[slice_idx, str_param] = arr_param > n_cutoff
            
            # TODO move away from this dict to a dataframe
            d_thresh_vals[str_type] = n_cutoff
            d_thresh_vals[f'n_{str_type}'] = df_keep.loc[slice_idx, str_param].sum()
            d_thresh_vals[f'pct_{str_type}'] = d_thresh_vals[f'n_{str_type}'] / n_total
            #print(f'{str_type}: {d_thresh_vals[f"n_{str_type}"]} / {n_total} = {d_thresh_vals[f"pct_{str_type}"]:.2f}')
        
        # NaN to True in df_keep for cell types not in ls_RGC_labels
        df_keep[str_param].fillna(True, inplace=True)
        self.d_thresh[str_set][str_param] = d_thresh_vals

        # Print summary by main cell types
        print_type_summary(df_keep, str_param, self.ls_RGC_labels, d_cutoffs=d_thresh_vals)
    
    def plot_dist_by_type(self, str_param, ax=None, b_plot_thresh=False, str_set='set1'):
        if ax is None:
            f, ax = plt.subplots()
        df_plot = self.df_qc[self.df_qc['cell_type'].isin(self.data.ls_RGC_labels)]
        sns.boxplot(x='cell_type', y=str_param, data=df_plot, ax=ax,
            order=self.data.ls_RGC_labels)
        ax.set_ylabel('')
        ax.set_title(str_param)

        if b_plot_thresh and str_set in self.d_thresh.keys():
            # Plot threshold
            d_vals = self.d_thresh[str_set][str_param]
            
            if d_vals['b_by_type']:
                for idx_t, str_type in enumerate(self.data.ls_RGC_labels):
                    ax.axhline(d_vals[str_type], color=f'C{idx_t}', linestyle='--')
                    ax.text(idx_t, d_vals[str_type], f'{d_vals[str_type]:.2f}', 
                        color='k', fontsize=8, ha='center', va='bottom')
                    # ax.text(idx_t, d_vals[str_type], f'{pct_cells:.2f}', 
                    #     color='k', fontsize=8, ha='center', va='bottom')
                    
            else:
                ax.axhline(d_vals['n_thresh'], color='k', linestyle='--')
                ax.set_title(f'{str_param} threshold: {d_vals["n_thresh"]}')        
        return ax
    
    def plot_pct_by_type(self, str_param, ax=None, str_set='set1'):
        if ax is None:
            f, ax = plt.subplots()

        d_vals = self.d_thresh[str_set][str_param]
        pct = np.array([d_vals[f'pct_{str_type}'] for str_type in self.ls_RGC_labels])
        df_pct = pd.DataFrame({'cell_type': self.ls_RGC_labels, 'pct': pct})
        ax=sns.barplot(x='cell_type', y='pct', data=df_pct, ax=ax, order=self.ls_RGC_labels)
        ax.bar_label(ax.containers[0], fmt='%.2f')
        ax.set_ylabel('Percent cells kept')
        if d_vals['b_by_type']:
            ax.set_title(f'{str_param} top {d_vals["n_top_pct"]} percentile')
        else:
            ax.set_title(f'{str_param} threshold: {d_vals["n_thresh"]}')
        return ax
    
    def plot_ncells_by_type(self, str_param, ax=None, str_set='set1'):
        if ax is None:
            f, ax = plt.subplots()

        d_vals = self.d_thresh[str_set][str_param]
        n_cells = np.array([d_vals[f'n_{str_type}'] for str_type in self.ls_RGC_labels])
        df_ncells = pd.DataFrame({'cell_type': self.ls_RGC_labels, 'n_cells': n_cells})
        ax=sns.barplot(x='cell_type', y='n_cells', data=df_ncells, ax=ax, order=self.ls_RGC_labels)
        ax.bar_label(ax.containers[0], fmt='%.0f')
        ax.set_ylabel('Number of cells kept')
        if d_vals['b_by_type']:
            ax.set_title(f'{str_param} top {d_vals["n_top_pct"]} percentile')
        else:
            ax.set_title(f'{str_param} threshold: {d_vals["n_thresh"]}')
        return ax
    
    
    def plot_report(self, str_param, str_set='set1'):
        f, axs = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
        self.plot_dist_by_type(str_param, ax=axs[0], b_plot_thresh=True, str_set=str_set)
        self.plot_pct_by_type(str_param, ax=axs[1], str_set=str_set)
        # self.plot_ncells_by_type(str_param, ax=axs[1], str_set=str_set)

        return f, axs
    
    def get_intersection_cells(self, ls_params: list, str_set='set1'):
        # Get cell IDs (index of df_keep) that are True for all parameters
        df_keep = self.d_thresh[str_set]['df_keep']
        # df_keep = df_keep.select_dtypes(include='bool') # This is not robust.
        df_keep = df_keep[ls_params]
        arr_keep = df_keep.values

        arr_intersection = np.all(arr_keep, axis=1)
        return df_keep.index[arr_intersection].values

    def plot_mosaics(self, ls_params, str_set='set1', sd_mult=1):
        # Plot mosaics for all cells, and for intersection cells
        _ = sp.plot_type_rfs(self.data, d_IDs=self.data.types.d_main_IDs, b_zoom=True,
                             sd_mult=sd_mult)

        good_ids = self.get_intersection_cells(ls_params, str_set)
        d_good_IDs = {}
        for str_type in self.ls_RGC_labels:
            d_good_IDs[str_type] = np.intersect1d(self.data.types.d_main_IDs[str_type], good_ids)
        _ = sp.plot_type_rfs(self.data, d_IDs=d_good_IDs, b_zoom=True, sd_mult=sd_mult)

    def update_ids(self, ls_params: list, str_set: str):
        # Update GOOD_CELL_IDS with intersection cells
        good_ids = self.get_intersection_cells(ls_params, str_set)
        self.data.update_ids(good_ids)
        # self.data.GOOD_CELL_IDS = good_ids
        # self.data.N_GOOD_CELLS = len(good_ids)
        # print(f'Updating GOOD_CELL_IDS to {self.data.N_GOOD_CELLS} cells.')
        
        # # Update d_main_IDs
        # for str_type in self.data.types.d_main_IDs.keys():
        #     type_ids = self.data.types.d_main_IDs[str_type]
        #     new_ids = np.intersect1d(type_ids, good_ids)
        #     self.data.types.d_main_IDs[str_type] = new_ids
        #     print(f'{str_type}: {len(new_ids)}/{len(type_ids)}')
        