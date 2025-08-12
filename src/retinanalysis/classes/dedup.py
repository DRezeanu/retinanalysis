import numpy as np
from retinanalysis.classes.response import MEAResponseBlock
from retinanalysis.classes.analysis_chunk import AnalysisChunk
import retinanalysis.utils.vision_utils as vu
import retinanalysis.utils.ei_utils as eu
import os
from retinanalysis.config.settings import DATA_DIR
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
from typing import Union
import pandas as pd
from matplotlib_venn import venn3

def generate_extended_pairings(pairs: set):
    '''
    Generates extended pairings for a set of cell ID pairs.
    Args:
        - pairs: set of tuples containing cell IDs.
    Returns:
        - extended_pairs: set of tuples containing all cell ids that are transitively connected.
    '''
    pairs_dict = {}
    for a, b in pairs:
        if a not in pairs_dict:
            pairs_dict[a] = set()
        if b not in pairs_dict:
            pairs_dict[b] = set()
        pairs_dict[a].add(b)
        pairs_dict[b].add(a)

    extended = set()
    for origin in pairs:
        a, b = origin

        paired_w_a = pairs_dict.get(a, set())
        paired_w_b = pairs_dict.get(b, set())
        all_paired = paired_w_a.union(paired_w_b)
        all_paired_tuple = tuple(sorted(all_paired))
        extended.add(all_paired_tuple)

    return extended


def compare_ei_methods(block: Union[AnalysisChunk,MEAResponseBlock], ei_threshold: float=0.8, axs=None):
    
    methods = ['full', 'space', 'power']
    full_ei_autocorr, ei_full_pairs = get_ei_autocorrelation(block,ei_method=methods[0], ei_threshold=ei_threshold)
    space_ei_autocorr, ei_space_pairs = get_ei_autocorrelation(block,ei_method=methods[1], ei_threshold=ei_threshold)
    power_ei_autocorr, ei_power_pairs = get_ei_autocorrelation(block,ei_method=methods[2], ei_threshold=ei_threshold)

    ig_zero_full = full_ei_autocorr[np.triu_indices_from(full_ei_autocorr, k=1)]
    ig_zero_space = space_ei_autocorr[np.triu_indices_from(space_ei_autocorr, k=1)]
    ig_zero_power = power_ei_autocorr[np.triu_indices_from(power_ei_autocorr, k=1)]

    full_thresh = ei_threshold
    space_thresh = ei_threshold
    power_thresh = ei_threshold

    # full_thresh = np.quantile(ig_zero_full, 0.9999)
    # space_thresh = np.quantile(ig_zero_space, 0.9999)
    # power_thresh = np.quantile(ig_zero_power, 0.9999)

    # full_ei_autocorr, ei_full_pairs = get_ei_autocorrelation(block,ei_method=methods[0], ei_threshold=full_thresh)
    # space_ei_autocorr, ei_space_pairs = get_ei_autocorrelation(block,ei_method=methods[1], ei_threshold=space_thresh)
    # power_ei_autocorr, ei_power_pairs = get_ei_autocorrelation(block,ei_method=methods[2], ei_threshold=power_thresh)

    if axs is None:
        fig, axs  = plt.subplots(2,4, figsize=(20,15))
    im = axs[0,0].imshow(full_ei_autocorr, cmap='hot', interpolation='nearest')
    im1 = axs[0,1].imshow(space_ei_autocorr, cmap='hot', interpolation='nearest')
    im2 = axs[0,2].imshow(power_ei_autocorr, cmap='hot', interpolation='nearest')
    fig.colorbar(im, ax=axs[0,0], orientation='vertical', fraction=0.02, pad=0.04)
    fig.colorbar(im1, ax=axs[0,1], orientation='vertical', fraction=0.02, pad=0.04)
    fig.colorbar(im2, ax=axs[0,2], orientation='vertical', fraction=0.02, pad=0.04)

    venn3([set(ei_full_pairs), set(ei_space_pairs), set(ei_power_pairs)], set_labels = methods, ax=axs[0,3])
    fig.suptitle('High EI Correlation Pairs by Method', fontsize=16)
    for i in range(3):
        axs[0,i].set_title(f'{methods[i]} method')
        axs[0,i].set_xlabel('Cell ID')
        axs[0,i].set_ylabel('Cell ID')

    axs[1,0].hist(ig_zero_full, bins=50, color='blue', alpha=0.7)
    axs[1,0].set_title('full method')
    axs[1,0].set_xlabel('R')
    axs[1,0].set_ylabel('Count')
    axs[1,0].semilogy()
    axs[1,0].axvline(full_thresh, color='red', linestyle='--', label=f'Threshold: {full_thresh:.2f}')
    axs[1,0].legend()
    axs[1,1].hist(ig_zero_space, bins=50, color='blue', alpha=0.7)
    axs[1,1].set_title('space method')
    axs[1,1].set_xlabel('R')
    axs[1,1].set_ylabel('Count')
    axs[1,1].semilogy()
    axs[1,1].axvline(space_thresh, color='red', linestyle='--', label=f'Threshold: {space_thresh:.2f}')
    axs[1,1].legend()
    axs[1,2].hist(ig_zero_power, bins=50, color='blue', alpha=0.7)
    axs[1,2].set_title('power method')
    axs[1,2].set_xlabel('R')
    axs[1,2].set_ylabel('Count')
    axs[1,2].semilogy()
    axs[1,2].axvline(power_thresh, color='red', linestyle='--', label=f'Threshold: {power_thresh:.2f}')
    axs[1,2].legend()

    fig.delaxes(axs[1,3])


    plt.tight_layout()
    plt.subplots_adjust(top=0.98)
    return axs, ei_full_pairs, ei_space_pairs, ei_power_pairs

def get_sm_autocorrelation(ac: AnalysisChunk, sm_threshold: float):
    '''
    Computes the autocorrelation of spatial maps in an AnalysisChunk.
    Args:
        - ac: AnalysisChunk object.
        - threshold: float, the autocorrelation threshold above which pairs of spatial maps are considered highly correlated.
    Returns:
        - sm_autocorr: 2D numpy array of autocorrelation values for spatial maps.
        - high_sm_pairs: set of tuples containing cell IDs of spatial maps with autocorrelation above the specified threshold.
    '''
    if not isinstance(ac, AnalysisChunk):
        raise ValueError("Spatial map autocorrelation can only be computed for AnalysisChunk objects.")

    sm_flat = [sm.flatten() for sm in ac.d_spatial_maps.values()]
    sm_flat = np.array(sm_flat)

    sm_autocorr = np.corrcoef(sm_flat)
    np.nan_to_num(sm_autocorr, copy=False, nan = 0, posinf = 0, neginf = 0)

    sm_upper_tri = np.triu(copy.deepcopy(sm_autocorr), k=1)
    high_sm_idx = np.where(sm_upper_tri > sm_threshold)
    high_sm_pairs = set([(ac.cell_ids[high_sm_idx[0][i]], ac.cell_ids[high_sm_idx[1][i]]) for i in range(len(high_sm_idx[0]))])
    return sm_autocorr, high_sm_pairs


def get_ei_autocorrelation(block: Union[AnalysisChunk,MEAResponseBlock], ei_method, ei_threshold: float):
    '''
    Computes the autocorrelation of EI maps in a ResponseBlock or AnalysisChunk.
    Args:
        - block: ResponseBlock or AnalysisChunk object.
        - ei_method: str, method for computing EI maps. Options are 'full', 'space', or 'power'.
        - threshold: float, the autocorrelation threshold above which pairs of EI maps are considered highly correlated.
    Returns:
        - ei_autocorr: 2D numpy array of autocorrelation values for EI maps.
        - high_ei_pairs: set of tuples containing cell IDs of EI maps with autocorrelation above the specified threshold.
    '''

    ei_corr = vu.ei_corr(block, block, method=ei_method)

    ei_upper_tri = np.triu(copy.deepcopy(ei_corr), k=1)
    high_ei_idx = np.where(ei_upper_tri > ei_threshold)
    high_ei_pairs = set([(block.cell_ids[high_ei_idx[0][i]], block.cell_ids[high_ei_idx[1][i]]) for i in range(len(high_ei_idx[0]))])

    return ei_corr, high_ei_pairs

class DedupBlock:
    def __init__(self, exp_name: str=None, datafile_name: str=None, chunk_name: str=None,
                 ss_version: str = 'kilosort2.5', pkl_file: str=None, is_noise: bool=False,
                 ei_method: str='full',
                 ei_threshold: float = 0.80, sm_threshold: float = 0.80,
                 **vu_kwargs):
        self.exp_name = exp_name
        self.datafile_name = datafile_name
        self.chunk_name = chunk_name
        self.ss_version = ss_version
        self.pkl_file = pkl_file
        self.is_noise = is_noise
        self.ei_threshold = ei_threshold
        self.sm_threshold = sm_threshold
        self.ei_method = ei_method
        if is_noise:
            self.AnalysisChunk = AnalysisChunk(exp_name=exp_name, chunk_name=chunk_name,
                                                ss_version=ss_version, pkl_file=pkl_file, b_load_spatial_maps=True, **vu_kwargs)
            self.cluster_to_index = dict(zip(self.AnalysisChunk.cell_ids, range(len(self.AnalysisChunk.cell_ids))))
        else:
            self.MEAResponseBlock = MEAResponseBlock(exp_name=exp_name, datafile_name=datafile_name, ss_version=ss_version, pkl_file=pkl_file)
            self.cluster_to_index = dict(zip(self.MEAResponseBlock.cell_ids, range(len(self.MEAResponseBlock.cell_ids))))

        #We want to spit out various sets that describe possible duplicate clusters by different metrics
        #organize in a dict
        #TODO: which of these do we actually need? prob can get some of this

        self.dedup_sets = dict()
        if is_noise:
            #if noise, get the highly correlated spatial maps, the transitively connected groups, and the decomposed list of clusters
            self.sm_autocorr, high_sm_pairs = get_sm_autocorrelation(self.AnalysisChunk, sm_threshold=self.sm_threshold)
            self.ei_autocorr, high_ei_pairs = get_ei_autocorrelation(self.AnalysisChunk, self.ei_method, ei_threshold=self.ei_threshold)
            extended_ei_pairs = generate_extended_pairings(high_ei_pairs)
            extended_sm_pairs = generate_extended_pairings(high_sm_pairs)
            problem_cells, all_ei_cells, all_sm_cells = self.isolate_problem_cells(high_ei_pairs, high_sm_pairs)
        else:
            self.ei_autocorr, high_ei_pairs = get_ei_autocorrelation(self.MEAResponseBlock,self.ei_method, ei_threshold=self.ei_threshold)
            extended_ei_pairs = generate_extended_pairings(high_ei_pairs)
            problem_cells, all_ei_cells = self.isolate_problem_cells(high_ei_pairs)

        print('Creating deduplication sets...')
        self.dedup_sets['all_pairs'] = high_ei_pairs.union(high_sm_pairs) if is_noise else high_ei_pairs #combined pairings
        self.dedup_sets['extended_pairs_across'] = generate_extended_pairings(self.dedup_sets['all_pairs']) #combined transitively connected groups across metrics
        self.dedup_sets['all_extended_pairs'] = extended_ei_pairs.union(extended_sm_pairs) if is_noise else extended_ei_pairs #transitively connected groups within metrics
        
        self.dedup_sets['problem_cells'] = problem_cells #all cells in any pair

        self.dedup_sets['ei_only_cells'] = problem_cells.difference(all_sm_cells) #cells only in high ei pairs
        self.dedup_sets['ei_only_pairs'] = self.dedup_sets['all_pairs'].difference(high_sm_pairs) if is_noise else high_ei_pairs #pairs only in high ei pairs
        self.dedup_sets['extended_ei_pairs'] = extended_ei_pairs #transitively connected groups within high ei pairs
        if is_noise:
            self.dedup_sets['sm_only_cells'] = problem_cells.difference(all_ei_cells) #cells only in high sm pairs
            self.dedup_sets['sm_only_pairs'] = self.dedup_sets['all_pairs'].difference(high_ei_pairs) #pairs only in high sm pairs
            self.dedup_sets['extended_sm_pairs'] = extended_sm_pairs

            #now only the intersecting sets
            self.dedup_sets['both_cells'] = all_ei_cells.intersection(all_sm_cells) #cells in both high ei and high sm pairs
            self.dedup_sets['both_pairs'] = high_ei_pairs.intersection(high_sm_pairs) #pairs in both high ei and high sm pairs
            self.dedup_sets['extended_both_pairs'] = generate_extended_pairings(self.dedup_sets['both_pairs']) #transitively connected groups within both high ei and high sm pairs
        
        print('Loading Kilosort amplitudes...')
        self.load_ks_amplitudes()


    def load_ks_amplitudes(self):
        if self.is_noise:
            amps = np.load(os.path.join(DATA_DIR, self.AnalysisChunk.exp_name, self.AnalysisChunk.chunk_name, self.AnalysisChunk.ss_version, 'amplitudes.npy'))
            temps = np.load(os.path.join(DATA_DIR, self.AnalysisChunk.exp_name, self.AnalysisChunk.chunk_name, self.AnalysisChunk.ss_version, 'spike_templates.npy'))
            times = np.load(os.path.join(DATA_DIR, self.AnalysisChunk.exp_name, self.AnalysisChunk.chunk_name, self.AnalysisChunk.ss_version, 'spike_times.npy'))
        else:
            amps = np.load(os.path.join(DATA_DIR, self.MEAResponseBlock.exp_name, self.MEAResponseBlock.datafile_name, self.MEAResponseBlock.ss_version, 'amplitudes.npy'))
            temps = np.load(os.path.join(DATA_DIR, self.MEAResponseBlock.exp_name, self.MEAResponseBlock.datafile_name, self.MEAResponseBlock.ss_version, 'spike_templates.npy'))
            times = np.load(os.path.join(DATA_DIR, self.MEAResponseBlock.exp_name, self.MEAResponseBlock.datafile_name, self.MEAResponseBlock.ss_version, 'spike_times.npy'))
        
        vision_temps = temps+1

        #combine into 3D array
        amplitudes = np.zeros((3, len(amps)))
        amplitudes[0,:] = np.squeeze(amps)
        amplitudes[1,:] = np.squeeze(vision_temps)
        amplitudes[2,:] = np.squeeze(times) * 1000 / 20000
        self.amplitudes = amplitudes


    def isolate_problem_cells(self, high_ei_pairs:set, high_sm_pairs:set=None):
        '''
        Isolates problem cells based on spatial map and EI autocorrelation.
        Returns:
            - problem_cells: set of cell IDs that are problematic based on spatial map and EI correlations.
            - all_ei_cells: set of all cell IDs with high EI autocorrelation.
            - all_sm_cells: set of all cell IDs with high spatial map autocorrelation (if available).
        '''
        
        # Compute autocorrelation for spatial maps

        if self.is_noise:
            problem_cells = {cell for pair in high_sm_pairs.union(high_ei_pairs) for cell in pair}
            all_sm_cells = {cell for pair in high_sm_pairs for cell in pair}
            all_ei_cells = {cell for pair in high_ei_pairs for cell in pair}

            return problem_cells, all_ei_cells, all_sm_cells

        else:
            problem_cells = {cell for pair in high_ei_pairs for cell in pair}
            all_ei_cells = {cell for pair in high_ei_pairs for cell in pair}

            return problem_cells, all_ei_cells


    
    def plotRFs_dedup(self, axs=None):
        '''
        Plots RFs of cells in an AnalysisChunk to visualize potential duplicates.
        Args:
            - ac: AnalysisChunk object.
        Returns:
            - fig, ax: matplotlib figure and axis objects.
        '''

        # assert hasattr(ac, 'chunk_name'), "Must use AnalysisChunk for plotting RFs."
        if not self.is_noise:
            raise ValueError("RF plotting can only be done for AnalysisChunk objects.")

        rf_params = self.AnalysisChunk.rf_params
        cell_ids = self.AnalysisChunk.cell_ids
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for cell in cell_ids:
            ell1 = Ellipse(xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                    width=rf_params[cell]['std_x']*2, height=rf_params[cell]['std_y']*2,
                    angle=rf_params[cell]['rot'], 
                    edgecolor='None', facecolor='None', lw=1, alpha=0.8)
            if cell in self.dedup_sets['problem_cells'].difference(self.dedup_sets['ei_only_cells']) and cell in self.dedup_sets['problem_cells'].difference(self.dedup_sets['sm_only_cells']):
                ell1.set_edgecolor('red')
                color = 'red'
                axs[1].add_patch(ell1)
                axs[1].annotate(f'{cell}', xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                            color='black', fontsize=10, ha='center', va='center')
            elif cell in self.dedup_sets['ei_only_cells']:
                ell1.set_edgecolor('orange')
                color = 'orange'
                axs[1].add_patch(ell1)
                axs[1].annotate(f'{cell}', xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                            color='black', fontsize=10, ha='center', va='center')
            elif cell in self.dedup_sets['sm_only_cells']:
                ell1.set_edgecolor('magenta')
                color = 'magenta'
                axs[1].add_patch(ell1)
                axs[1].annotate(f'{cell}', xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                            color='black', fontsize=10, ha='center', va='center')
            else:
                color = 'black'

            ell = Ellipse(xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                    width=rf_params[cell]['std_x']*2, height=rf_params[cell]['std_y']*2,
                    angle=rf_params[cell]['rot'], 
                    edgecolor=color, facecolor='None', lw=1, alpha=0.5)
            axs[0].add_patch(ell)
        axs[0].set_xlim(0, self.AnalysisChunk.numXChecks)
        axs[0].set_ylim(self.AnalysisChunk.numYChecks, 0)
        axs[1].set_xlim(0, self.AnalysisChunk.numXChecks)
        axs[1].set_ylim(self.AnalysisChunk.numYChecks, 0)
        axs[0].set_title('All cells')
        axs[1].set_title('Cells with high correlations')

        custom_lines = [Line2D([0], [0], color='red', lw=1, label='Both'),
                        Line2D([0], [0], color='orange', lw=1, label='EI only'),
                        Line2D([0], [0], color='magenta', lw=1, label='RF only')]

        axs[1].legend(handles=custom_lines, loc='upper right')

        return axs

    def plot_correlations(self, axs=None):
        '''
        Plots histograms of spatial map and EI autocorrelation values for an AnalysisChunk or ResponseBlock.
        Returns:
            - fig, ax: matplotlib figure and axis objects.
        '''

        if self.is_noise:
            if axs is None:
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            axs[0,0].imshow(self.ei_autocorr, cmap='viridis', interpolation='nearest')
            axs[0,0].set_title('EI Autocorrelation Matrix')
            axs[0,1].imshow(self.sm_autocorr, cmap='viridis', interpolation='nearest')
            axs[0,1].set_title('Spatial Map Autocorrelation Matrix')
            axs[0,0].set_xlabel('Cell ID')
            axs[0,1].set_xlabel('Cell ID')
            axs[0,0].set_ylabel('Cell ID')
            axs[0,1].set_ylabel('Cell ID')

            axs[1,0].hist(np.triu(self.ei_autocorr, k=1).flatten(), bins=50, color='blue', alpha=0.7)
            axs[1,0].set_title('EI Autocorrelation Histogram')
            axs[1,0].set_xlabel('R')
            axs[1,0].set_ylabel('Count')
            axs[1,0].semilogy()
            axs[1,0].axvline(self.ei_threshold, color='red', linestyle='--', label=f'Threshold: {self.ei_threshold}')
            axs[1,0].legend()
            axs[1,1].hist(np.triu(self.sm_autocorr, k=1).flatten(), bins=50, color='green', alpha=0.7)
            axs[1,1].set_title('Spatial Map Autocorrelation Histogram')
            axs[1,1].set_xlabel('R')
            axs[1,1].set_ylabel('Count')
            axs[1,1].semilogy()
            axs[1,1].axvline(self.sm_threshold, color='red', linestyle='--', label=f'Threshold: {self.sm_threshold}')
            axs[1,1].legend()
        else:
            if axs is None:
                fig, axs = plt.subplots(2,1,figsize=(6, 5))
            axs[0].imshow(self.ei_autocorr, cmap='hot', interpolation='nearest')
            axs[0].set_title('EI Autocorrelation Matrix')
            axs[0].set_xlabel('Cell ID')
            axs[0].set_ylabel('Cell ID')
            
            axs[1].hist(np.triu_indices_from(self.ei_autocorr, k=1).flatten(), bins=50, color='blue', alpha=0.7)
            axs[1].set_title('EI Autocorrelation Histogram')
            axs[1].set_xlabel('R')
            axs[1].set_ylabel('Count')
            axs[1].semilogy()
            axs[1].axvline(self.ei_threshold, color='red', linestyle='--', label=f'Threshold: {self.ei_threshold}')
            axs[1].legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        return axs

    def visualize_groups(self, group: tuple, axs=None, detailed=False):
        '''
        Generates plots for groups of potentially duplicated clusters.
        Args:
            - group: tuple of duplicates
            - block: AnalysisChunk or ResponseBlock object.
        Returns:
            - fig, axs: matplotlib figure and axis objects.
        '''
        if self.is_noise:
            block = self.AnalysisChunk
        else:
            block = self.MEAResponseBlock
        
        n_clusters = len(group)
        if self.is_noise:
            if axs is None:
                if detailed:
                    fig,axs  = plt.subplots(8, n_clusters, figsize=(5*n_clusters, 25))
                else:
                    fig, axs = plt.subplots(3, n_clusters, figsize=(5*n_clusters, 15))
        else:
            if axs is None:
                if detailed:
                    fig, axs = plt.subplots(6, n_clusters, figsize=(5*n_clusters, 25))
                else:
                    fig, axs = plt.subplots(2, n_clusters, figsize=(5*n_clusters, 15))

        for i, cell in enumerate(group):
            if self.is_noise:
                sm = block.d_spatial_maps[cell][:,:,0]
                im1 = axs[0, i].imshow(sm, cmap='gray')
                axs[0, i].set_title(f'ID {cell}')
                axs[0, i].axis('off')
                timecourse_g = block.vcd.main_datatable[cell]['GreenTimeCourse']
                timecourse_b = block.vcd.main_datatable[cell]['BlueTimeCourse']
                axs[1, i].plot(timecourse_g, color='green')
                axs[1, i].plot(timecourse_b, color='blue')
                axs[1, i].set_title(f'ID {cell}')
                axs[1, i].set_xlabel('Time (ms)')
                axs[1, i].set_ylabel('STA (a.u.)')

                if detailed:
                    ax1 = eu.plot_ei_map(cell, block.vcd, axs=axs[2:, i])
                    axs[2, i].set_title(f'ID {cell} - EI Map')
                else:
                    ei = block.vcd.get_ei_for_cell(cell).ei
                    sorted_electrodes = eu.sort_electrode_map(block.vcd.get_electrode_map())
                    ei = eu.reshape_ei(ei, sorted_electrodes)
                    ei = np.log10(np.max(np.abs(ei), axis=2) + 1e-6)  # Log scale for visualization
                    im3 = axs[2, i].imshow(ei, cmap='hot')
                    axs[2, i].set_title(f'ID {cell}')
            else:
                if detailed:
                    ax1 = eu.plot_ei_map(cell, block.vcd, axs=axs[0:, i])
                    axs[0, i].set_title(f'ID {cell} - EI Map')
                else:
                    ei = block.vcd.get_ei_for_cell(cell).ei
                    sorted_electrodes = eu.sort_electrode_map(block.vcd.get_electrode_map())
                    ei = eu.reshape_ei(ei, sorted_electrodes)
                    ei = np.log10(np.max(np.abs(ei), axis=2) + 1e-6)
                    im1 = axs[0, i].imshow(ei, cmap='hot')
                    axs[0, i].set_title(f'ID {cell}')

        #pull each cell in the group's correlation to each other cell using np.ix_:
        ei_corrs = self.ei_autocorr[np.ix_([self.cluster_to_index[c] for c in group], [self.cluster_to_index[c] for c in group])]
        if self.is_noise:
            sm_corrs = self.sm_autocorr[np.ix_([self.cluster_to_index[c] for c in group], [self.cluster_to_index[c] for c in group])]
        
        fig.suptitle(f'Average EI correlation: {np.mean(ei_corrs):.2f}, Average SM correlation: {np.mean(sm_corrs):.2f}' if self.is_noise else f'Average EI correlation: {np.mean(ei_corrs):.2f}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        return axs

    def plot_amplitude_histograms(self, group: tuple, axs=None):
        '''
        Plots histograms of amplitudes for a group of cells.
        Args:
            - amps: kilsort output of amplitudes and corresponding templates (optional, will load if set to none)
            - group: group of associated cells
        Returns:
            - axs: matplotlib axis objects with histograms
        '''

        if axs is None:
            fig, axs = plt.subplots(1,1, figsize=(10, 5))
        ax_histy = axs.inset_axes([1.05, 0, 0.25, 1], sharey=axs)
        ax_histy.tick_params(axis='y', labelleft=False)

        hists = []
        amps = []

        for i, cell in enumerate(group):
            amps.append(self.amplitudes[0, self.amplitudes[1,:]==cell])
            times = self.amplitudes[2, self.amplitudes[1,:]==cell] * 1000 / 20000  # convert to seconds assuming 20kHz sampling rate
            axs.plot(times,amps[i], label=f'ID {cell}', alpha=0.5)
        
        min_val = min([np.min(a) for a in amps])
        max_val = max([np.max(a) for a in amps])
        num_bins = 50
        bin_edges = np.linspace(min_val, max_val, num_bins+1)

        for i, cell in enumerate(group):
            hists.append(np.histogram(amps[i], bins=bin_edges)[0])
            ax_histy.stairs(hists[i], bin_edges, fill=True, alpha=0.5, orientation='horizontal', label=f'ID {cell}')

        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Amplitude')
        ax_histy.set_xlabel('Count')
        axs.legend(loc='upper left')
        ax_histy.legend(loc='upper right')
        fig.suptitle('Amplitude Histograms', fontsize=16)

        plt.tight_layout()
        return axs

    def get_amplitude_overlap(self, pair:tuple):
        '''
        generates amplitude histograms for all cells in a set of pairs.
        Args:
            - amps: kilsort output of amplitudes and corresponding templates (optional, will load if set to none)
            
        Returns:
            - overlap fraction (float) 
        '''

        amp1 = self.amplitudes[0, self.amplitudes[1,:]==pair[0]]
        amp2 = self.amplitudes[0, self.amplitudes[1,:]==pair[1]]

        min_val = min(np.min(amp1), np.min(amp2))
        max_val = max(np.max(amp1), np.max(amp2))
        num_bins = 50
        bin_edges = np.linspace(min_val, max_val, num_bins+1)
        hist1, _ = np.histogram(amp1, bins=bin_edges)
        hist2, _ = np.histogram(amp2, bins=bin_edges)
        intersect_sum = np.sum(np.minimum(hist1, hist2))
        total_sum = np.sum(hist1)
        overlap_fraction = intersect_sum / total_sum if total_sum > 0 else 0
        return overlap_fraction
    
    def plot_pcs(self, pair:tuple, n_pcs=2, n_chans=2):
        if not hasattr(self, 'spike_pcs'):
            if self.is_noise:
                self.spike_pcs = np.load(os.path.join(DATA_DIR, self.AnalysisChunk.exp_name, self.AnalysisChunk.chunk_name, self.AnalysisChunk.ss_version, 'pc_features.npy'), mmap_mode='r')
                self.spike_pc_idx = np.load(os.path.join(DATA_DIR, self.AnalysisChunk.exp_name, self.AnalysisChunk.chunk_name, self.AnalysisChunk.ss_version, 'pc_feature_ind.npy'), mmap_mode='r')
            else: 
                self.spike_pcs = np.load(os.path.join(DATA_DIR, self.MEAResponseBlock.exp_name, self.MEAResponseBlock.datafile_name, self.MEAResponseBlock.ss_version, 'pc_features.npy'), mmap_mode='r')
                self.spike_pc_idx = np.load(os.path.join(DATA_DIR, self.MEAResponseBlock.exp_name, self.MEAResponseBlock.datafile_name, self.MEAResponseBlock.ss_version, 'pc_feature_ind.npy'), mmap_mode='r')
        spike_idxs = []
        feat_chan_ids = []
        channels = []
        spike_times = []
        n_combs = n_chans * n_pcs
        for cluster in pair:
            spike_idxs.append(np.where(self.amplitudes[1,:]==cluster)[0])
            feat_chan_ids.append(self.spike_pc_idx[cluster-1, :].astype(int))
            cluster_chans = []
            for c in range(n_chans):
                cluster_chans.append(feat_chan_ids[-1][c])
            channels.append(cluster_chans)
            spike_times.append(self.amplitudes[2, spike_idxs[-1]] * 1000 / 20000)
        fig, axs = plt.subplots(n_combs, n_combs, figsize=(5*n_chans, 5*n_pcs))
        for i in range(n_combs):
            for j in range(n_combs):
                if i == j:
                    for idx,cluster2 in enumerate(pair):
                        x = spike_times[idx]
                        y = self.spike_pcs[spike_idxs[idx], i%2, i//2]
                        axs[i, j].scatter(x, y, alpha=0.5, label=f'Cluster {cluster2}')
                        axs[i, j].set_xlabel('Time (s)')
                        axs[i, j].set_ylabel(f'Chan{i//2+1} PC{i%2+1}')

                else:
                    for idx,cluster2 in enumerate(pair):
                        axs[i, j].set_xlabel('Time (s)')
                        axs[i, j].set_ylabel(f'Chan{i//2+1} PC{i%2+1}')
                        x = self.spike_pcs[spike_idxs[idx], i%2, i//2]
                        y = self.spike_pcs[spike_idxs[idx], j%2, j//2]
                        axs[i, j].scatter(x, y, alpha=0.5, label=f'Cluster {cluster2}')
                        axs[i,j].axvline(0, color='gray', linewidth=0.5)
                        axs[i,j].axhline(0, color='gray', linewidth=0.5)

                axs[i,j].legend(loc='upper right')
                axs[i,j].margins(0.2,0.2)
        fig.suptitle(f'PCs for clusters {pair}', fontsize=16)
        return axs

                


    def get_summary_stats(self, pairs:set):
        '''
        Generates summary statistics for a set of potentially duplicated clusters.
        Args:
            - pairs: set of tuples containing cell IDs.
        Returns:
            - summary_stats: dataframe containing summary statistics for the pairs.
        '''

        stats = []
        header = ['cluster_a', 'cluster_b', 'ei_corr', 'sm_corr', 'overlap_fraction']

        for a, b in pairs:
            if self.is_noise:
                sm_corr = self.sm_autocorr[self.cluster_to_index[a], self.cluster_to_index[b]]
            ei_corr = self.ei_autocorr[self.cluster_to_index[a], self.cluster_to_index[b]]
            overlap_fraction = self.get_amplitude_overlap((a, b))
            stats.append([a, b, ei_corr, sm_corr if self.is_noise else None, overlap_fraction])
        
        summary_stats = pd.DataFrame(stats, columns=header)
        return summary_stats

    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  chunk_name: {self.chunk_name}\n"
        str_self += f"  ss_version: {self.ss_version}\n"
        str_self += f"  noise_status: {self.is_noise}\n"
        str_self += f"  data_file: {self.datafile_name}\n"
        if self.is_noise:
            str_self += f"  cell_ids of length: {len(self.AnalysisChunk.cell_ids)}\n"
        else:
            str_self += f"  cell_ids of length: {len(self.MEAResponseBlock.cell_ids)}\n"
        str_self += f"  ei_threshold: {self.ei_threshold}\n"
        str_self += f"  sm_threshold: {self.sm_threshold}\n"
        str_self += f"  ei_method: {self.ei_method}\n"
        return str_self


    
