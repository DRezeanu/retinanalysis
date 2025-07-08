import numpy as np
from retinanalysis.response import ResponseBlock
from retinanalysis.stim import StimBlock
from retinanalysis.analysis_chunk import AnalysisChunk
import visionloader as vl
import retinanalysis.vision_utils as vu
import retinanalysis.ei_utils as eu
import os
from retinanalysis.settings import NAS_ANALYSIS_DIR
from retinanalysis.settings import NAS_DATA_DIR
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
from typing import Union
from itertools import combinations
import pandas as pd

def load_ks_amplitudes(block: Union[AnalysisChunk, ResponseBlock]):
    amps = np.load(os.path.join(NAS_DATA_DIR, block.exp_name, block.chunk_name, block.ss_version, 'amplitudes.npy'))
    temps = np.load(os.path.join(NAS_DATA_DIR, block.exp_name, block.chunk_name, block.ss_version, 'spike_templates.npy'))

    vision_temps = temps+1

    amplitudes = np.vstack((np.squeeze(amps), np.squeeze(vision_temps)))
    return amplitudes

def get_sm_autocorrelation(ac: AnalysisChunk, threshold: float = 0.80):
    '''
    Computes the autocorrelation of spatial maps in an AnalysisChunk.
    Args:
        - ac: AnalysisChunk object.
        - threshold: float, the autocorrelation threshold above which pairs of spatial maps are considered highly correlated.
    Returns:
        - sm_autocorr: 2D numpy array of autocorrelation values for spatial maps.
        - high_sm_pairs: set of tuples containing cell IDs of spatial maps with autocorrelation above the specified threshold.
    '''
    # assert hasattr(ac, 'chunk_name'), "Must use AnalysisChunk for spatial maps."
    assert isinstance(ac, AnalysisChunk), "Input must be an AnalysisChunk."

    sm_flat = [sm.flatten() for sm in ac.d_spatial_maps.values()]
    sm_flat = np.array(sm_flat)

    sm_autocorr = np.corrcoef(sm_flat)
    np.nan_to_num(sm_autocorr, copy=False, nan = 0, posinf = 0, neginf = 0)

    sm_upper_tri = np.triu(copy.deepcopy(sm_autocorr), k=1)
    high_sm_idx = np.where(sm_upper_tri > threshold)
    high_sm_pairs = set([(ac.cell_ids[high_sm_idx[0][i]], ac.cell_ids[high_sm_idx[1][i]]) for i in range(len(high_sm_idx[0]))])

    return sm_autocorr, high_sm_pairs

def get_ei_autocorrelation(block: Union[AnalysisChunk, ResponseBlock], ei_method: str = 'full', threshold: float = 0.80):
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

    assert isinstance(block, AnalysisChunk) or isinstance(block, ResponseBlock), "Input must be a ResponseBlock or AnalysisChunk."

    ei_corr = vu.ei_corr(block, block, method=ei_method)

    ei_upper_tri = np.triu(copy.deepcopy(ei_corr), k=1)
    high_ei_idx = np.where(ei_upper_tri > threshold)
    high_ei_pairs = set([(block.cell_ids[high_ei_idx[0][i]], block.cell_ids[high_ei_idx[1][i]]) for i in range(len(high_ei_idx[0]))])

    return ei_corr, high_ei_pairs

def isolate_problem_cells(block: Union[AnalysisChunk, ResponseBlock], ei_method: str = 'full', sm_threshold: float = 0.80, ei_threshold: float = 0.80):
    '''
    Isolates problem cells based on spatial map and EI autocorrelation.
    Args:
        - block: ResponseBlock or AnalysisChunk object.
        - ei_method: str, method for computing EI maps. Options are 'full', 'space', or 'power'.
        - sm_threshold: float, the autocorrelation threshold for spatial maps.
        - ei_threshold: float, the autocorrelation threshold for EI maps.
    Returns:
        - problem_cells: set of cell IDs that are problematic based on spatial map and EI correlations.
        - all_ei_cells: set of all cell IDs with high EI autocorrelation.
        - all_sm_cells: set of all cell IDs with high spatial map autocorrelation (if available).
    '''
    
    # Compute autocorrelation for spatial maps
    if isinstance(block, AnalysisChunk):
        sm_autocorr, high_sm_pairs = get_sm_autocorrelation(block, threshold=sm_threshold)
    elif isinstance(block, ResponseBlock):
        sm_autocorr, high_sm_pairs = None, None
    else:
        raise ValueError("Input block must be either an AnalysisChunk or a ResponseBlock.")

    # Compute autocorrelation for EI maps
    ei_autocorr, high_ei_pairs = get_ei_autocorrelation(block, ei_method=ei_method, threshold=ei_threshold)

    # Identify problematic cells based on spatial map if available and EI correlations
    if sm_autocorr is not None:
        problem_cells = {cell for pair in high_sm_pairs.union(high_ei_pairs) for cell in pair}
        all_sm_cells = {cell for pair in high_sm_pairs for cell in pair}
        all_ei_cells = {cell for pair in high_ei_pairs for cell in pair}

        return problem_cells, all_ei_cells, all_sm_cells
    else:
        problem_cells = {cell for pair in high_ei_pairs for cell in pair}
        all_ei_cells = {cell for pair in high_ei_pairs for cell in pair}

        return problem_cells, all_ei_cells
    
def plotRFs_dedup(ac: AnalysisChunk, ei_method: str = 'full', sm_threshold: float = 0.80, ei_threshold: float = 0.80, axs=None):
    '''
    Plots RFs of cells in an AnalysisChunk to visualize potential duplicates.
    Args:
        - ac: AnalysisChunk object.
    Returns:
        - fig, ax: matplotlib figure and axis objects.
    '''

    # assert hasattr(ac, 'chunk_name'), "Must use AnalysisChunk for plotting RFs."
    assert isinstance(ac, AnalysisChunk), "Input must be an AnalysisChunk."

    problem_cells, all_ei_cells, all_sm_cells = isolate_problem_cells(ac, ei_method=ei_method, sm_threshold=sm_threshold, ei_threshold=ei_threshold)

    rf_params = ac.rf_params
    cell_ids = ac.cell_ids
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for cell in cell_ids:
        ell1 = Ellipse(xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                width=rf_params[cell]['std_x']*2, height=rf_params[cell]['std_y']*2,
                angle=rf_params[cell]['rot'], 
                edgecolor='None', facecolor='None', lw=1, alpha=0.8)
        if cell in all_ei_cells and cell in all_sm_cells:
            ell1.set_edgecolor('red')
            color = 'red'
            axs[1].add_patch(ell1)
            axs[1].annotate(f'{cell}', xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                        color='black', fontsize=10, ha='center', va='center')
        elif cell in all_ei_cells:
            ell1.set_edgecolor('orange')
            color = 'orange'
            axs[1].add_patch(ell1)
            axs[1].annotate(f'{cell}', xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                        color='black', fontsize=10, ha='center', va='center')
        elif cell in all_sm_cells:
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
    axs[0].set_xlim(0, ac.numXChecks)
    axs[0].set_ylim(ac.numYChecks, 0)
    axs[1].set_xlim(0, ac.numXChecks)
    axs[1].set_ylim(ac.numYChecks, 0)
    axs[0].set_title('All cells')
    axs[1].set_title('Cells with high correlations')

    custom_lines = [Line2D([0], [0], color='red', lw=1, label='Both'),
                    Line2D([0], [0], color='orange', lw=1, label='EI only'),
                    Line2D([0], [0], color='magenta', lw=1, label='RF only')]

    axs[1].legend(handles=custom_lines, loc='upper right')

    return axs

def plot_correlations(block: Union[AnalysisChunk, ResponseBlock], ei_method: str = 'full', sm_threshold: float = 0.80, ei_threshold: float = 0.80, axs=None):
    '''
    Plots histograms of spatial map and EI autocorrelation values for an AnalysisChunk or ResponseBlock.
    Args:
        - block: AnalysisChunk or ResponseBlock object.
        - ei_method: str, method for computing EI maps. Options are 'full', 'space', or 'power'.
        - sm_threshold: float, the autocorrelation threshold for spatial maps.
        - ei_threshold: float, the autocorrelation threshold for EI maps.
    Returns:
        - fig, ax: matplotlib figure and axis objects.
    '''

    # Compute autocorrelation for spatial maps
    if isinstance(block, AnalysisChunk):
        sm_autocorr, high_sm_pairs = get_sm_autocorrelation(block, threshold=sm_threshold)
        sm_corr_values = sm_autocorr[np.triu_indices_from(sm_autocorr, k=1)]
    elif isinstance(block, ResponseBlock):
        sm_corr_values = None
    else:
        raise ValueError("Input block must be either an AnalysisChunk or a ResponseBlock.")

    # Compute autocorrelation for EI maps
    ei_autocorr, high_ei_pairs = get_ei_autocorrelation(block, ei_method=ei_method, threshold=ei_threshold)
    ei_corr_values = ei_autocorr[np.triu_indices_from(ei_autocorr, k=1)]

    if sm_corr_values is not None:
        if axs is None:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0,0].imshow(ei_autocorr, cmap='viridis', interpolation='nearest')
        axs[0,0].set_title('EI Autocorrelation Matrix')
        axs[0,1].imshow(sm_autocorr, cmap='viridis', interpolation='nearest')
        axs[0,1].set_title('Spatial Map Autocorrelation Matrix')
        axs[0,0].set_xlabel('Cell ID')
        axs[0,1].set_xlabel('Cell ID')
        axs[0,0].set_ylabel('Cell ID')
        axs[0,1].set_ylabel('Cell ID')

        axs[1,0].hist(ei_corr_values, bins=50, color='blue', alpha=0.7)
        axs[1,0].set_title('EI Autocorrelation Histogram')
        axs[1,0].set_xlabel('R')
        axs[1,0].set_ylabel('Count')
        axs[1,0].semilogy()
        axs[1,0].axvline(ei_threshold, color='red', linestyle='--', label=f'Threshold: {ei_threshold}')
        axs[1,0].legend()
        axs[1,1].hist(sm_corr_values, bins=50, color='green', alpha=0.7)
        axs[1,1].set_title('Spatial Map Autocorrelation Histogram')
        axs[1,1].set_xlabel('R')
        axs[1,1].set_ylabel('Count')
        axs[1,1].semilogy()
        axs[1,1].axvline(sm_threshold, color='red', linestyle='--', label=f'Threshold: {sm_threshold}')
        axs[1,1].legend()
    else:
        if axs is None:
            fig, axs = plt.subplots(2,1,figsize=(6, 5))
        axs[0].imshow(ei_autocorr, cmap='hot', interpolation='nearest')
        axs[0].set_title('EI Autocorrelation Matrix')
        axs[0].set_xlabel('Cell ID')
        axs[0].set_ylabel('Cell ID')
        
        axs[1].hist(ei_corr_values, bins=50, color='blue', alpha=0.7)
        axs[1].set_title('EI Autocorrelation Histogram')
        axs[1].set_xlabel('R')
        axs[1].set_ylabel('Count')
        axs[1].semilogy()
        axs[1].axvline(ei_threshold, color='red', linestyle='--', label=f'Threshold: {ei_threshold}')
        axs[1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return axs

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

def visualize_groups(group: tuple, block: Union[AnalysisChunk, ResponseBlock], axs=None, detailed=False):
    '''
    Generates plots for groups of potentially duplicated clusters.
    Args:
        - group: tuple of duplicates
        - block: AnalysisChunk or ResponseBlock object.
    Returns:
        - fig, axs: matplotlib figure and axis objects.
    '''
    cluster_to_index = dict(zip(block.cell_ids, range(len(block.cell_ids))))

    if isinstance(block, AnalysisChunk):
        plot_sm = True
        sm_autocorr, high_sm_pairs = get_sm_autocorrelation(block, threshold=0.80)
    elif isinstance(block, ResponseBlock):
        plot_sm = False
        sm_autocorr, high_sm_pairs = None, None
    else:
        raise ValueError("Input block must be either an AnalysisChunk or a ResponseBlock.")
    
    ei_autocorr, high_ei_pairs = get_ei_autocorrelation(block, ei_method='full', threshold=0.80)
    
    n_clusters = len(group)
    if plot_sm:
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
        if plot_sm:
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
    ei_corrs = ei_autocorr[np.ix_([cluster_to_index[c] for c in group], [cluster_to_index[c] for c in group])]
    if plot_sm:
        sm_corrs = sm_autocorr[np.ix_([cluster_to_index[c] for c in group], [cluster_to_index[c] for c in group])]
    
    fig.suptitle(f'Average EI correlation: {np.mean(ei_corrs):.2f}, Average SM correlation: {np.mean(sm_corrs):.2f}' if plot_sm else f'Average EI correlation: {np.mean(ei_corrs):.2f}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return axs

def plot_amplitude_histograms(block: Union[AnalysisChunk, ResponseBlock], group: tuple, axs=None, amplitudes=None):
    '''
    Plots histograms of amplitudes for a group of cells.
    Args:
        - amps: kilsort output of amplitudes and corresponding templates (optional, will load if set to none)
        - group: group of associated cells
    Returns:
        - axs: matplotlib axis objects with histograms
    '''
    if amplitudes is None:
        amplitudes = load_ks_amplitudes(block)

    if axs is None:
        fig, axs = plt.subplots(1,1, figsize=(10, 5))
    ax_histy = axs.inset_axes([1.05, 0, 0.25, 1], sharey=axs)
    ax_histy.tick_params(axis='y', labelleft=False)

    hists = []
    amps = []

    for i, cell in enumerate(group):
        amps.append(amplitudes[0, amplitudes[1,:]==cell])
        axs.plot(amps[i], label=f'ID {cell}', alpha=0.5)
    
    min_val = min([np.min(a) for a in amps])
    max_val = max([np.max(a) for a in amps])
    num_bins = 50
    bin_edges = np.linspace(min_val, max_val, num_bins+1)

    for i, cell in enumerate(group):
        hists.append(np.histogram(amps[i], bins=bin_edges)[0])
        ax_histy.stairs(hists[i], bin_edges, fill=True, alpha=0.5, orientation='horizontal', label=f'ID {cell}')

    axs.set_xlabel('Spike index')
    axs.set_ylabel('Amplitude')
    ax_histy.set_xlabel('Count')
    axs.legend(loc='upper left')
    ax_histy.legend(loc='upper right')
    fig.suptitle('Amplitude Histograms', fontsize=16)

    plt.tight_layout()
    return axs

def get_amplitude_overlap(block: Union[ResponseBlock, AnalysisChunk], pair:tuple, amplitudes=None):
    '''
    generates amplitude histograms for all cells in a set of pairs.
    Args:
        - amps: kilsort output of amplitudes and corresponding templates (optional, will load if set to none)
        
    Returns:
        - overlap fraction (float) 
    '''

    if amplitudes is None:
        amplitudes = load_ks_amplitudes(block)

    amp1 = amplitudes[0, amplitudes[1,:]==pair[0]]
    amp2 = amplitudes[0, amplitudes[1,:]==pair[1]]

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

def get_summary_stats(block: Union[AnalysisChunk, ResponseBlock], ei_method: str = 'full', ei_threshold: float = 0.80, sm_threshold: float = 0.80):
    '''
    Generates summary statistics for a set of potentially duplicated clusters.
    Args:
        - pairs: set of tuples containing cell IDs.
        - block: AnalysisChunk or ResponseBlock object.
        - ei_method: str, method for computing EI maps. Options are 'full', 'space', or 'power'.
        - ei_threshold: float, the autocorrelation threshold for EI maps.
        - sm_threshold: float, the autocorrelation threshold for spatial maps.
    Returns:
        - summary_stats: dataframe containing summary statistics for the pairs.
    '''

    cluster_to_index = dict(zip(block.cell_ids, range(len(block.cell_ids))))
    if isinstance(block, AnalysisChunk):
        sm_autocorr, high_sm_pairs = get_sm_autocorrelation(block, threshold=sm_threshold)
    elif isinstance(block, ResponseBlock):
        sm_autocorr, high_sm_pairs = None, None
    else:
        raise ValueError("Input block must be either an AnalysisChunk or a ResponseBlock.")
    ei_autocorr, high_ei_pairs = get_ei_autocorrelation(block, ei_method=ei_method, threshold=ei_threshold)

    stats = []
    header = ['cluster_a', 'cluster_b', 'ei_corr', 'sm_corr', 'overlap_fraction']

    extended_pairs = generate_extended_pairings(high_ei_pairs.union(high_sm_pairs))
    for tup in extended_pairs:
        group = np.array(tup)
        pairs = list(combinations(group, 2))
        for a, b in pairs:
            cluster_a = a
            cluster_b = b
            if sm_autocorr is not None:
                sm_corr = sm_autocorr[cluster_to_index[cluster_a], cluster_to_index[cluster_b]]
            ei_corr = ei_autocorr[cluster_to_index[cluster_a], cluster_to_index[cluster_b]]
            overlap_fraction = get_amplitude_overlap(block, (cluster_a, cluster_b))
            stats.append([cluster_a, cluster_b, ei_corr, sm_corr if sm_autocorr is not None else None, overlap_fraction])
    
    summary_stats = pd.DataFrame(stats, columns=header)

    return summary_stats, extended_pairs

    
