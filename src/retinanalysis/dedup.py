import numpy as np
from retinanalysis.response import ResponseBlock
from retinanalysis.stim import StimBlock
from retinanalysis.analysis_chunk import AnalysisChunk
import visionloader as vl
import retinanalysis.vision_utils as vu
import os
from retinanalysis.settings import NAS_ANALYSIS_DIR
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy

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
    assert hasattr(ac, 'chunk_name'), "Must use AnalysisChunk for spatial maps."

    sm_flat = [sm.flatten() for sm in ac.d_spatial_maps.values()]
    sm_flat = np.array(sm_flat)

    sm_autocorr = np.corrcoef(sm_flat)
    np.nan_to_num(sm_autocorr, copy=False, nan = 0, posinf = 0, neginf = 0)

    sm_upper_tri = np.triu(copy.deepcopy(sm_autocorr), k=1)
    high_sm_idx = np.where(sm_upper_tri > threshold)
    high_sm_pairs = set([(ac.cell_ids[high_sm_idx[0][i]], ac.cell_ids[high_sm_idx[1][i]]) for i in range(len(high_sm_idx[0]))])

    return sm_autocorr, high_sm_pairs

def get_ei_autocorrelation(block: object, ei_method: str = 'full', threshold: float = 0.80):
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

    # assert isinstance(block, AnalysisChunk) or isinstance(block, ResponseBlock), "Input must be a ResponseBlock or AnalysisChunk."

    ei_corr = vu.ei_corr(block.vcd, block.vcd, method=ei_method)

    ei_upper_tri = np.triu(copy.deepcopy(ei_corr), k=1)
    high_ei_idx = np.where(ei_upper_tri > threshold)
    high_ei_pairs = set([(block.cell_ids[high_ei_idx[0][i]], block.cell_ids[high_ei_idx[1][i]]) for i in range(len(high_ei_idx[0]))])

    return ei_corr, high_ei_pairs

def isolate_problem_cells(block: object, ei_method: str = 'full', sm_threshold: float = 0.80, ei_threshold: float = 0.80):
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
    if hasattr(block, 'chunk_name'):
        sm_autocorr, high_sm_pairs = get_sm_autocorrelation(block, threshold=sm_threshold)
    elif hasattr(block, 'protocol_name'):
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
    
def plotRFs_dedup(ac: AnalysisChunk, ei_method: str = 'full', sm_threshold: float = 0.80, ei_threshold: float = 0.80):
    '''
    Plots RFs of cells in an AnalysisChunk to visualize potential duplicates.
    Args:
        - ac: AnalysisChunk object.
    Returns:
        - fig, ax: matplotlib figure and axis objects.
    '''

    assert hasattr(ac, 'chunk_name'), "Must use AnalysisChunk for plotting RFs."

    problem_cells, all_ei_cells, all_sm_cells = isolate_problem_cells(ac, ei_method=ei_method, sm_threshold=sm_threshold, ei_threshold=ei_threshold)

    rf_params = ac.rf_params
    cell_ids = ac.cell_ids
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for cell in cell_ids:
        ell1 = Ellipse(xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                width=rf_params[cell]['std_x']*2, height=rf_params[cell]['std_y']*2,
                angle=rf_params[cell]['rot'], 
                edgecolor='None', facecolor='None', lw=1, alpha=0.8)
        if cell in all_ei_cells and cell in all_sm_cells:
            ell1.set_edgecolor('red')
            color = 'red'
            ax[1].add_patch(ell1)
            ax[1].annotate(f'{cell}', xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                        color='black', fontsize=10, ha='center', va='center')
        elif cell in all_ei_cells:
            ell1.set_edgecolor('orange')
            color = 'orange'
            ax[1].add_patch(ell1)
            ax[1].annotate(f'{cell}', xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                        color='black', fontsize=10, ha='center', va='center')
        elif cell in all_sm_cells:
            ell1.set_edgecolor('magenta')
            color = 'magenta'
            ax[1].add_patch(ell1)
            ax[1].annotate(f'{cell}', xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                        color='black', fontsize=10, ha='center', va='center')
        else:
            color = 'black'

        ell = Ellipse(xy=(rf_params[cell]['center_x'], rf_params[cell]['center_y']),
                width=rf_params[cell]['std_x']*2, height=rf_params[cell]['std_y']*2,
                angle=rf_params[cell]['rot'], 
                edgecolor=color, facecolor='None', lw=1, alpha=0.5)
        ax[0].add_patch(ell)
    ax[0].set_xlim(0, ac.numXChecks)
    ax[0].set_ylim(ac.numYChecks, 0)
    ax[1].set_xlim(0, ac.numXChecks)
    ax[1].set_ylim(ac.numYChecks, 0)
    ax[0].set_title('All cells')
    ax[1].set_title('Cells with high correlations')

    custom_lines = [Line2D([0], [0], color='red', lw=1, label='Both'),
                    Line2D([0], [0], color='orange', lw=1, label='EI only'),
                    Line2D([0], [0], color='magenta', lw=1, label='RF only')]

    ax[1].legend(handles=custom_lines, loc='upper right')

    return fig, ax

def plot_histograms(block: object, ei_method: str = 'full', sm_threshold: float = 0.80, ei_threshold: float = 0.80):
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
    if hasattr(block, 'chunk_name'):
        sm_autocorr, high_sm_pairs = get_sm_autocorrelation(block, threshold=sm_threshold)
        sm_corr_values = sm_autocorr[np.triu_indices_from(sm_autocorr, k=1)]
    elif hasattr(block, 'protocol_name'):
        sm_corr_values = None
    else:
        raise ValueError("Input block must be either an AnalysisChunk or a ResponseBlock.")

    # Compute autocorrelation for EI maps
    ei_autocorr, high_ei_pairs = get_ei_autocorrelation(block, ei_method=ei_method, threshold=ei_threshold)
    ei_corr_values = ei_autocorr[np.triu_indices_from(ei_autocorr, k=1)]

    if sm_corr_values is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].hist(ei_corr_values, bins=50, color='blue', alpha=0.7)
        ax[0].set_title('EI Autocorrelation Histogram')
        ax[0].set_xlabel('R')
        ax[0].set_ylabel('Count')
        ax[0].semilogy()
        ax[0].axvline(ei_threshold, color='red', linestyle='--', label=f'Threshold: {ei_threshold}')
        ax[0].legend()
        ax[1].hist(sm_corr_values, bins=50, color='green', alpha=0.7)
        ax[1].set_title('Spatial Map Autocorrelation Histogram')
        ax[1].set_xlabel('R')
        ax[1].set_ylabel('Count')
        ax[1].semilogy()
        ax[1].axvline(sm_threshold, color='red', linestyle='--', label=f'Threshold: {sm_threshold}')
        ax[1].legend()
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(ei_corr_values, bins=50, color='blue', alpha=0.7)
        ax.set_title('EI Autocorrelation Histogram')
        ax.set_xlabel('R')
        ax.set_ylabel('Count')
        ax.semilogy()
        ax.axvline(ei_threshold, color='red', linestyle='--', label=f'Threshold: {ei_threshold}')
        ax.legend()

    return fig, ax




    
