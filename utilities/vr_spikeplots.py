import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from utilities import vr_spikeoutputs as so
import seaborn as sns
from utilities import vr_celltype_io as ctio

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def plot_crf(spikeout: so.SpikeOutputs, ax=None, ls_types=None):
    if ls_types is None:
        ls_types = spikeout.ls_RGC_labels
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 5))
    
    contrast_vals = spikeout.d_crf['unique_params']['contrast']

    for idx, str_type in enumerate(ls_types):
        type_ids = spikeout.types.d_main_IDs[str_type]
        type_idx = ctio.map_ids_to_idx(type_ids, spikeout.d_crf['cluster_id'])

        arr_f1_amp = spikeout.d_crf['4Hz_amp'][type_idx, :]
        
        mean = np.mean(arr_f1_amp, axis=0)
        std = np.std(arr_f1_amp, axis=0)
        ax.plot(contrast_vals, mean, marker='o', c=f'C{idx}', label=str_type)
        ax.fill_between(contrast_vals, mean-std, mean+std, color=f'C{idx}', alpha=0.2)
        
    ax.set_xlabel('Contrast')
    ax.set_xticks(contrast_vals)
    ax.set_ylabel('F1 peak-trough amplitude (Hz)')
    ax.legend()

    return ax

def plot_isi_qc(spikeout: so.SpikeOutputs, axs=None):
    if axs is None:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)
    bad_idx = spikeout.bad_isi_idx
    good_idx = spikeout.good_isi_idx

    for p_idx, c_idx in enumerate([bad_idx, good_idx]):
        ax = axs[p_idx]
        ax.hist(spikeout.pct_refractory[c_idx], color=f'C{p_idx}', alpha=0.5)
        ax.set_ylabel('Number of clusters')
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_xlabel('Percent of spikes within refractory period.')
        median = np.median(spikeout.pct_refractory[c_idx])
        ax.axvline(median, c='k', linestyle='--')
        ax.text(0.95, 0.95, f'Median = {median:.2f}%',
            transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
  
    
    axs[0].set_title(f'ISI violations n = {len(bad_idx)}')
    axs[1].set_title(f'ISI OK n = {len(good_idx)}')
    return axs

def get_rf_ells(ls_cells: list, d_sta: dict, NOISE_GRID_SIZE: float, sd_mult: float=1.3):
    ells = [Ellipse(xy=(d_sta[n_ID]['x0']*NOISE_GRID_SIZE, 
                        d_sta[n_ID]['y0']*NOISE_GRID_SIZE), 
        width=d_sta[n_ID]['SigmaX']*2*NOISE_GRID_SIZE*sd_mult, 
        height=d_sta[n_ID]['SigmaY']*2*NOISE_GRID_SIZE*sd_mult,
        angle=d_sta[n_ID]['Theta']) for n_ID in ls_cells]

    return ells


def plot_rfs(spikeout: so.SpikeOutputs, ls_cells, 
             ell_color=None, ax=None, sd_mult=0.8, 
             alpha=0.6, facecolor='k', SCALING=None, b_label=False,
             b_zoom=True, lw=1):
    if not ax:
        f, ax = plt.subplots(figsize=(7, 7))

    if SCALING is None:
        SCALING = spikeout.NOISE_GRID_SIZE
        n_pad = 5
        ax.set_xlim((0-n_pad)*SCALING, (spikeout.N_WIDTH+n_pad)*SCALING)
        ax.set_ylim((0-n_pad)*SCALING, (spikeout.N_HEIGHT+n_pad)*SCALING)
        ax.axhline(0, c='k', linewidth=1)
        ax.axvline(0, c='k', linewidth=1)
        ax.axhline(spikeout.N_HEIGHT*SCALING, c='k', linewidth=1)
        ax.axvline(spikeout.N_WIDTH*SCALING, c='k', linewidth=1)

    if not ell_color:
        ell_color = np.random.rand(3)

    ells = get_rf_ells(ls_cells, spikeout.d_sta, SCALING, sd_mult)

    # If facecolor is tuple, make it a list of tuples
    if isinstance(facecolor, tuple) or isinstance(facecolor, str):
        ls_facecolors = [facecolor] * len(ells)
    else:
        ls_facecolors = facecolor

    for idx_ell, ell in enumerate(ells):
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        ell.set_edgecolor(ell_color)
        ell.set_facecolor(ls_facecolors[idx_ell])
        ell.set_linewidth(lw)

    if b_label:
        for idx, ell in enumerate(ells):
            ax.text(ell.center[0], ell.center[1], str(ls_cells[idx]), 
                    horizontalalignment='center', verticalalignment='center', fontsize=8,
                    fontweight='bold')
            
    # If zoom, find min and max ell centers and set lims
    if b_zoom and len(ells) > 0:
        x0s = np.array([ell.center[0] for ell in ells])
        y0s = np.array([ell.center[1] for ell in ells])
        x0_min, x0_max = np.min(x0s), np.max(x0s)
        y0_min, y0_max = np.min(y0s), np.max(y0s)
        # Don't want to stretch, so pick min and max of x and y
        # x0_min, x0_max = np.min([x0_min, y0_min]), np.max([x0_max, y0_max])
        # y0_min, y0_max = x0_min, x0_max
        pad = 400
        ax.set_xlim(x0_min-pad, x0_max+pad)
        ax.set_ylim(y0_min-pad, y0_max+pad)
    return ax, ells

def plot_type_rfs(data: so.SpikeOutputs, ls_RGC_keys=None,#['OffP', 'OffM', 'OnP', 'OnM', 'SBC'],
                    ls_colors = sns.color_palette(), axs=None, ls_facecolors=None, alpha=0.6,
                    b_ticks_off=True, ls_RGC_labels=None, d_IDs=None,
                    b_zoom=False, sd_mult=0.8, scaling=None, n_scalebar=100):
    # If d_IDs is None, use data.types.d_main_IDs
    if d_IDs is None:
        d_IDs = data.types.d_main_IDs
    
    if ls_RGC_keys is None:
        ls_RGC_keys = list(d_IDs.keys())
    
    ncols = len(ls_RGC_keys)
    if axs is None:
        f, axs = plt.subplots(ncols=ncols, figsize=(ncols*8, 5), sharey=True, sharex=True)
        # f.patch.set_facecolor('none')
        # f.text(0.1, 0.98, data.str_experiment, fontsize=12)

    # if ncols=1, axs is not an iterable, make it an array
    if ncols==1:
        axs = np.array([axs])
    
    if ls_facecolors is None:
        ls_facecolors = ls_colors

    if ls_RGC_labels is None:
        ls_RGC_labels = ls_RGC_keys
    
    g_x0, g_x1 = np.inf, -np.inf
    g_y0, g_y1 = np.inf, -np.inf
    for i, str_key in enumerate(ls_RGC_keys):
        ls_cells = d_IDs[str_key]
        ls_label = ls_RGC_labels[i]

        # Remove any cells that don't have keys in data.d_sta
        ls_cells = [c for c in ls_cells if c in data.d_sta.keys()]
        
        ax = axs[i]
        _, ells = plot_rfs(data, ls_cells, ell_color=ls_colors[i], facecolor=ls_facecolors[i], 
                           ax=ax, alpha=alpha, b_zoom=b_zoom, sd_mult=sd_mult, SCALING=scaling)
        ax.set_title(ls_label+f' (n={len(ls_cells)}) RFs')

        # Get current axis limits
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        g_x0, g_x1 = min(g_x0, x0), max(g_x1, x1)
        g_y0, g_y1 = min(g_y0, y0), max(g_y1, y1)

    if b_zoom:
        for ax in axs:
            ax.set_xlim(g_x0, g_x1)
            ax.set_ylim(g_y0, g_y1)

    if b_ticks_off:
        for ax in axs:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Set axs and figure bg transparent
    for ax in axs:
        ax.set_facecolor('none')
    

    # # Add scalebar
    # offset = 0.08*(x1-x0)
    # for ax in axs:
    #     # ax.plot([100, 100+n_scalebar], [100, 100], c='k', linewidth=2)
    #     # ax.text(100+n_scalebar/2, 100-10, f'{n_scalebar} um', horizontalalignment='center')
    #     ax.plot([x0+offset, x0+offset+n_scalebar], [y0+offset, y0+offset], c='k', linewidth=2)
    #     ax.plot([x0+offset, x0+offset], [y0+offset, y0+offset+n_scalebar], c='k', linewidth=2)
    #     ax.text(x0+offset+n_scalebar/2, y0+offset*0.9, f'{n_scalebar} um', horizontalalignment='center', verticalalignment='top')

    return axs

def plot_type_tcs(data: so.SpikeOutputs, ls_RGC_keys=None,
                  axs=None, ls_colors = None, alpha=0.6, b_plot_mean=True, 
                  ls_RGC_labels=None, lw=1, d_IDs=None, n_dt = 1/120*1e3):
    # If d_IDs is None, use data.types.d_main_IDs
    if d_IDs is None:
        d_IDs = data.types.d_main_IDs
    if ls_RGC_keys is None:
        ls_RGC_keys = list(d_IDs.keys())

    # Plot time courses
    if axs is None:
        f, axs = plt.subplots(ncols=len(ls_RGC_keys), figsize=(len(ls_RGC_keys)*5, 5), sharey=True, sharex=True)
        # f.text(0.1, 0.98, data.str_experiment, fontsize=12)
        f.patch.set_facecolor('none')

    # if ncols=1, axs is not an array, make it an array
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    
    if ls_colors is None:
        ls_colors = ['k', 'gray']#['C1', 'C0']

    if ls_RGC_labels is None:
        ls_RGC_labels = ls_RGC_keys

    for i, str_key in enumerate(ls_RGC_keys):
        ls_cells = data.types.d_main_IDs[str_key]
        str_label = ls_RGC_labels[i]

        axs[i].axhline(0, c='k', linewidth=1)

        # Remove any cells that don't have keys in data.d_sta
        ls_cells = [c for c in ls_cells if c in data.d_sta.keys()]
        
        n_pts = len(data.d_sta[ls_cells[0]]['RedTimeCourse'])
        if n_dt is not None:
            x = np.arange(n_pts)*n_dt
            x = -x[::-1]
        else:
            x = np.arange(n_pts)
        
        if b_plot_mean:
            tcs = np.array([data.d_sta[n_ID]['RedTimeCourse'] for n_ID in ls_cells])
            mean = np.mean(tcs, axis=0)
            std = np.std(tcs, axis=0)
            axs[i].plot(x, mean, c=ls_colors[0], linewidth=lw)
            axs[i].fill_between(x, 
                                mean-std, mean+std, alpha=alpha, color=ls_colors[0])

            tcs = np.array([data.d_sta[n_ID]['BlueTimeCourse'] for n_ID in ls_cells])

            mean = np.mean(tcs, axis=0)
            std = np.std(tcs, axis=0)
            axs[i].plot(x, mean, c=ls_colors[1], linewidth=lw)
            axs[i].fill_between(x,
                                mean-std, mean+std, alpha=alpha, color=ls_colors[1])

        else:    
            for n_ID in ls_cells:
                axs[i].plot(x, data.d_sta[n_ID]['RedTimeCourse'], c=ls_colors[0], alpha=alpha, linewidth=lw)
                axs[i].plot(x, data.d_sta[n_ID]['BlueTimeCourse'], c=ls_colors[1], alpha=alpha, linewidth=lw)

        
        axs[i].set_title(str_label+f' (n={len(ls_cells)}) timecourses')
        n_spacing = 10
        axs[i].set_xlim(right=0)
        if n_dt is not None:
            axs[i].set_xlabel('Time (ms)')
            # xticks = np.round(np.arange(-n_pts*n_dt, 0, n_dt*n_spacing), decimals=-1)
            xticks = np.arange(-500, 10, 100)
            axs[i].set_xticks(xticks)
        else:
            axs[i].set_xlabel('Time (frames)')
            axs[i].set_xticks(np.arange(0, n_pts, n_spacing))

    axs[0].set_ylabel('STA (a.u.)')
    # Set axs and figure bg transparent
    for ax in axs:
        ax.set_facecolor('none')
        ax.grid(True)
    
    return axs

def plot_type_rfs_and_tcs(data: so.SpikeOutputs, ls_RGC_keys=None, d_rf_kwargs={}, d_tc_kwargs={}):
    #['OffP', 'OffM', 'OnP', 'OnM', 'SBC']):
    rf_axs = plot_type_rfs(data, ls_RGC_keys, **d_rf_kwargs)
    tc_axs = plot_type_tcs(data, ls_RGC_keys, **d_tc_kwargs)

    return [rf_axs, tc_axs]