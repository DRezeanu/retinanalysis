from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visionloader import VisionCellDataTable

import numpy as np
import matplotlib.pyplot as plt



def sort_electrode_map(electrode_map: np.ndarray) -> np.ndarray:
    """
    Sort electrodes by their x, y locations.

    This uses lexsort to sort electrodes by their x, y locations
    First sort by rows, break ties by columns. 
    As each row is jittered but within row the electrodes have exact same y location.

    Parameters:
    electrode_map (numpy.ndarray): The electrode locations of shape (512, 2).

    Returns:
    numpy.ndarray: Sorted indices of the electrodes (512,). 
    """
    sorted_indices = np.lexsort((electrode_map[:, 0], electrode_map[:, 1]))
    return sorted_indices

def reshape_ei(ei: np.ndarray, sorted_electrodes: np.ndarray,
               n_rows: int=16) -> np.ndarray:
    """
    Reshape the EI matrix from 512 x 201 to 16 x 32 x 201 based on electrode locations.

    Parameters:
    ei (numpy.ndarray): The EI matrix of shape (electrode, frames).
    sorted_electrodes (numpy.ndarray): The sorted indices of the electrodes.
    n_rows (int): The number of rows to reshape the EI matrix into. Default is 16.

    Returns:
    numpy.ndarray: The reshaped EI matrix of shape (16, 32, 201).
    """
    if ei.shape[0] != 512:
        print(f'Warning: Expected EI shape (512, 201), got {ei.shape}')
    n_electrodes = ei.shape[0]
    n_frames = ei.shape[1]
    n_cols = n_electrodes // n_rows  # Assuming 512 electrodes and 16 rows

    if n_cols * n_rows != n_electrodes:
        raise ValueError(f"Number of electrodes {n_electrodes} is not compatible with {n_rows} rows and {n_cols} columns.")

    sorted_ei = ei[sorted_electrodes]

    # Reshape the sorted EI matrix
    reshaped_ei = sorted_ei.reshape(n_rows, n_cols, n_frames)

    return reshaped_ei

def get_ei_ts_min(ei, e_idxs, shape=(16, 32)):
    n_markers = len(e_idxs)
    ei_ts_min = np.zeros(n_markers)
    for i in range(n_markers):
        y, x = np.unravel_index(e_idxs[i], shape)
        ei_ts = ei[y, x, :]
        ei_ts_min[i] = np.argmin(ei_ts)
    return ei_ts_min


def get_top_electrodes(n_ID: int, vcd: VisionCellDataTable, n_interval=2, n_markers=5, b_sort=True):
    # Reshape EI timeseries
    ei = vcd.get_ei_for_cell(n_ID).ei
    sorted_electrodes = sort_electrode_map(vcd.get_electrode_map())
    ei = reshape_ei(ei, sorted_electrodes)

    # Get EI map = abs max projection across timeframes
    ei_map = np.max(np.abs(ei), axis=2)
    ei_map = np.log10(ei_map + 1e-6)

    ## Label top n_markers pixels spaced by n_interval in the heatmap
    # Sorted index of pixels
    ei_map_sidx = np.argsort(ei_map.flatten())[::-1]
    top_idx = ei_map_sidx[::n_interval][:n_markers]

    # Sort top_idx by argmin of EI time series
    if b_sort:
        amin_ei_ts = get_ei_ts_min(ei, top_idx, ei_map.shape)
        top_idx = top_idx[np.argsort(amin_ei_ts)]

    return top_idx


def get_ei_and_map(n_ID: int, vcd: VisionCellDataTable):
     # Reshape EI timeseries
    ei = vcd.get_ei_for_cell(n_ID).ei
    sorted_electrodes = sort_electrode_map(vcd.get_electrode_map())
    ei = reshape_ei(ei, sorted_electrodes)
    ei_map = np.max(np.abs(ei), axis=2)
    return ei, ei_map

def plot_ei_spatial_map(ei_map, sorted_electrodes, top_idx, axs=None, n_ID=None, vcd=None):
    """
    Plot the spatial EI map and highlight the peak and selected electrodes.
    """
    n_markers = len(top_idx)
    if axs is None:
        f, ax = plt.subplots(figsize=(6, 4))
    else:
        ax = axs

    im = ax.imshow(ei_map, cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax, label='log10(abs(EI amplitude))')

    # Get index of peak
    peak_channel = np.argmax(ei_map)
    peak_idx = np.unravel_index(peak_channel, ei_map.shape)
    peak_channel_idx = sorted_electrodes[peak_channel]
    ax.plot(peak_idx[1], peak_idx[0], 'o', color='blue')
    ax.axhline(peak_idx[0], color='blue')
    ax.axvline(peak_idx[1], color='blue')

    for i in range(n_markers):
        top = top_idx[i]
        channel_idx = sorted_electrodes[top]
        y, x = np.unravel_index(top, ei_map.shape)
        ax.plot(x, y, 'o', color='C2', ms=5)
        ax.text(x, y, str(i), color='k')

    str_title = ''
    if n_ID is not None and vcd is not None:
        sample_rate = 20000.0 # Hz
        sts = vcd.get_spike_times_for_cell(n_ID)
        num_sps = len(sts)
        max_st = sts.max()/sample_rate
        avg_rate = num_sps/max_st
        str_title += f'ID {n_ID}\nPeak: {peak_idx[0].item(), peak_idx[1].item()}, e{peak_channel_idx}\n{num_sps} sps ({avg_rate:.1f} Hz)\n'
    ax.set_title(str_title)
    return ax

def plot_ei_timeseries(ei, sorted_electrodes, top_idx, 
    axs=None, c='C2', label=None, b_vline=True, b_title=False):
    """
    Plot the EI timeseries for the selected electrodes.
    """
    n_markers = len(top_idx)
    if axs is None:
        f, axs = plt.subplots(nrows=n_markers, figsize=(6, 4))

    if n_markers == 1:
        axs = [axs]

    for i in range(n_markers):
        top = top_idx[i]
        channel_idx = sorted_electrodes[top]
        y, x = np.unravel_index(top, ei.shape[:2])
        ei_ts = ei[y, x, :]
        ax = axs[i]
        ax.plot(ei_ts, c, label=label)
        
        ax.set_xticks([])
        if b_title:
            ax.set_title(f'{i} (e{channel_idx})')
        else:
            ax.set_ylabel(f'{i} (e{channel_idx})')
        if i == n_markers - 1:
            ax.set_xticks(np.arange(0, len(ei_ts), 50))
            ax.set_xlabel('Timeframe')

        if b_vline:
            ax.axvline(np.argmin(ei_ts), color='k')

    return axs

def plot_ei_map(n_ID: int, vcd: VisionCellDataTable, top_idx=None, 
                axs=None, n_interval=2, n_markers=5, 
                **ts_kwargs):
    
    sorted_electrodes = sort_electrode_map(vcd.get_electrode_map())
    if top_idx is None:
        # Get top electrodes if not provided
        top_idx = get_top_electrodes(
            n_ID, vcd, n_interval=n_interval, 
            n_markers=n_markers, b_sort=True)

    ei, ei_map = get_ei_and_map(n_ID, vcd)
    # Log is better for visualization
    ei_map = np.log10(ei_map + 1e-6)
    
    # top_idx is the top n_markers pixels to plot, returned by get_top_electrodes
    n_markers = len(top_idx)
    # Prepare axes if not provided
    if axs is None:
        fig, axs = plt.subplots(
            nrows=n_markers + 1,
            figsize=(6, 8),
            gridspec_kw={'height_ratios': [1] + [1 / n_markers] * n_markers}
        )

    # Plot spatial map on first axis
    plot_ei_spatial_map(
        ei_map, sorted_electrodes, top_idx,
        axs=axs[0], n_ID=n_ID, vcd=vcd
    )

    # Plot timeseries on remaining axes
    if n_markers > 0:
        plot_ei_timeseries(
            ei, sorted_electrodes, top_idx, axs=axs[1:], 
            **ts_kwargs
        )

    plt.tight_layout()
    return axs


def compute_ei_map_grid(
    ei: np.ndarray,
    channel_positions: np.ndarray) -> np.ndarray:
    from scipy.interpolate import griddata

    if ei.shape[0] != 512:
        print(f'Warning: Expected EI shape (512, n_timepoints), got {ei.shape}')

    xrange = (np.min(channel_positions[:, 0]), np.max(channel_positions[:, 0]))
    yrange = (np.min(channel_positions[:, 1]), np.max(channel_positions[:, 1]))

    y_dim = 30 # Fixed y dimension for scaling
    x_dim = int((xrange[1] - xrange[0])/(yrange[1] - yrange[0]) * y_dim) # x dim is proportional to y dim

    x_e = np.linspace(xrange[0], xrange[1], x_dim)
    y_e = np.linspace(yrange[0], yrange[1], y_dim)

    grid_x, grid_y = np.meshgrid(x_e, y_e)
    grid_x = grid_x.T
    grid_y = grid_y.T

    # ei_energy = np.log10(np.mean(np.power(ei, 2), axis=1) + .000000001)
    ei_energy = np.log10(np.max(np.abs(ei), axis=1) + 1e-9)
    ei_energy_grid = griddata(
        channel_positions, ei_energy, 
        (grid_x, grid_y), method='linear', 
        fill_value=np.median(ei_energy))

    return ei_energy_grid.T

def plot_test(n_ID, vcd):
    import matplotlib.colors as mcolors
    
    sorted_electrodes = sort_electrode_map(vcd.get_electrode_map())

    ei, ei_map = get_ei_and_map(n_ID, vcd)
    top_idx = get_top_electrodes(n_ID, vcd, n_interval=1, n_markers=20, b_sort=True)
    ei_ts_min = get_ei_ts_min(ei, top_idx, ei_map.shape)
    diff = np.abs(np.diff(ei_ts_min))

    # Get idx where ts diff is > threshold
    threshold_delta = 1
    idx_large_diff = np.where(diff >= threshold_delta)[0]

    top_idx = top_idx[idx_large_diff]
    diff = diff[idx_large_diff]
    c_idx = sorted_electrodes[top_idx]

    # Electrode locations
    electrode_map = vcd.get_electrode_map()
    elec_locs = electrode_map[c_idx]

    # Compute speeds
    speeds = []
    for i in range(len(elec_locs)-1):
        dist = np.linalg.norm(elec_locs[i+1] - elec_locs[i])
        dt = diff[i]
        speed = dist / dt
        speeds.append(speed)
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(speeds), vmax=max(speeds))
    f, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(elec_locs[:, 0], elec_locs[:, 1], c='k', s=40)
    for i in range(len(elec_locs)-1):
        cval = speeds[i]
        color = cmap(norm(cval))
        ax.plot([elec_locs[i, 0], elec_locs[i+1, 0]], [elec_locs[i, 1], elec_locs[i+1, 1]],
                color=color, lw=3)
        # ax.text(elec_locs[i, 0], elec_locs[i, 1], f"{cval:.2f}", color='k')
    # Overlap ei map
    ei_map = compute_ei_map_grid(vcd.get_ei_for_cell(n_ID).ei, electrode_map)

    ax.imshow(ei_map[::-1], cmap='hot', alpha=0.3, 
    extent=[electrode_map[:,0].min(), electrode_map[:,0].max(),
            electrode_map[:,1].min(), electrode_map[:,1].max()], aspect='auto')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label='Propagation speed (distance/frame)')
    ax.set_title(f"Spike propagation speed for cell {n_ID}")
    




def plot_spike_propagation_speed(n_ID: int, vcd, n_markers=5, n_interval=2, ax=None, cmap='viridis'):
    """
    Plot spike propagation speed between spatially closest high-amplitude electrodes.
    Lines are colored by speed (distance/time).
    """
    # Get EI and electrode map
    ei = vcd.get_ei_for_cell(n_ID).ei
    electrode_map = vcd.get_electrode_map()  # shape (512, 2)
    sorted_electrodes = sort_electrode_map(electrode_map)
    ei = reshape_ei(ei, sorted_electrodes)
    ei_map = np.max(np.abs(ei), axis=2)
    ei_map = np.log10(ei_map + 1e-6)

    # Get top electrodes (indices in flattened grid)
    top_idx = get_top_electrodes(n_ID, vcd, n_interval=n_interval, n_markers=n_markers, b_sort=True)
    # Get their (y, x) grid positions and original electrode indices
    grid_pos = [np.unravel_index(idx, ei_map.shape) for idx in top_idx]
    elec_idx = [sorted_electrodes[idx] for idx in top_idx]
    # Get their physical locations
    elec_locs = np.array([electrode_map[eidx] for eidx in elec_idx])  # shape (n_markers, 2)

    # Get time of minimum for each
    min_times = []
    for (y, x) in grid_pos:
        ei_ts = ei[y, x, :]
        min_times.append(np.argmin(ei_ts))
    min_times = np.array(min_times)

    # Find spatially closest pairs (using KDTree for efficiency)
    from scipy.spatial import KDTree
    tree = KDTree(elec_locs)
    pairs = []
    for i, loc in enumerate(elec_locs):
        dists, idxs = tree.query(loc, k=2)  # k=2: self and nearest neighbor
        j = idxs[1]  # idxs[0] is self
        if i < j:  # avoid duplicates
            pairs.append((i, j, dists[1]))

    # Calculate speeds and prepare for plotting
    speeds = []
    for i, j, dist in pairs:
        dt = abs(min_times[j] - min_times[i])
        if dt == 0:
            speed = np.nan  # avoid division by zero
        else:
            speed = dist / dt
        speeds.append(speed)
    speeds = np.array(speeds)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(elec_locs[:, 0], elec_locs[:, 1], c='k', s=40, zorder=2)
    norm = mcolors.Normalize(vmin=np.nanmin(speeds), vmax=np.nanmax(speeds))
    cmap = plt.get_cmap(cmap)
    for idx, (i, j, dist) in enumerate(pairs):
        cval = speeds[idx]
        # color = plt.get_cmap(cmap)(mcolors.Normalize()(cval if np.isfinite(cval) else 0))
        color = cmap(norm(cval)) if np.isfinite(cval) else (0.5, 0.5, 0.5, 1.0)
        ax.plot([elec_locs[i, 0], elec_locs[j, 0]], [elec_locs[i, 1], elec_locs[j, 1]],
                color=color, lw=3, zorder=1)
        # Optionally, annotate speed
        # ax.text((elec_locs[i, 0]+elec_locs[j, 0])/2, (elec_locs[i, 1]+elec_locs[j, 1])/2, f"{cval:.2f}", color='k')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=np.nanmin(speeds), vmax=np.nanmax(speeds)))
    plt.colorbar(sm, ax=ax, label='Propagation speed (distance/frame)')
    ax.set_title(f"Spike propagation speed for cell {n_ID}")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    return ax, speeds