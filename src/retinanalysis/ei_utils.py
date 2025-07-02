import numpy as np
import matplotlib.pyplot as plt
import visionloader as vl

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

def get_top_electrodes(n_ID: int, vcd: vl.VisionCellDataTable, n_interval=2, n_markers=5, b_sort=True):
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
        amin_ei_ts = np.zeros(n_markers)
        for i in range(n_markers):
            y, x = np.unravel_index(top_idx[i], ei_map.shape)
            # ei_ts = ei_grid[:, y, x]
            ei_ts = ei[y, x, :]
            amin_ei_ts[i] = np.argmin(ei_ts)
        top_idx = top_idx[np.argsort(amin_ei_ts)]

    return top_idx


def plot_ei_map(n_ID: int, vcd: vl.VisionCellDataTable, top_idx=None, axs=None, label=None):
    if top_idx is None:
        # Get top electrodes if not provided
        top_idx = get_top_electrodes(n_ID, vcd)

    # Reshape EI timeseries
    ei = vcd.get_ei_for_cell(n_ID).ei
    sorted_electrodes = sort_electrode_map(vcd.get_electrode_map())
    ei = reshape_ei(ei, sorted_electrodes)
    ei_map = np.max(np.abs(ei), axis=2)
    # Log is better for visualization
    ei_map = np.log10(ei_map + 1e-6)
    
    # top_idx is the top n_markers pixels to plot, returned by get_top_electrodes
    n_markers = len(top_idx)
    if axs is None:
        f, axs = plt.subplots(nrows=n_markers+1, figsize=(6, 8),
                              gridspec_kw={'height_ratios': [1]+[1/n_markers]*n_markers})

    sample_rate = 20000.0 # Hz
    sts = vcd.get_spike_times_for_cell(n_ID)
    num_sps = len(sts)
    max_st = sts.max()/sample_rate
    avg_rate = num_sps/max_st

    ax0 = axs[0]
    im=ax0.imshow(ei_map, cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax0, label='log10(abs(EI amplitude))')

    # Get index of peak
    peak_channel = np.argmax(ei_map)
    peak_idx = np.unravel_index(peak_channel, ei_map.shape)
    ax0.plot(peak_idx[1], peak_idx[0], 'o', color='blue')
    ax0.axhline(peak_idx[0], color='blue')
    ax0.axvline(peak_idx[1], color='blue')

    for i in range(n_markers):
        y, x = np.unravel_index(top_idx[i], ei_map.shape)
        ax0.plot(x, y, 'o', color='C2', ms=5)
        ax0.text(x, y, str(i), color='k')

        ax = axs[i+1]
        # ei_ts = ei_grid[:, y, x]
        ei_ts = ei[y, x, :]
        ax.plot(ei_ts, 'C2')
        ax.axvline(np.argmin(ei_ts), color='k')
        ax.set_xticks([])
        ax.set_ylabel(f'{i} (e{top_idx[i]})')
    
    # Set xticks for last ax
    ax.set_xticks(np.arange(0, len(ei_ts), 50))
    ax.set_xlabel('Timeframe')

    str_title = ''
    if label is not None:
        str_title += f'{label} '
    str_title += f'ID {n_ID}\nPeak: {peak_idx}, e{peak_channel}\n{num_sps} sps ({avg_rate:.1f} Hz)\n'
    ax0.set_title(str_title)
    plt.tight_layout()
    return ax0