import numpy as np
import matplotlib.pyplot as plt
import retinanalysis as ra
import os
# Set axes title fontsize to 16
plt.rcParams['axes.titlesize'] = 16


def plot_cell_of_interest(rb: ra.MEAResponseBlock, ac: ra.AnalysisChunk, 
                          cell_id: int, d_metrics: dict, n_pad=2, str_save=None):
    window_ms = d_metrics['window_ms']                          
    onset_time = np.mean(rb.d_timing['actual_onset_times_ms'])
    offset_time = np.mean(rb.d_timing['actual_offset_times_ms'])
    c_idx = np.where(rb.df_spike_times['cell_id'] == cell_id)[0][0]
    onset_sps = d_metrics['onset_sps'][c_idx].mean()
    offset_sps = d_metrics['offset_sps'][c_idx].mean()
    stable_sps = d_metrics['stable_sps'][c_idx].mean()

    all_sts = rb.df_spike_times.at[c_idx, 'spike_times']

    f, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(20,6), 
        gridspec_kw={'width_ratios': [3, 1, 1]})

    # Get cell type from all typing files
    if 'typing_file_0' in ac.df_cell_params.columns:
        types = ac.df_cell_params.query('cell_id==@cell_id').loc[:,'typing_file_0':].values.flatten()
        f.suptitle(f'Cell ID {cell_id}\n{types}')
    else:
        f.suptitle(f'Cell ID {cell_id}\nNo typing files found')


    # For first ax, plot spike times till window after pretime
    ax = axs[0, 0]
    
    max_t = onset_time + window_ms*n_pad
    plot_sts = [sts[sts < max_t] for sts in all_sts]
    ax.eventplot(plot_sts)
    ax.axvline(onset_time, c='k')
    ax.axvline(onset_time + window_ms, c='k', linestyle='--')
    y_max = ax.get_ylim()[1]
    ax.text(onset_time + 0.1, y_max-1, 'Onset', color='k', fontsize=12)
    ax.set_xlim((0, max_t))
    ax.set_ylabel('Epoch index')
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.set_title(f'Onset rate: {onset_sps:.2f}')

    # For next ax, plot spike times in window before stimulus offset
    ax = axs[1, 0]
    min_t = offset_time - window_ms*n_pad
    plot_sts = [sts[sts >= min_t] for sts in all_sts]
    plot_sts = [sts - min_t for sts in plot_sts]
    ax.eventplot(plot_sts)
    ax.axvline(offset_time-min_t, c='k')
    ax.axvline(offset_time-min_t - window_ms, c='k', linestyle='--')
    y_max = ax.get_ylim()[1]
    ax.text(offset_time-min_t + 0.1, y_max-1, 'Offset', color='k', fontsize=12)
    ax.set_ylabel('Epoch index')
    ax.set_xlabel('Time (ms)')
    ax.set_xlim((0, max_t))
    ax.set_title(f'Stable rate: {stable_sps:.2f}, Post-Offset rate: {offset_sps:.2f}')

    # Plot spatial map
    if hasattr(ac, 'd_spatial_maps'):
        if cell_id in ac.d_spatial_maps:
            ax = axs[0, 1]
            sm = ac.d_spatial_maps[cell_id][:,:,0]
            im = ax.imshow(sm, cmap='viridis')
            ax.set_title('Spatial Map')
    else:
        ax = axs[0, 1]
        ax.text(0.5, 0.5, 'No spatial map found', ha='center', va='center')
        ax.axis('off')

    # Plot timecourse
    ax = axs[0, 2]
    rg_tc = ac.vcd.main_datatable[cell_id]['GreenTimeCourse']
    b_tc = ac.vcd.main_datatable[cell_id]['BlueTimeCourse']
    n_pts = len(b_tc)
    time = np.arange(n_pts) * 1/120*1e3
    time = -time[::-1]
    ax.plot(time, rg_tc, label='RedGreen', c='k')
    ax.plot(time, b_tc, label='Blue', c='grey')
    ax.set_xlim(-200, 0)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('STA (au)')
    ax.set_title('Timecourse')
    ax.grid()
    ax.legend()

    # Plot ISI
    ax = axs[1, 1]
    diffs = np.concatenate([np.diff(sts) for sts in all_sts])
    bin_edges = np.linspace(0,300,301)
    isi = np.histogram(diffs, bins=bin_edges)[0]
    if np.sum(isi) > 0:
        isi = isi / np.sum(isi)
    ax.plot(bin_edges[:-1], isi, c='k')
    ax.set_xlim(0,100)
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Inter-Spike Interval')

    # Plot EI
    ei, ei_map = ra.ei_utils.get_ei_and_map(cell_id, ac.vcd)
    # Log is better for visualization
    ei_map = np.log10(ei_map + 1e-6)
    ax = axs[1, 2]
    im = ax.imshow(ei_map, cmap='hot', aspect='auto')
    # plt.colorbar(im, ax=ax, label='log10(abs(EI amplitude))')
    ax.axis('off')
    ax.set_title('log EI Map')

    plt.tight_layout()

    if str_save is not None:
        plt.savefig(str_save, bbox_inches='tight')
        plt.close()

def compute_metrics(rb: ra.MEAResponseBlock, window_ms=2000.0):
    onset_times_ms = rb.d_timing['actual_onset_times_ms']
    offset_times_ms = rb.d_timing['actual_offset_times_ms']
    tail_time_ms = rb.d_timing['tail_time_ms']
    n_cells = len(rb.cell_ids)
    n_epochs = rb.n_epochs
    
    onset_sps = np.zeros((n_cells, n_epochs))
    stable_sps = np.zeros((n_cells, n_epochs))
    offset_sps = np.zeros((n_cells, n_epochs))
    metrics = np.zeros((n_cells, 3))  # onset, stable, offset

    for c_idx, cell_id in enumerate(rb.cell_ids):
        sts = rb.df_spike_times.at[c_idx, 'spike_times']
        for e_idx in range(n_epochs):
            e_sts = sts[e_idx]
            onset_time = onset_times_ms[e_idx]
            offset_time = offset_times_ms[e_idx] 
            
            m1 = np.sum((e_sts >= onset_time) & (e_sts < onset_time + window_ms))
            m2 = np.sum((e_sts >= offset_time - window_ms) & (e_sts < offset_time))
            m3 = np.sum(e_sts >= offset_time)
            onset_sps[c_idx, e_idx] = m1 / window_ms * 1000  # Convert to spikes per second
            stable_sps[c_idx, e_idx] = m2 / window_ms * 1000
            offset_sps[c_idx, e_idx] = m3 / tail_time_ms * 1000  # Convert to spikes per second
            
        metrics[c_idx, 0] = np.mean(onset_sps[c_idx])
        metrics[c_idx, 1] = np.mean(stable_sps[c_idx])
        metrics[c_idx, 2] = np.mean(offset_sps[c_idx])

    d_metrics = {
        'onset_sps': onset_sps,
        'stable_sps': stable_sps,
        'offset_sps': offset_sps,
        'metrics': metrics,
        'window_ms': window_ms,
    }
    return d_metrics

def plot_metrics_summary(metrics, str_save=None):
    n_cells = metrics.shape[0]
    onset_stable_ratio  = metrics[:,0] / metrics[:,1]
    offset_stable_ratio = metrics[:,2] / metrics[:,1]
    # Set nans to mean value
    onset_stable_ratio[np.isnan(onset_stable_ratio)] = np.nanmean(onset_stable_ratio)
    offset_stable_ratio[np.isnan(offset_stable_ratio)] = np.nanmean(offset_stable_ratio)
    
    # Get bottom 20 percent threshold for onset/stable ratio
    threshold_onset_stable = np.percentile(onset_stable_ratio, 20)
    intersection = np.where(onset_stable_ratio < threshold_onset_stable)[0]
    
    # For those cells, get top 20 percent threshold for offset/stable ratio
    filtered_offset_ratio = offset_stable_ratio[intersection]
    threshold_offset_stable = np.percentile(filtered_offset_ratio, 80)
    intersection = intersection[filtered_offset_ratio > threshold_offset_stable]
    others = np.where(~np.isin(np.arange(n_cells), intersection))[0]
    intersection = intersection.astype(int)

    # Sort intersection and others by descending order of offset_stable_ratio
    intersection = intersection[np.argsort(offset_stable_ratio[intersection])[::-1]]
    others = others[np.argsort(offset_stable_ratio[others])[::-1]]
    print(f'Cells below 10% threshold for onset/stable ratio: {len(intersection)}')
    
    # Color the intersection cells in the scatter plot
    colors = np.full(n_cells, 'gray', dtype=object)
    colors[intersection] = 'red'

    f, axs = plt.subplots(ncols=3, figsize=(24,6))
    ax=axs[0]
    ax.scatter(onset_stable_ratio, offset_stable_ratio, alpha=0.8, c=colors)
    ax.set_xlabel('Onset/Stable sps')
    ax.set_ylabel('Offset/Stable sps')
    ax.set_title('Onset/Stable vs Offset/Stable sps')
    ax.axvline(threshold_onset_stable, c='k', linestyle='--', label='10% threshold')
    ax.axhline(threshold_offset_stable, c='k', linestyle='--')
    ax.grid()

    ax=axs[1]
    # ax.plot(metrics[intersection].T, c='red', alpha=0.7, marker='o')
    im = ax.imshow(metrics[intersection], aspect='auto', cmap='Reds')
    plt.colorbar(im, ax=ax, label='Average nSps')
    ax.set_xticks([0, 1, 2], ['Onset', 'Stable', 'Offset'])
    # ax.set_ylabel('Average nSps')
    ax.set_title('Cells of interest')
    ax.set_ylabel('Cell index')

    ax = axs[2]
    # ax.plot(metrics[others].T, c='gray', alpha=0.7, marker='o')
    im = ax.imshow(metrics[others], aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Average nSps')
    ax.set_xticks([0, 1, 2], ['Onset', 'Stable', 'Offset'])
    # ax.set_ylabel('Average nSps')
    ax.set_title('Other cells')
    ax.set_ylabel('Cell index')
    plt.tight_layout()

    if str_save is not None:
        plt.savefig(str_save, bbox_inches='tight')
        plt.close()
    return intersection

def cell_search(exp_name, chunk_name, datafile_name, ss_version, str_save_dir, window_ms=2000.0):
    print(f"Processing {exp_name} {datafile_name} {chunk_name} {ss_version}")
    rb = ra.MEAResponseBlock(exp_name, datafile_name, ss_version=ss_version, include_ei=False)
    ac = ra.AnalysisChunk(exp_name, chunk_name, ss_version=ss_version)
    cell_ids = np.intersect1d(rb.cell_ids, ac.cell_ids)
    rb.cell_ids = cell_ids
    ac.cell_ids = cell_ids

    if 'SpatialNoise' in ac.noise_protocol:
        # Apply timing correction
        frame_times_ms = rb.d_timing['frameTimesMs']
        stage_frame_rate = rb.d_timing['stage_frame_rate']
        pre_time_ms = rb.d_timing['pre_time_ms']
        stim_time_ms = rb.d_timing['stim_time_ms']
        on_frames = np.floor((stim_time_ms+pre_time_ms) * 1e-3 * stage_frame_rate * 1.011).astype(int)
        rb.d_timing['actual_offset_times_ms'] = [frame_times_ms[i][on_frames] for i in range(len(frame_times_ms))]

    d_metrics = compute_metrics(rb, window_ms=window_ms)
    print('Computed metrics.')
    str_summary = os.path.join(str_save_dir, f'1_summary.png')
    intersection = plot_metrics_summary(d_metrics['metrics'], str_save=str_summary)
    print('Saved summary plot.')
    for c_idx in intersection:
        str_save_cell = os.path.join(str_save_dir, f'cell_{c_idx}_id_{rb.cell_ids[c_idx]}.png')
        plot_cell_of_interest(rb, ac, rb.cell_ids[c_idx], d_metrics, str_save=str_save_cell)
    print(f'Plotted {len(intersection)} cells of interest.')
    print('*-----------------------------------------------------*')