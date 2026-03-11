import numpy as np
import gc
import torch
from tqdm import tqdm
import os

import elephant, warnings
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import multiprocessing as mp

import sys
import visionloader as vl
import visionwriter as vw

import subprocess

FS = 20000 # hz
TORCH_N_THREADS = 12
CCH_SAMPLES = 20 # reset to 50 after testing
CCH_WIDTH_THRESHOLD = 0.001
REFRACTORY_PERIOD_MS = 1.5
REFRACTORY_SAMPLES = int((REFRACTORY_PERIOD_MS / 1000) * FS)

def merge_spike_trains(d_sts, str_method, d_eis=None):
    """
    Merges spike trains. 
    - Refractory violations WITHIN a unit are marked valid=False, shared=False.
    - Refractory violations ACROSS units are marked valid=False, shared=True.
    """
    cell_ids = list(d_sts.keys())

    # --- 1. Method Configuration ---
    if str_method == 'corrected_cch':
        raise NotImplementedError("corrected_cch method not implemented yet")
    elif str_method == 'corrected_ei':
        raise NotImplementedError("corrected_ei method not implemented yet")
    elif str_method == 'dirty':
        d_shifts = {cid: 0 for cid in cell_ids}
        min_isi_samples = REFRACTORY_SAMPLES
        print("Merging spikes without any shift correction ('dirty' method)")
        print(f"Using minimum ISI of {min_isi_samples} samples ({(min_isi_samples/FS)*1000:.2f} ms)")
    else:
        raise ValueError(f"Unknown method: {str_method}")

    # --- 2. Vectorized Data Collection ---
    original_unit_list = []
    original_time_list = []

    for cid in cell_ids:
        sts = np.array(d_sts[cid], dtype=int)
        # Remove isi violating spikes within each cell
        isi = np.diff(sts)
        bad = np.where(isi < min_isi_samples)[0] + 1
        
        if len(bad) > 0:
            print(f"Cell {cid}: Removing {len(bad)} of {sts.shape[0]} spikes violating refractory period")
            sts = np.delete(sts, bad)
            print(f"Cell {cid}: {sts.shape[0]} spikes remain after removal")
        n_sps = sts.shape[0]
        original_unit_list.append(np.full(n_sps, cid))
        original_time_list.append(sts)
    
    if not original_time_list:
        return pd.DataFrame(columns=['original_unit', 'original_spike_time', 
                                     'adjusted_spike_time', 'valid', 'shared', 'spike_id'])

    original_unit = np.concatenate(original_unit_list)
    original_spike_time = np.concatenate(original_time_list)
    total_spikes = len(original_spike_time)

    # --- 3. Adjust and Sort ---
    shifts = np.array([d_shifts[cid] for cid in original_unit])
    adjusted_spike_time = original_spike_time - shifts

    # Sort everything by the adjusted time
    order = np.argsort(adjusted_spike_time)
    original_unit = original_unit[order]
    original_spike_time = original_spike_time[order]
    adjusted_spike_time = adjusted_spike_time[order]

    # --- 4. Linear Refractory Scan with Unit Check ---
    valid = np.zeros(total_spikes, dtype=bool)
    shared = np.zeros(total_spikes, dtype=bool)

    last_valid_idx = -1

    for i in range(total_spikes):
        t = adjusted_spike_time[i]
        
        # Check if this is the first spike OR if it is far enough from the last VALID spike
        if last_valid_idx == -1 or (t - adjusted_spike_time[last_valid_idx] >= min_isi_samples):
            valid[i] = True
            last_valid_idx = i
        else:
            # COLLISION DETECTED (Refractory Violation)
            valid[i] = False
            
            # Check who we collided with
            current_unit = original_unit[i]
            survivor_unit = original_unit[last_valid_idx]
            
            if current_unit != survivor_unit:
                # Different units -> This is a MERGE
                shared[i] = True
                shared[last_valid_idx] = True
            else:
                # Same unit -> This is just noise/violation
                # shared remains False
                pass

    # --- 5. ID Assignment ---
    # Invalid spikes inherit the ID of the preceding valid spike
    spike_id = np.cumsum(valid) - 1

    # --- 6. Construct DataFrame ---
    df_st = pd.DataFrame({
        'original_unit': original_unit,
        'original_spike_time': original_spike_time,
        'adjusted_spike_time': adjusted_spike_time,
        'valid': valid,
        'shared': shared,
        'spike_id': spike_id
    })
    
    return df_st


def summarize_merger_df(df_st):
    # For each original unit, summarize the number of unique and shared spikes
    summary = []
    for cid, group in df_st.groupby('original_unit'):
        n_shared = group['shared'].sum()
        n_unique = (~group['shared']).sum()
        n_total = len(group)
        summary.append({
            'original_unit': cid,
            'n_unique': n_unique,
            'n_shared': n_shared,
            'n_total': n_total
        })
    return pd.DataFrame(summary)


# def calculate_cch(spikes_delay, spikes_all, num_cells, t_stop, spike_bin_size = 1 * pq.s):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         spike_trains = [SpikeTrain(spikes_all[cc], t_stop=t_stop) for cc in range(num_cells * 2)]
#         binned_spike_trains = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_cells * 2)], bin_size=spike_bin_size)

#     spike_corrs_big = elephant.spike_train_correlation.correlation_coefficient(binned_spike_trains)
#     cch_all = spike_corrs_big[:num_cells, num_cells:]
#     return cch_all

# def get_cchs(d_merge: dict, vcd: vl.VisionCellDataTable):
#     print('Calculating CCHs')
    
#     spike_bin_size = 1 * pq.s
#     cchs = {}
    

#     all_spike_times = []
#     all_unit_ids = []
    
#     for ref_unit, comp_units in d_merge.items():
#         st = vcd.get_spike_times_for_cell(ref_unit)
#         all_spike_times.append(st * pq.s)
#         all_unit_ids.append(ref_unit)

#         for cell_id in comp_units:
#             st = vcd.get_spike_times_for_cell(cell_id)
#             all_spike_times.append(st * pq.s)
#             all_unit_ids.append(cell_id)
    
#     num_cells = len(all_spike_times)
#     all_unit_inds = {unit_id:ind for ind, unit_id in enumerate(all_unit_ids)}
    
#     delays_half = np.around(np.arange(0,CCH_SAMPLES),0) * pq.s
    
#     cch_all = np.zeros((num_cells, num_cells, len(delays_half)))
#     t_stop = np.ceil(np.max([np.max(all_spike_times[cc]) for cc in range(num_cells)])) + 100
    
#     params = []
    
#     # for dd, delay in tqdm(enumerate(delays_half), total=len(delays_half)):
#     #     spikes_delay = [s + delay for s in all_spike_times]
#     #     spikes_all = all_spike_times + spikes_delay

#         # with warnings.catch_warnings():
#         #     warnings.simplefilter("ignore")
#         #     spike_trains = [SpikeTrain(spikes_all[cc], t_stop=t_stop) for cc in range(num_cells * 2)]
#         #     binned_spike_trains = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_cells * 2)], bin_size=spike_bin_size)

#         # spike_corrs_big = elephant.spike_train_correlation.correlation_coefficient(binned_spike_trains)
#         # cch_all[:, :, dd] = spike_corrs_big[:num_cells, num_cells:]
    
    
#     with mp.Pool(processes=TORCH_N_THREADS) as pool:
#         for dd, delay in enumerate(delays_half):
#             spikes_delay = [s + delay for s in all_spike_times]
#             spikes_all = all_spike_times + spikes_delay
#             params.append((spikes_delay, spikes_all, num_cells, t_stop, spike_bin_size))
#         cch_all_list = list(tqdm(pool.starmap(calculate_cch, params), total=len(params)))
#         pool.close()
#         pool.join()
#     cch_all = np.array(cch_all_list).transpose(1,2,0)
    
#     cch = np.concatenate([np.flip(cch_all, 2), np.swapaxes(cch_all, 0,1)[...,1:]], axis=2)
#     delays = np.concatenate([-np.flip(delays_half), delays_half[1:]])
    
#     for ref_unit, comp_units in d_merge.items():
#         cchs[ref_unit] = np.array([cch[0,all_unit_inds[comp_unit],:] for comp_unit in comp_units])
    
#     return cchs, delays

# def merge_spikes_for_unit(ref_id, units_to_merge, cchs, delays, vcd: vl.VisionCellDataTable):
#     if len(units_to_merge) == 1:
#         return vcd.get_spike_times_for_cell(units_to_merge[0])

#     ref_cchs = cchs[ref_id]
#     cch_widths = [0]
#     peak_indices = [len(delays)//2]
#     df_st = {'original_unit':[], 'original_spike_time':[], 'adjusted_spike_time':[], 'unique':[],'valid':[], 'shared':[], 'spike_id':[]}
    
#     for merge_cch in ref_cchs:
#         peak = np.argmax(merge_cch)
#         peak_indices.append(peak)

#         nonzero_diff = np.abs(np.diff(merge_cch, n=1)) > CCH_WIDTH_THRESHOLD
#         cch_width = 0
#         forward, backward = np.split(nonzero_diff, [peak])
#         forward = forward[::-1]
#         for f in forward:
#             if f:
#                 cch_width += 1
#             else:
#                 break
#         for b in backward:
#             if b:
#                 cch_width += 1
#             else:
#                 break
#         cch_widths.append(cch_width)
        
#     max_width = np.max(cch_widths)
    
#     for i, cid in enumerate(units_to_merge):
#         for spike_time in vcd.get_spike_times_for_cell(cid):
#             df_st['original_unit'].append(cid)
#             df_st['original_spike_time'].append(spike_time)
#             df_st['adjusted_spike_time'].append(spike_time - int(delays[peak_indices[i]]))
#             df_st['valid'].append(True)
#             df_st['unique'].append(True)
#             df_st['spike_id'].append(0)
#             df_st['shared'].append(False)
#     df_st = pd.DataFrame(df_st)
#     df_st.sort_values(by='adjusted_spike_time', inplace=True)
#     df_st.reset_index(inplace=True, drop=True)

#     # set all spikes within refractory period to not unique
#     bad_spikes = np.where(np.diff(df_st.adjusted_spike_time) < np.max([max_width, REFRACTORY_SAMPLES]))[0]
#     df_st.loc[bad_spikes+1, 'unique'] = False
#     df_st.loc[bad_spikes, 'shared'] = True
#     df_st.loc[bad_spikes+1, 'shared'] = True

#     id = -1
#     for ind, spike in df_st.iterrows():
#         if spike['unique']:
#             id += 1
#             df_st.loc[ind, 'spike_id'] = id
#         else:
#             df_st.loc[ind, 'spike_id'] = id

#     # return df_st.loc[df_st.unique, 'adjusted_spike_time'].values
#     return df_st