from __future__ import annotations
from typing import (Union,
                    List,
                    Dict,
                    Tuple,
                    TYPE_CHECKING)

if TYPE_CHECKING:
    from retinanalysis.classes.analysis_chunk import AnalysisChunk
    from retinanalysis.classes.response import MEAResponseBlock
    from visionloader import VisionCellDataTable
    
from retinanalysis.utils import (NAS_DATA_DIR,
                                 NAS_ANALYSIS_DIR,
                                 get_exp_summary)

import os
import numpy as np
# from retinanalysis.utils.datajoint_utils import get_exp_summary
from visionloader import load_vision_data

from matplotlib.patches import Ellipse
import xarray as xr

def get_analysis_vcd(exp_name: str, chunk_name: str, ss_version: str,
                    include_ei: bool = True, include_neurons: bool = True,
                    verbose: bool = True) -> VisionCellDataTable:
        data_path = os.path.join(NAS_ANALYSIS_DIR, exp_name, chunk_name, ss_version)
        
        if verbose:
            print(f'Loading VCD from {data_path} ...')

        vcd = load_vision_data(data_path, ss_version, include_ei = include_ei,
                                  include_noise = False, include_sta = False,
                                  include_params = True, include_runtimemovie_params = True,
                                  include_neurons = include_neurons)
        
        if verbose:
            print(f'VCD loaded with {len(vcd.get_cell_ids())} cells.\n')

        return vcd

def get_protocol_vcd(exp_name: str, datafile_name: str, ss_version: str,
                     verbose: bool = True) -> VisionCellDataTable:
        
        data_path = os.path.join(NAS_DATA_DIR, exp_name, datafile_name, ss_version)
        
        if verbose:
            print(f'Loading VCD from {data_path} ...')
            
        vcd = load_vision_data(
            data_path, datafile_name, 
            include_ei = True, include_neurons = True
            )
        if verbose:
            print(f'VCD loaded with {len(vcd.get_cell_ids())} cells.\n')

        return vcd

def get_roi_dict(location: List[float], distance_x: float, distance_y: float):
    
    roi = dict()
    roi['x_min'] = location[0] - distance_x
    roi['x_max'] = location[0] + distance_x
    roi['y_min'] = location[1] - distance_y
    roi['y_max'] = location[1] + distance_y
    
    return roi

def cluster_match(ref_object: Union[AnalysisChunk, MEAResponseBlock], test_object: Union[AnalysisChunk, MEAResponseBlock],
                corr_cutoff: float = 0.8, method: str = 'all', use_isi: bool = False,
                use_timecourse: bool = False, n_removed_channels: int = 1, verbose: bool = True):


        
        ref_vcd = ref_object.vcd
        test_vcd = test_object.vcd
        ref_ids = ref_object.cell_ids
        test_ids = test_object.cell_ids
        
        if 'all' in method:
            arr_full_corr: np.ndarray = ei_corr(ref_object, test_object, method = 'full', n_removed_channels = n_removed_channels)
            arr_space_corr: np.ndarray = ei_corr(ref_object, test_object, method = 'space', n_removed_channels = n_removed_channels)
            arr_power_corr: np.ndarray = ei_corr(ref_object, test_object, method = 'power', n_removed_channels = n_removed_channels)
        elif 'full' in method:
            arr_full_corr: np.ndarray = ei_corr(ref_object, test_object, method = method, n_removed_channels = n_removed_channels)
            arr_space_corr: np.ndarray = np.zeros(arr_full_corr.shape)
            arr_power_corr: np.ndarray = np.zeros(arr_full_corr.shape)
        elif 'space' in method:
            arr_space_corr: np.ndarray = ei_corr(ref_object, test_object, method = method, n_removed_channels = n_removed_channels)
            arr_full_corr: np.ndarray = np.zeros(arr_space_corr.shape)
            arr_power_corr: np.ndarray = np.zeros(arr_space_corr.shape)
        elif 'power' in method:
            arr_power_corr: np.ndarray = ei_corr(ref_object, test_object, method = method, n_removed_channels = n_removed_channels)
            arr_space_corr: np.ndarray = np.zeros(arr_power_corr.shape)
            arr_full_corr: np.ndarray = np.zeros(arr_power_corr.shape)
        else:
            raise NameError("Method property must be 'all', 'full', 'space', or 'power'")

        # to avoid circular imports, we're only importing classes inside the utils when needed for checking. Annoying but 
        # this is just an issue with python
        from retinanalysis.classes.response import MEAResponseBlock
        if isinstance(ref_object, MEAResponseBlock) or isinstance(test_object, MEAResponseBlock):
            if use_timecourse:
                raise FileNotFoundError("Response blocks don't have .params files, can't use timecourse for cluster matching")

        match_dict = dict()
        match_count = 0
        bad_match_count = 0
        isi_corr = 1
        rgb_corr = 1

        if verbose:
            from retinanalysis.classes.analysis_chunk import AnalysisChunk
            if isinstance(ref_object, AnalysisChunk):
                if isinstance(test_object, AnalysisChunk):
                    print(f"Cluster matching {ref_object.exp_name} {ref_object.chunk_name} with {os.path.splitext(test_object.chunk_name)[1][1:]} ...")
                else:
                    print(f"Cluster matching {ref_object.exp_name} {ref_object.chunk_name} with {os.path.splitext(test_object.protocol_name)[1][1:]} ...")
            else:
                print(f"Cluster matching {ref_object.exp_name} {ref_object.protocol_name} with {os.path.splittext(test_object.protocol_name)[1][1:]} ...")

        for idx, ref_cell in enumerate(ref_ids):
            sorted_full_corr = np.sort(arr_full_corr[idx,:])
            sorted_full_corr = np.flip(sorted_full_corr)

            sorted_space_corr = np.sort(arr_space_corr[idx,:])
            sorted_space_corr = np.flip(sorted_space_corr)

            sorted_power_corr = np.sort(arr_power_corr[idx,:])
            sorted_power_corr = np.flip(sorted_power_corr)

            max_corrs = np.array([sorted_full_corr[0], sorted_space_corr[0], sorted_power_corr[0]])
            next_max_corrs = np.array([sorted_full_corr[1], sorted_space_corr[1], sorted_power_corr[1]])
            corr_filter = next_max_corrs < max_corrs*0.9

            max_inds = np.array([np.argmax(arr_full_corr[idx,:]),
                                 np.argmax(arr_space_corr[idx,:]),
                                 np.argmax(arr_power_corr[idx,:])])

            max_rev_inds = np.array([np.argmax(arr_full_corr[:, max_inds[0]]),
                                     np.argmax(arr_space_corr[:, max_inds[1]]),
                                     np.argmax(arr_power_corr[:, max_inds[2]])])

            if any(corr_filter):
                max_corrs = max_corrs[corr_filter]
                max_inds = max_inds[corr_filter]
                max_rev_inds = max_rev_inds[corr_filter]
            else:
                bad_match_count += 1
                continue

            best_match = np.argmax(max_corrs)
            max_corr = max_corrs[best_match]

            max_ind = max_inds[best_match]
            max_rev_ind = max_rev_inds[best_match]

            if max_corr > corr_cutoff:
                if use_timecourse:
                    ref_rg = ref_vcd.main_datatable[ref_cell]['GreenTimeCourse']
                    ref_b = ref_vcd.main_datatable[ref_cell]['BlueTimeCourse']
                    test_rg = test_vcd.main_datatable[test_ids[max_ind]]['GreenTimeCourse']
                    test_b = test_vcd.main_datatable[test_ids[max_ind]]['BlueTimeCourse']
                    
                    ref_rgb = np.concatenate([ref_rg, ref_b])
                    test_rgb = np.concatenate([test_rg, test_b])
                    np.nan_to_num(ref_rgb, copy=False, nan=0, neginf=0, posinf=0)
                    np.nan_to_num(test_rgb, copy = False, nan=0, neginf=0, posinf=0)
                    
                    rgb_corr = np.corrcoef(ref_rgb, test_rgb)[0,1]

                if use_isi:
                    ref_isi = ref_vcd.get_acf_numpairs_for_cell(ref_cell)
                    match_isi = test_vcd.get_acf_numpairs_for_cell(test_ids[max_ind])
                    np.nan_to_num(ref_isi, copy=False, nan=0, neginf=0, posinf=0)
                    np.nan_to_num(match_isi, copy = False, nan=0, neginf=0, posinf=0)

                    isi_corr = np.corrcoef(ref_isi, match_isi)[0,1]

                if  isi_corr < 0.3 or rgb_corr < 0.3:
                    bad_match_count += 1
                elif ref_ids[max_rev_ind] != ref_cell:
                    bad_match_count += 1
                else:
                    match_dict[ref_cell] = test_ids[max_ind]
                    match_count += 1    

            else:
                bad_match_count += 1

        if verbose:        
            percent_good = match_count/len(ref_ids)
            percent_bad = bad_match_count/len(ref_ids)

            # print(f"\nRef clusters matched: {match_count}")
            # print(f"Ref clusters unmatched: {bad_match_count}")
            print(f"{np.round(percent_good*100, 2)}% matched, {np.round(percent_bad*100, 2)}% unmatched.\n")
            
        match_dict = dict(sorted(match_dict.items()))

        return match_dict

def get_protocol_from_datafile(exp_name: str, datafile_name: str) -> str:
    exp_summary = get_exp_summary(exp_name)
    protocol_name = exp_summary.query('datafile_name == @datafile_name').reset_index(drop = True)
    return protocol_name.loc[0,'protocol_name']

def get_classification_file_path(classification_file_name: str, exp_name: str, chunk_name: str, 
                                 ss_version: str = 'kilosort2.5') -> str:
    
    classification_file_path = os.path.join(NAS_ANALYSIS_DIR, exp_name, chunk_name, ss_version, classification_file_name)
    
    return classification_file_path

def get_ells(analysis_chunk: AnalysisChunk, d_cells_by_type: Dict[str,List[int]],
              std_scaling: float = 1.6, units: str = 'pixels') -> Tuple[Dict[str, dict], int]:
    
    if 'microns' in units.lower():
        scale_factor = analysis_chunk.microns_per_stixel
    elif 'pixels' in units.lower():
        scale_factor = analysis_chunk.pixels_per_stixel
    elif 'stixels' in units.lower():
        scale_factor = 1
    else:
        raise NameError("Units string must be 'microns', 'pixels' or 'stixels'.")
    
    rf_params = analysis_chunk.rf_params

    d_ells_by_type = dict()
    for idx, ct in enumerate(d_cells_by_type.keys()):
        d_ells_by_id = dict()
        for id in d_cells_by_type[ct]:
            d_ells_by_id[id] = Ellipse(xy=(rf_params[id]['center_x']*scale_factor,
                                    rf_params[id]['center_y']*scale_factor),
                                    width = rf_params[id]['std_x']*std_scaling*scale_factor,
                                    height = rf_params[id]['std_y']*std_scaling*scale_factor,
                                    angle = rf_params[id]['rot'],
                                    facecolor= f'C{idx}', edgecolor= f'C{idx}',
                                    alpha = 0.7)

        d_ells_by_type[ct] = d_ells_by_id
    
    return d_ells_by_type, scale_factor

def get_timecourses(analysis_chunk: AnalysisChunk, d_cells_by_type: dict) -> Dict[str, dict]: 

    d_timecourses_by_type = dict()

    for ct in d_cells_by_type.keys():
        rg_timecourses = [analysis_chunk.vcd.main_datatable[cell]['GreenTimeCourse'] for cell in d_cells_by_type[ct]]
        rg_timecourses = np.array(rg_timecourses)
        if rg_timecourses.shape[0] > 1:
            rg_mean = np.mean(rg_timecourses, axis = 0)
            rg_std = np.std(rg_timecourses, axis = 0)
        else:
            rg_mean = rg_timecourses.squeeze()
            rg_std = 0

        b_timecourses = [analysis_chunk.vcd.main_datatable[cell]['BlueTimeCourse'] for cell in d_cells_by_type[ct]]
        b_timecourses = np.array(b_timecourses)

        if b_timecourses.shape[0] > 1:
            b_mean = np.mean(b_timecourses, axis = 0)
            b_std = np.std(b_timecourses, axis = 0)
        else:
            b_mean = b_timecourses.squeeze()
            b_std = 0

        d_timecourses_by_type[ct] = {'rg_timecourses' : rg_timecourses, 'rg_mean' : rg_mean, 'rg_std' : rg_std,
                            'b_timecourses' : b_timecourses, 'b_mean' : b_mean, 'b_std' : b_std}

    return d_timecourses_by_type

def get_spike_xarr(response_block: MEAResponseBlock, protocol_ids: List[int] = None,
                   cell_types: List[str] = None) -> xr.DataArray:

    spike_time_df = response_block.df_spike_times
    num_epochs = response_block.n_epochs

    if protocol_ids is None and cell_types is None:
        filtered_df = spike_time_df
        cell_types = filtered_df['cell_type'].unique()
        
    elif protocol_ids is None:
        filtered_df = spike_time_df.query('cell_type == @cell_types')

    elif cell_types is None:
        filtered_df = spike_time_df.query('cell_id == @protocol_ids')
        cell_types = filtered_df['cell_type'].unique()
        
    else:
        filtered_df = spike_time_df.query('cell_id == @protocol_ids and cell_type == @cell_types')

    d_spike_times = dict()
    for ct in cell_types:
        df_type = filtered_df.query('cell_type == @ct').reset_index(drop = True)
        type_ids = df_type['cell_id'].values
        xarrays = np.empty((len(type_ids), num_epochs), dtype = object)
        xarrays[:,:] = np.array([df_type.loc[idx, 'spike_times'] for idx, id in enumerate(type_ids)], dtype = object)
        xarr = xr.DataArray(xarrays, dims = ['cell', 'epoch'], coords = {'cell' : type_ids,
                                                                         'epoch' : np.arange(1,num_epochs+1)})
        d_spike_times[ct] = xarr

    return d_spike_times

def get_spike_dict(response_block: MEAResponseBlock, protocol_ids: List[int] = None, 
                         cell_types: List[str] = None) -> dict:
    
    spike_time_df = response_block.df_spike_times

    if protocol_ids is None and cell_types is None:
        filtered_df = spike_time_df
        cell_types = filtered_df['cell_type'].unique()
    
    elif protocol_ids is None:
        filtered_df = spike_time_df.query('cell_type == @cell_types')

    elif cell_types is None:
        filtered_df = spike_time_df.query('cell_id == @protocol_ids')
        cell_types = filtered_df['cell_type'].unique()
        
    else:
        filtered_df = spike_time_df.query('cell_id == @protocol_ids and cell_type == @cell_types')

    d_spike_times = dict()
    for ct in cell_types:
        d_times_and_ids = dict()
        df_type = filtered_df.query('cell_type == @ct').reset_index(drop = True)
        type_ids = df_type['cell_id'].values
        arr_spike_times = [df_type.loc[idx, 'spike_times'] for idx, id in enumerate(type_ids)]
        arr_spike_times = np.array(arr_spike_times, dtype = object)
        d_times_and_ids['spike_times'] = arr_spike_times
        d_times_and_ids['cell_ids'] = type_ids

        d_spike_times[ct] = d_times_and_ids

    return d_spike_times

def classification_transfer(analysis_chunk: AnalysisChunk, target_object: Union[AnalysisChunk, MEAResponseBlock],
                                 ss_version: str = None, input_typing_file: str = None, 
                                 output_typing_file: str = 'RA_autoClassification.txt', **kwargs):

    """Transfer classification between analysis an chunk and another analysis chunk or a response block
    Inputs:
        analysis_chunk: AnalysisChunk
        target_object: AnalysisChunk or ResponseBlock 
        ss_version: str such as 'kilosort2.5', if None, uses same ss_version as analysis_chunk
        input_typing_file: str, filename of classification file to use, if None will use
                            the first typing file in analysis_chunk.typing_files
        output_typing_file: str, filename of classification file to export, default is
                            RA_autoClassification.txt

    Kwargs to pass to cluster_match:
        use_isi: bool, default = false
        use_timecourse: bool, default = false
        corr_cutoff: float, default = 0.8
        method: str, default = 'full'
        n_removed_channels: int, default = 1
        """

    if len(analysis_chunk.typing_files) == 0:
            raise FileNotFoundError("No typing files available for this analysis chunk")
        
    if target_object == analysis_chunk:
        raise Exception(f"Target chunk ({target_object.chunk_name}) cannot be the same as analysis chunk {analysis_chunk.chunk_name}")

    # If no input typing file is specified, use typing_file_0
    if input_typing_file is None:
        input_typing_file = analysis_chunk.typing_files[0] 

    # Flag if input typing file is not actually part of the current analysis chunk
    if input_typing_file not in analysis_chunk.typing_files:
        raise FileNotFoundError("Input typing file not found in current chunk")         

    # If no spike sorting version is given, use same ss_version as analysis chunk
    if ss_version is None:
        ss_version = ss_version

    # To avoid circular imports, we're only importing classes inside the utils when needed for checking. Annoying but 
    # this is just an issue with python
    from retinanalysis.classes.analysis_chunk import AnalysisChunk
    if isinstance(target_object, AnalysisChunk):
        print(f"Cluster matching {analysis_chunk.chunk_name} with {target_object.chunk_name}\n")
        destination_file_path = os.path.join(NAS_ANALYSIS_DIR, analysis_chunk.exp_name,
                                                target_object.chunk_name, ss_version, output_typing_file)
        
    else:
        print(f"Cluster matching {analysis_chunk.chunk_name} with {target_object.protocol_name}\n")
        destination_file_path = os.path.join(os.getcwd(), output_typing_file)
        if 'use_timecourse' in kwargs:
            if kwargs['use_timecourse']:
                raise FileNotFoundError("Response blocks don't have a .params file, can't use timecourse for cluster matching")

    # Cluster Match
    target_ids = target_object.cell_ids

    match_dict = cluster_match(analysis_chunk, target_object, **kwargs)
    
    # Create classification file and drop it in the destination path
    input_file_path = os.path.join(NAS_ANALYSIS_DIR, analysis_chunk.exp_name,
                                    analysis_chunk.chunk_name, analysis_chunk.ss_version,
                                    input_typing_file)
    
    matched_count = 0
    unmatched_count = 0
    input_classification_dict = create_dictionary_from_file(input_file_path, delimiter = ' ')

    with open(destination_file_path, mode='w') as output_file:
        for key in match_dict.keys():
            matched_count += 1
            print(match_dict[key], input_classification_dict[key], file = output_file)

    partial_output = create_dictionary_from_file(destination_file_path, delimiter = ' ')

    with open(destination_file_path, mode = 'a') as output_file:
        for id in target_ids:
            if id in partial_output:
                pass
            else:
                print(id, 'All/Unknown', file = output_file)
                unmatched_count += 1

    print(f"\nTarget clusters matched: {matched_count}\nTarget clusters unmatched: {unmatched_count}\n")
    print(f"Classification file {output_typing_file} created at: {destination_file_path}")

    return match_dict

def ei_corr(ref_object: Union[AnalysisChunk, MEAResponseBlock], target_object: Union[AnalysisChunk, MEAResponseBlock],
            method: str = 'full', n_removed_channels: int = 1) -> np.ndarray:


        # Pull reference eis
        ref_ids = ref_object.cell_ids
        ref_eis = [ref_object.vcd.get_ei_for_cell(cell).ei for cell in ref_ids]

        if n_removed_channels > 0:
            max_ref_vals = [np.array(np.max(ei, axis = 1)) for ei in ref_eis]
            ref_to_remove = [np.argsort(val)[-n_removed_channels:] for val in max_ref_vals]
            ref_eis = [np.delete(ei, ref_to_remove[idx], axis = 0) for idx, ei in enumerate(ref_eis)]

        # Set any EI value where the ei is less than 1.5* its standard deviation to 0
        for idx, ei in enumerate(ref_eis):
            ref_eis[idx][abs(ei) < (ei.std()*1.5)] = 0

        # For 'full' method: flatten each 512 x 201 ei array into a vector
        # and stack flattened eis into a numpy array
        if 'full' in method:
            ref_eis_flat = [ei.flatten() for ei in ref_eis]
            ref_eis = np.array(ref_eis_flat)
        # For 'time' method, take max of absolute value over time and
        # stack the resulting 512 x 1 vectors into a numpy array 
        elif 'space' in method:
            ref_eis_mean = [np.max(np.abs(ei), axis = 1) for ei in ref_eis]
            ref_eis = np.array(ref_eis_mean)
        # For 'power' method, square each 512 x 201 ei array, take the mean over time,
        # and stack the resulting 512 x 1 vectors into a numpy array
        elif 'power' in method:
            ref_eis_mean = [np.mean(ei**2, axis = 1) for ei in ref_eis]
            ref_eis = np.array(ref_eis_mean)
        else:
            raise NameError("Method poperty must be 'full', 'time', or 'power'.")


        # Pull test eis
        test_ids = target_object.cell_ids
        test_eis = [target_object.vcd.get_ei_for_cell(cell).ei for cell in test_ids]

        if n_removed_channels > 0:
            max_test_vals = [np.array(np.max(ei, axis = 1)) for ei in test_eis]
            test_to_remove = [np.argsort(val)[-n_removed_channels:] for val in max_test_vals]
            test_eis = [np.delete(ei, test_to_remove[idx], axis = 0) for idx, ei in enumerate(test_eis)]

        # Set the EI value where the EI is less than 1.5* its standard deviation to 0
        for idx, ei in enumerate(test_eis):
            test_eis[idx][abs(ei) < (ei.std()*1.5)] = 0

        # For 'full' method: flatten each 512 x 201 ei array into a vector
        # and stack flattened eis into a numpy array
        if 'full' in method:
            test_eis_flat = [ei.flatten() for ei in test_eis]
            test_eis = np.array(test_eis_flat)
        # For 'time' method, take max of absolute value over time and
        # stack the resulting 512 x 1 vectors into a numpy array 
        elif 'space' in method:
            test_eis_mean = [np.max(np.abs(ei), axis = 1) for ei in test_eis]
            test_eis = np.array(test_eis_mean)
        # For 'power' method, square each 512 x 201 ei array, take the mean over time,
        # and stack the resulting 512 x 1 vectors into a numpy array
        elif 'power' in method:
            test_eis_mean = [np.mean(ei**2, axis = 1) for ei in test_eis]
            test_eis = np.array(test_eis_mean)
        else:
            raise NameError("Method poperty must be 'full', 'space', or 'power'.")


        num_pts = ref_eis.shape[1]

        # Calculate covariance and correlation
        c = test_eis @ ref_eis.T / num_pts
        d = np.mean(test_eis, axis = 1)[:,None] * np.mean(ref_eis, axis = 1)[:,None].T
        covs = c - d

        std_calc = np.std(test_eis, axis = 1)[:,None] * np.std(ref_eis, axis = 1)[:, None].T
        corr = covs / std_calc

        # Set nan values and infinite values to 0
        np.nan_to_num(corr, copy=False, nan = 0, posinf = 0, neginf = 0)

        return corr.T

def create_dictionary_from_file(file_path, delimiter=' '):
    result_dict = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into key and value using the specified delimiter
            key, value = map(str.strip, line.split(delimiter, 1))
                        
            # Add key-value pair to the dictionary
            result_dict[int(key)] = value
    
    return result_dict

def get_presentation_times(frame_times, preFrames, flashFrames, gapFrames, images_per_epoch):
    flash_times = []
    gap_times = []

    for epoch in range(frame_times.shape[0]):
        flash_times.append([frame_times[epoch, preFrames + flashFrames*idx+gapFrames*idx] for idx in range(images_per_epoch)])
        gap_times.append([frame_times[epoch, preFrames + flashFrames*(idx+1)+gapFrames*idx] for idx in range(images_per_epoch)])

    pre_times = [frame_times[epoch,preFrames] for epoch in range(frame_times.shape[0])]
    pre_times = np.array(pre_times)
    flash_times = np.array(flash_times)
    gap_times = np.array(gap_times)

    return flash_times, gap_times, pre_times