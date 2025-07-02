from retinanalysis.settings import mea_config
import os
import numpy as np
import retinanalysis.datajoint_utils as dju
import visionloader as vl


NAS_DATA_DIR = mea_config['data'] 
NAS_ANALYSIS_DIR = mea_config['analysis']
META_DIR = mea_config['meta']
TAGS_DIR = mea_config['tags']
H5_DIR = mea_config['h5']
USER = mea_config['user']

def get_analysis_vcd(exp_name, chunk_name, ss_version):
        data_path = os.path.join(NAS_ANALYSIS_DIR, exp_name, chunk_name, ss_version)
        print(f'Loading VCD from {data_path} ...')
        vcd = vl.load_vision_data(data_path, ss_version, include_ei = True,
                                  include_noise = False, include_sta = False,
                                  include_params = True, include_runtimemovie_params = True,
                                  include_neurons = True)
        
        return vcd

def get_protocol_vcd(exp_name, datafile_name, ss_version):
        data_path = os.path.join(NAS_DATA_DIR, exp_name, datafile_name, ss_version)
        print(f'Loading VCD from {data_path} ...')
        vcd = vl.load_vision_data(
            data_path, datafile_name, 
            include_ei = True, include_neurons = True
            )
        
        return vcd


def cluster_match(ref_vcd: vl.VisionCellDataTable, test_vcd: vl.VisionCellDataTable,
                corr_cutoff: float = 0.8, method: str = 'all', use_isi: bool = False,
                use_timecourse: bool = False):
        
        if 'all' in method:
            arr_full_corr: np.ndarray = ei_corr(ref_vcd, test_vcd, method = 'full')
            arr_space_corr: np.ndarray = ei_corr(ref_vcd, test_vcd, method = 'space')
            arr_power_corr: np.ndarray = ei_corr(ref_vcd, test_vcd, method = 'power')
        elif 'full' in method:
            arr_full_corr: np.ndarray = ei_corr(ref_vcd, test_vcd, method = method)
            arr_space_corr: np.ndarray = np.zeros(arr_full_corr.shape)
            arr_power_corr: np.ndarray = np.zeros(arr_full_corr.shape)
        elif 'space' in method:
            arr_space_corr: np.ndarray = ei_corr(ref_vcd, test_vcd, method = method)
            arr_full_corr: np.ndarray = np.zeros(arr_space_corr.shape)
            arr_power_corr: np.ndarray = np.zeros(arr_space_corr.shape)
        elif 'power' in method:
            arr_power_corr: np.ndarray = ei_corr(ref_vcd, test_vcd, method = method)
            arr_space_corr: np.ndarray = np.zeros(arr_power_corr.shape)
            arr_full_corr: np.ndarray = np.zeros(arr_power_corr.shape)
        else:
            raise NameError("Method property must be 'all', 'full', 'space', or 'power'")

        match_dict = dict()
        match_count = 0
        bad_match_count = 0
        isi_corr = 1
        rgb_corr = 1

        ref_ids = ref_vcd.get_cell_ids()
        test_ids = test_vcd.get_cell_ids()

        for idx, ref_cell in enumerate(ref_ids):
            sorted_full_corr = np.sort(arr_full_corr[idx,:])
            sorted_full_corr = np.flip(sorted_full_corr)

            sorted_space_corr = np.sort(arr_space_corr[idx,:])
            sorted_space_corr = np.flip(sorted_space_corr)

            sorted_power_corr = np.sort(arr_power_corr[idx,:])
            sorted_power_corr = np.flip(sorted_power_corr)

            max_corrs = np.array([sorted_full_corr[0], sorted_space_corr[0], sorted_power_corr[0]])
            next_max_corrs = np.array([sorted_full_corr[1], sorted_space_corr[1], sorted_power_corr[1]])

            max_inds = np.array([np.argmax(arr_full_corr[idx,:]),
                                 np.argmax(arr_space_corr[idx,:]),
                                 np.argmax(arr_power_corr[idx,:])])

            max_rev_inds = np.array([np.argmax(arr_full_corr[:, max_inds[0]]),
                                     np.argmax(arr_space_corr[:, max_inds[1]]),
                                     np.argmax(arr_power_corr[:, max_inds[2]])])

            best_match = np.argmax(max_corrs)
            max_corr = max_corrs[best_match]

            next_max_corr = next_max_corrs[best_match]

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

                if next_max_corr > (max_corr*0.90) or isi_corr < 0.3 or rgb_corr < 0.3:
                    bad_match_count += 1
                elif ref_ids[max_rev_ind] != ref_cell:
                    bad_match_count += 1
                else:
                    match_dict[ref_cell] = test_ids[max_ind]
                    match_count += 1    

            else:
                bad_match_count += 1
        
        percent_good = match_count/len(ref_ids)
        percent_bad = bad_match_count/len(ref_ids)

        print(f"\nRef clusters matched: {match_count}")
        print(f"Ref clusters unmatched: {bad_match_count}")
        print(f"{np.round(percent_good*100, 2)}% matched, {np.round(percent_bad*100, 2)}% unmatched.")
        
        match_dict = dict(sorted(match_dict.items()))

        return match_dict

def get_protocol_from_datafile(exp_name, datafile_name):
    exp_summary = dju.get_mea_exp_summary(exp_name)
    protocol_name = exp_summary.query('datafile_name == @datafile_name').reset_index(drop = True)
    return protocol_name.loc[0,'protocol_name']

def get_classification_file_path(classification_file_name: str, exp_name: str, chunk_name: str, 
                                 ss_version: str = 'kilosort2.5'):
    
    classification_file_path = os.path.join(NAS_ANALYSIS_DIR, exp_name, chunk_name, ss_version, classification_file_name)
    
    return classification_file_path


def ei_corr(ref_vcd: vl.VisionCellDataTable, target_vcd: vl.VisionCellDataTable,
            method: str = 'full', n_removed_channels: int = 0) -> np.ndarray:


        # Pull reference eis
        ref_ids = ref_vcd.get_cell_ids()
        ref_eis = [ref_vcd.get_ei_for_cell(cell).ei for cell in ref_ids]

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
        test_ids = target_vcd.get_cell_ids()
        test_eis = [target_vcd.get_ei_for_cell(cell).ei for cell in test_ids]

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

    flash_times = np.array(flash_times)
    gap_times = np.array(gap_times)

    return flash_times, gap_times