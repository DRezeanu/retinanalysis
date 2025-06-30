from utils.settings import mea_config
import os
import numpy as np
import utils.datajoint_utils as dju
import visionloader as vl

NAS_DATA_DIR = mea_config['data']
NAS_ANALYSIS_DIR = mea_config['analysis']
META_DIR = mea_config['meta']
TAGS_DIR = mea_config['tags']
H5_DIR = mea_config['h5']
USER = mea_config['user']

def get_classification_file_path(classification_file_name: str, exp_name: str, chunk_name: str, 
                                 nas_analysis_dir: str = NAS_ANALYSIS_DIR, ks_version: str = 'kilosort2.5'):
    
    analysis_path = get_analysis_path(exp_name, chunk_name, nas_analysis_dir)
    classification_file_path = os.path.join(analysis_path, ks_version,
                                 classification_file_name)
    
    return classification_file_path

def get_analysis_path(exp_name: str, chunk_name: str, nas_analysis_dir: str = NAS_ANALYSIS_DIR) -> str:
    """
    Constructs the full path to the analysis directory for a specific sorting chunk.
    """

    analysis_path = os.path.join(nas_analysis_dir, exp_name, chunk_name)
    
    return analysis_path


def get_data_path(exp_name: str, target_protocol: str, protocol_index: int = 0, nas_data_dir: str = NAS_DATA_DIR) -> str:
    """
    Constructs the full path to the data directory for a specific experiment protocol.
    """

    target_protocol_options: np.ndarray = dju.search_protocol(target_protocol)
    try:
        target_protocol = target_protocol_options[0]
    except:
        pass

    experiment_summary = dju.get_mea_exp_summary(exp_name)

    protocol_df = experiment_summary[(experiment_summary['protocol_name'] == target_protocol)].iloc[protocol_index]

    data_val = protocol_df['data_dir']
    
    data_path = os.path.join(nas_data_dir, data_val)

    return data_path

def get_vcd(exp_name: str, data_name: str, ks_version: str = 'kilosort2.5',
            data_index: int = 0, ei: bool = True, sta: bool = False, params: bool = True,
            runtimemovie_params: bool = True, neurons: bool = False, noise: bool = False,
            nas_analysis_dir: str = NAS_ANALYSIS_DIR, nas_data_dir: str = NAS_DATA_DIR) -> vl.VisionCellDataTable:
    """
    Retrieves a VisionCellDataTable (VCD) object using the visionloader module.
    """
    
    if 'chunk' in data_name:
        # we're pulling analysis data by chunk name
        data_path = get_analysis_path(exp_name, data_name, nas_analysis_dir)
        data_path = os.path.join(data_path, ks_version)

        vcd = []
        vcd = vl.load_vision_data(data_path, ks_version, include_ei = ei,
                                  include_noise = noise, include_sta = sta,
                                  include_params = params, include_runtimemovie_params = runtimemovie_params,
                                  include_neurons = neurons)
        
    else:
        # we're pulling data for a protocol
        data_path = get_data_path(exp_name, data_name, data_index, nas_data_dir)
        data_val = os.path.basename(data_path)
        data_path = os.path.join(data_path, ks_version)
        

        vcd = []
        vcd = vl.load_vision_data(data_path, data_val, include_ei = ei,
                                  include_noise = noise, include_sta = sta,
                                  include_params = False, include_runtimemovie_params = False,
                                  include_neurons = neurons)
        
    return vcd


def find_matches(exp_name: str, ref_data: str, match_data: str, corr_cutoff: float = 0.7,
                 ks_version: str = 'kilosort2.5', ref_index: int = 0,  match_index: int = 0,
                 use_ei: bool = True, use_sta: bool = False, use_params: bool = False,
                 use_neurons: bool = False, use_noise = False, use_isi = False, nas_analysis_dir: str = NAS_ANALYSIS_DIR,
                 nas_data_dir: str = NAS_DATA_DIR, classification_file: str = "dragos_autoClassification.txt"):
    """
    Finds matching cells between two sorting chunks (reference and match) based on electrical images (EIs)
    and optionally other features.

    This function uses Vision Cell Data Table (VCD) objects to load and compare cell data
    from two different sources: a reference dataset and a dataset to find matches in.
    It calculates the cross-correlation between EIs to identify potential matches and
    applies additional criteria (if specified) to filter out bad matches.
    """

    print("\nLoading reference data...")
    ref_vcd = get_vcd(exp_name, ref_data, ks_version, ref_index,
                    use_ei, use_sta, use_params,
                    neurons = use_neurons, noise = use_noise, 
                    nas_analysis_dir = nas_analysis_dir, nas_data_dir = nas_data_dir)
    
    print("\nLoading matching data...")
    test_vcd = get_vcd(exp_name, match_data, ks_version,
                        match_index, use_ei, use_sta, use_params,
                        neurons = use_neurons, noise = use_noise,
                        nas_analysis_dir = nas_analysis_dir, nas_data_dir = nas_data_dir)


    ref_ids = ref_vcd.get_cell_ids()

    test_ids = test_vcd.get_cell_ids()

    print(f"\nNumber of cells found in {ref_data}: {len(ref_ids)}")
    print(f"Number of cells found in {match_data}: {len(test_ids)}")

    cell_idx = dict()
    match_count = 0
    bad_match_count = 0
    sta_corr = 1
    isi_corr = 1

    corr = ei_corr(ref_vcd, test_vcd)

    for idx, ref_cell in enumerate(ref_ids):
        sorted_corr = np.sort(corr[idx,:])
        sorted_corr = np.flip(sorted_corr)

        max_corr = sorted_corr[0]
        next_max_corr = sorted_corr[1]
        max_ind = np.argmax(corr[idx,:])
        max_rev_ind = np.argmax(corr[:,max_ind])

        if max_corr > corr_cutoff:
            if (use_sta):
                ref_sta = ref_vcd.get_sta_for_cell(ref_cell)
                ref_sta.red[abs(ref_sta.red) < (ref_sta.red.std()*1.5)] = 0
                startInt = ref_sta.red[1, 1, :].size - 15
                endInt = startInt + 10                                      
                match_sta = test_vcd.get_sta_for_cell(test_ids[max_ind])            
                match_sta.red[abs(match_sta.red) < (match_sta.red.std()*1.5)] = 0
                sta_corr = (np.corrcoef(ref_sta.red[:,:,startInt:endInt].flatten(), match_sta.red[:,:,startInt:endInt].flatten())[0,1] +
                    np.corrcoef(ref_sta.blue[:,:,startInt:endInt].flatten(), match_sta.blue[:,:,startInt:endInt].flatten())[0,1] )/2.0
                
            if (use_isi):
                ref_isi = ref_vcd.get_acf_numpairs_for_cell(ref_cell)
                match_isi = test_vcd.get_acf_numpairs_for_cell(test_ids[max_ind])
                np.nan_to_num(ref_isi, copy=False, nan=0, neginf=0, posinf=0)
                np.nan_to_num(match_isi, copy = False, nan=0, neginf=0, posinf=0)
                isi_corr = np.corrcoef(ref_isi, match_isi)[0,1]

            if next_max_corr > (max_corr*0.90) or sta_corr < 0 or isi_corr < 0.2:
                bad_match_count += 1
            elif ref_ids[max_rev_ind] != ref_cell:
                bad_match_count += 1
                # print(f"{ref_cell} is not the best match for {test_ids[max_ind]}.")
            else:
                cell_idx[ref_cell] = test_ids[max_ind]
                match_count += 1    

        else:
            bad_match_count += 1
    
    percent_good = match_count/len(ref_ids)
    percent_bad = bad_match_count/len(ref_ids)

    print(f"\nRef clusters matched: {match_count}")
    print(f"Ref clusters unmatched: {bad_match_count}")
    print(f"{np.round(percent_good*100, 2)}% matched, {np.round(percent_bad*100, 2)}% unmatched.")
    
    cell_idx_sorted = dict(sorted(cell_idx.items()))

    return cell_idx_sorted, ref_ids, test_ids

def auto_classification(exp_name: str, ref_data: str, match_data: str, classification_file: str,
                            corr_cutoff: float = 0.8, ks_version: str = 'kilosort2.5', ref_index: int = 0,
                            match_index: int = 0, use_ei: bool = True, use_sta: bool = False, use_params: bool = False,
                            use_neurons: bool = False, use_noise: bool = False, use_isi: bool = False,
                            output_filename: str = 'dragos_autoClassification.txt', nas_analysis_dir: str = NAS_ANALYSIS_DIR,
                            nas_data_dir: str = NAS_DATA_DIR):
    """
    Finds matching cells between two sorting chunks (reference and match) based on electrical images (EIs)
    and creates a classification file using a given noise classification chunk.
    """
        
    match_dict, ref_ids, test_ids = find_matches(exp_name, ref_data, match_data, corr_cutoff, ks_version,
                              ref_index, match_index, use_ei, use_sta, use_params,
                              use_neurons, use_noise, use_isi, nas_analysis_dir, nas_data_dir)
    
    classification_file_path = get_classification_file_path(classification_file, exp_name, ref_data, nas_analysis_dir, ks_version)
    print(f"\nUsing {classification_file} classification file for {ref_data}")

    matched_count = 0
    unmatched_count = 0
    arr_types = create_dictionary_from_file(classification_file_path, delimiter = ' ')

    with open(output_filename, mode='w') as output_file:
        for key in match_dict.keys():
            matched_count += 1
            print(match_dict[key], arr_types[key], file = output_file)

    partial_output = create_dictionary_from_file(output_filename, delimiter = ' ')

    with open(output_filename, mode = 'a') as output_file:
        for test_id in test_ids:
            if test_id in partial_output:
                pass
            else:
                print(test_id, 'All/unmatched', file = output_file)
                unmatched_count += 1

    percent_good = matched_count/len(test_ids)
    percent_bad = unmatched_count/len(test_ids)

    print(f"\nTarget clusters matched: {matched_count}\nTarget clusters unmatched: {unmatched_count}")
    print(f"{np.round(percent_good*100, 2)}% matched, {np.round(percent_bad*100, 2)}% unmatched.\n")

    match_dict_sorted = dict(sorted(match_dict.items()))

    return match_dict_sorted

def ei_corr(ref_vcd: vl.VisionCellDataTable, target_vcd: vl.VisionCellDataTable, method: str = 'full') -> np.ndarray:

        # Pull reference eis
        ref_ids = ref_vcd.get_cell_ids()
        ref_eis = [ref_vcd.get_ei_for_cell(cell).ei for cell in ref_ids]

        # Set any EI value where the ei is less than 1.5* its standard deviation to 0
        for idx, ei in enumerate(ref_eis):
            ref_eis[idx][abs(ei) < (ei.std()*1.5)] = 0

        # Flatten 512 x 201 array into a vector
        ref_eis_flat = [ei.flatten() for ei in ref_eis]
        ref_eis = np.array(ref_eis_flat)

        # Pull test eis
        test_ids = target_vcd.get_cell_ids()
        test_eis = [target_vcd.get_ei_for_cell(cell).ei for cell in test_ids]

        # Set the EI value where the EI is less than 1.5* its standard deviation to 0
        for idx, ei in enumerate(test_eis):
            test_eis[idx][abs(ei) < (ei.std()*1.5)] = 0

        # Flatten all the eis and turn them into numpy array
        test_eis_flat = [ei.flatten() for ei in test_eis]
        test_eis = np.array(test_eis_flat)

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
    


def find_nearest_noise(exp_name: str, target_protocol: str, protocol_index: int = 0) -> str:
    
    experiment_summary = dju.get_mea_exp_summary(exp_name)
    noise_protocol = dju.search_protocol('.SpatialNoise')[0]
    target_protocol_name = dju.search_protocol(target_protocol)[0]
    
    noise_condition_1 = experiment_summary['protocol_name'] == noise_protocol
    noise_condition_2 = experiment_summary['chunk_name'].str.contains('chunk')
    noise_runs = experiment_summary[noise_condition_1 & noise_condition_2]

    target_condition_1 = experiment_summary['protocol_name'] == target_protocol_name
    target_run = experiment_summary[target_condition_1].iloc[protocol_index]

    target_run_start = target_run['minutes_since_start']-target_run['duration_minutes']
    target_run_stop = target_run['minutes_since_start']

    noise_run_start = noise_runs['minutes_since_start']-noise_runs['duration_minutes']
    noise_run_stop = noise_runs['minutes_since_start']

    protocolstop_to_noisestart = abs(noise_run_start - target_run_stop)
    protocolstart_to_noisestop = abs(noise_run_stop - target_run_start)

    minimum_distance = np.minimum(protocolstart_to_noisestop, protocolstop_to_noisestart)

    closest_noise_chunk = noise_runs[(protocolstart_to_noisestop == min(minimum_distance))]
    if closest_noise_chunk.empty:
        closest_noise_chunk = noise_runs[(protocolstop_to_noisestart == min(minimum_distance))]

    return closest_noise_chunk.reset_index().at[0,'chunk_name']

def get_presentation_times(frame_times, preFrames, flashFrames, gapFrames, images_per_epoch):
    flash_times = []
    gap_times = []

    for epoch in range(frame_times.shape[0]):
        flash_times.append([frame_times[epoch, preFrames + flashFrames*idx+gapFrames*idx] for idx in range(images_per_epoch)])
        gap_times.append([frame_times[epoch, preFrames + flashFrames*(idx+1)+gapFrames*idx] for idx in range(images_per_epoch)])

    flash_times = np.array(flash_times)
    gap_times = np.array(gap_times)

    return flash_times, gap_times