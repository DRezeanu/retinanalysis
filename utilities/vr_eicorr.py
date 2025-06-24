import os
import numpy as np
import matplotlib.pyplot as plt
import visionloader as vl
import config as cfg
import tqdm
import pickle
from scipy.interpolate import griddata
from utilities import vr_spikeoutputs as so
from utilities import vr_spikeplots as sp

def sort_electrode_map(electrode_map: np.ndarray) -> np.ndarray:
    """
    Sort electrodes by their x, y locations.

    This uses lexsort to sort electrodes by their x, y locations
    First sort by rows, break ties by columns. 
    As each row is jittered but within row the electrodes have exact same y location.

    Parameters:
    electrode_map (numpy.ndarray): The electrode locations of shape (512, 2).

    Returns:
    numpy.ndarray: The reshaped EI matrix of shape (16, 32, 201).
    """
    sorted_indices = np.lexsort((electrode_map[:, 0], electrode_map[:, 1]))
    
    # dbug: Make scatterplot of electrode locations
    # plt.scatter(electrode_map[sorted_indices, 0], electrode_map[sorted_indices, 1], cmap='coolwarm')
    # for i in range(electrode_map.shape[0]):
    #     plt.text(electrode_map[sorted_indices[i], 0], electrode_map[sorted_indices[i], 1], str(i), fontsize=8)
    
    return sorted_indices

def reshape_ei(ei: np.ndarray, sorted_electrodes: np.ndarray) -> np.ndarray:
    """
    Reshape the EI matrix from 512 x 201 to 16 x 32 x 201 based on electrode locations.

    Parameters:
    ei (numpy.ndarray): The EI matrix of shape (512, 201).
    sorted_electrodes (numpy.ndarray): The sorted indices of the electrodes.

    Returns:
    numpy.ndarray: The reshaped EI matrix of shape (16, 32, 201).
    """
    sorted_ei = ei[sorted_electrodes]

    # Reshape the sorted EI matrix
    reshaped_ei = sorted_ei.reshape(16, 32, 201)
    
    return reshaped_ei

class MapWithinNoise(object):
    def __init__(self, vcd: vl.VisionCellDataTable):
        self.vcd = vcd
        self.ids = vcd.get_cell_ids()
        for id in self.ids:
            if 'EI' not in self.vcd.main_datatable[id].keys():
                self.ids.remove(id)
        self.ids = np.array(self.ids)
        self.n_ids = len(self.ids)
        self.d_ei = {id: vcd.get_ei_for_cell(id).ei for id in self.ids}

    def compute_ei_corr(self):
        ei_corr = np.zeros((self.n_ids, self.n_ids))
        ei_corr += np.eye(self.n_ids)
        for i in tqdm.trange(self.n_ids):
            id1 = self.ids[i]
            ei1 = self.d_ei[id1].flatten()
            for j in range(i+1, self.n_ids):
                id2 = self.ids[j]
                ei2 = self.d_ei[id2].flatten()
                ei_corr[i, j] = np.corrcoef(ei1, ei2)[0, 1]
        
        # Fill in lower triangle
        for i in range(self.n_ids):
            for j in range(i+1, self.n_ids):
                ei_corr[j, i] = ei_corr[i, j]
        self.ei_corr = ei_corr

class MapAcrossChunk(object):
    def __init__(self, vcd_src: vl.VisionCellDataTable, #str_dest_path: str,
                 vcd_dest: vl.VisionCellDataTable,
                 str_ei_corr_p: str = None):
        self.vcd_src = vcd_src
        self.vcd_dest = vcd_dest
        self.d_match_IDs = {} # This will be filled by get_match_IDs for {src: dest} cluster IDs
        # self.str_dest_path = str_dest_path

        # # Load dest vcd
        # self.vcd_dest = vl.load_vision_data(os.path.dirname(str_dest_path),
        #                             os.path.basename(str_dest_path)[:-3],
        #                             include_ei=True, include_neurons=True)
        
        self.ids_src = self.vcd_src.get_cell_ids()
        self.ids_dest = self.vcd_dest.get_cell_ids()

        # Debug mode: keep only 10 IDs
        # self.ids_src = self.ids_src[:10]
        # self.ids_dest = self.ids_dest[:10]

        # keep only IDs that have EI key in vcd.main_datatable
        ls_remove = []
        for id in self.ids_src:
            if 'EI' not in self.vcd_src.main_datatable[id].keys():
                ls_remove.append(id)
        for id in ls_remove:
            self.ids_src.remove(id)
        
        ls_remove = []
        for id in self.ids_dest:
            if 'EI' not in self.vcd_dest.main_datatable[id].keys():
                ls_remove.append(id)
        for id in ls_remove:
            self.ids_dest.remove(id)

        self.ids_src = np.array(self.ids_src)
        self.ids_dest = np.array(self.ids_dest)

        # Get source EIs
        self.d_ei_src = {} # id: self.vcd_src.get_ei_for_cell(id).ei for id in self.ids_src
        for id in self.ids_src:
            self.d_ei_src[id] = self.vcd_src.get_ei_for_cell(id).ei
        
        # Get destination EIs
        self.d_ei_dest = {}
        for id in self.ids_dest:
            self.d_ei_dest[id] = self.vcd_dest.get_ei_for_cell(id).ei

        # Get src and dest ei maps
        print('Computing EI energy maps')
        self.sorted_electrodes = sort_electrode_map(self.vcd_src.get_electrode_map())
        self.ei_maps_src = compute_ei_energy(self.sorted_electrodes, self.vcd_src, self.ids_src)
        self.ei_maps_dest = compute_ei_energy(self.sorted_electrodes, self.vcd_dest, self.ids_dest)

        # Load EI correlation if exists
        self.str_ei_corr_p = str_ei_corr_p
        if str_ei_corr_p is not None:
            if os.path.exists(str_ei_corr_p):
                print(f'Loading EI correlation from {str_ei_corr_p}')
                with open(str_ei_corr_p, 'rb') as f:
                    self.ei_corr = pickle.load(f)


    
    def compute_ei_corr(self):
        # Check if EI correlation already exists
        if hasattr(self, 'ei_corr'):
            print('EI correlation already exists')
            return
        ei_corr = np.zeros((len(self.ids_src), len(self.ids_dest)))

        print(f'Computing EI correlation for {ei_corr.shape[0]} source cells and {ei_corr.shape[1]} destination cells')
        for i in tqdm.trange(ei_corr.shape[0]):
            id_src = self.ids_src[i]
            ei_src = self.d_ei_src[id_src].flatten()
            for j in range(ei_corr.shape[1]):
                id_dest = self.ids_dest[j]
                ei_dest = self.d_ei_dest[id_dest].flatten()
                ei_corr[i, j] = np.corrcoef(ei_src, ei_dest)[0, 1]
        self.ei_corr = ei_corr
        if self.str_ei_corr_p is not None:
            with open(self.str_ei_corr_p, 'wb') as f:
                pickle.dump(ei_corr, f)

    def load_ei_corr(self, str_pkl: str):
        with open(str_pkl, 'rb') as f:
            self.ei_corr = pickle.load(f)
    
    def save_mapping(self, str_txt: str):
        # Save d_match_IDs to txt file
        # Format: source ID, destination ID
        with open(str_txt, 'w') as f:
            # Write header
            f.write('Source_ID Destination_ID\n')
            for k, v in self.d_match_IDs.items():
                f.write(f'{k} {v}\n')
        print(f'Saved mapping to {str_txt}')




def plot_ei_corr_summary(ei: MapAcrossChunk, n_thresh = 0.8):
    f, axs = plt.subplots(ncols=2, figsize=(8, 4))
    ax = axs[0]
    cm=ax.imshow(ei.ei_corr, vmax=1.0)
    plt.colorbar(cm)
    ax.set_title('EI correlation matrix')
    ax.set_xlabel('Destination IDs')
    ax.set_ylabel('Source IDs')

    ax = axs[1]
    # For each column, get the highest value
    max_corr = np.max(ei.ei_corr, axis=0)
    ax.hist(max_corr, bins=len(max_corr)//4)
    ax.axvline(n_thresh, color='r')
    n_matches = np.sum(max_corr > n_thresh)
    n_total = len(max_corr)
    ax.set_xlim(right=1.0)
    ax.set_xlabel('Best match correlation')
    ax.set_title(f'{n_matches}/{n_total} cells above threshold')

def compute_grid_ei_frame(vcd, ei_frame):
    electrode_map = vcd.get_electrode_map()
    xrange = (np.min(electrode_map[:, 0]), np.max(electrode_map[:, 0]))
    yrange = (np.min(electrode_map[:, 1]), np.max(electrode_map[:, 1]))

    y_dim = 30 # general EI map scaling
    x_dim = int((xrange[1] - xrange[0])/(yrange[1] - yrange[0]) * y_dim) # x dim is proportional to y dim

    x_e = np.linspace(xrange[0], xrange[1], x_dim)
    y_e = np.linspace(yrange[0], yrange[1], y_dim)

    grid_x, grid_y = np.meshgrid(x_e, y_e)
    grid_x = grid_x.T
    grid_y = grid_y.T
    ei_grid = griddata(electrode_map, ei_frame, (grid_x, grid_y), 
                       method='linear', fill_value=np.median(ei_frame))
    return ei_grid.T

def compute_grid_all_ei_frames(vcd, n_ID):
    ei = vcd.get_ei_for_cell(n_ID).ei
    ei_grid = []
    for i in range(ei.shape[1]):
        ei_frame = ei[:, i]
        grid_frame = compute_grid_ei_frame(vcd, ei_frame)
        ei_grid.append(grid_frame)

    ei_grid = np.array(ei_grid)
    return ei_grid

def compute_ei_energy(sorted_electrodes, vcd, ids):
    ei_maps = []
    for n_ID in ids:
        ei = vcd.get_ei_for_cell(n_ID).ei
        ei = reshape_ei(ei, sorted_electrodes)
        
        # peak_electrode = np.argmax(np.max(np.abs(ei), axis=1))
        # peak_frame = np.argmin(ei[peak_electrode, :])

        ei_energy = np.log10(np.mean(np.power(ei, 2), axis=2) + .000000001)
        # ei_energy_grid = compute_grid_ei_frame(vcd, ei_energy)
        ei_maps.append(ei_energy)
    return ei_maps

def print_ei_corr_summary(ei: MapAcrossChunk, data: so.SpikeOutputs, 
                          str_type, n_thresh = 0.8):
    ids_src = data.types.d_main_IDs[str_type]
    ids_src = np.intersect1d(ids_src, ei.ids_src)

    n_matches = 0
    n_mult_matches = 0
    for idx_t_id, t_id in enumerate(ids_src):
        idx_src_id = np.where(ei.ids_src == t_id)[0][0]
        idx_matches = np.where(ei.ei_corr[idx_src_id, :] > n_thresh)[0]
        if len(idx_matches) == 1:
            n_matches += 1
        if len(idx_matches) > 1:
            n_mult_matches += 1
    n_no_matches = len(ids_src) - n_matches - n_mult_matches
    print(f'{n_matches} matches, {n_mult_matches} multiple matches, {n_no_matches} no matches of {len(ids_src)} {str_type} cells')

def format_number(n):
    suffixes = ['', 'k', 'M', 'B', 'T']
    suffix_index = 0
    while n >= 1000 and suffix_index < len(suffixes) - 1:
        n /= 1000
        suffix_index += 1
    return f'{n:.1f}{suffixes[suffix_index]}'

        
def get_match_IDs(mapper: MapAcrossChunk, data: so.SpikeOutputs, ls_types: list,
                  n_thresh: float=0.8):
    """
    Returns:
    d_match_IDs (dict): Dictionary of matched {source ID: destination ID}.
    arr_matches (list): List of tuples (destination ID, cell type) for matched IDs.
    """
    
    d_match_IDs = {}
    arr_matches = []
    for str_type in ls_types:
        type_ids = data.types.d_main_IDs[str_type]
        type_ids = np.intersect1d(type_ids, mapper.ids_src)
        
        n_1_to_1 = 0
        n_1_to_many = 0
        n_many_to_1 = 0


        for i, n_ID in enumerate(type_ids):
            idx_src_id = np.where(mapper.ids_src == n_ID)[0][0]
            idx_matches = np.where(mapper.ei_corr[idx_src_id, :] > n_thresh)[0]
            if len(idx_matches) == 1:
                idx_match = idx_matches[0]

                # Check if there is many-to-one mapping
                idx_src_matches = np.where(mapper.ei_corr[:, idx_match] > n_thresh)[0]
                if len(idx_src_matches)==1:
                    n_ID_dest = mapper.ids_dest[idx_match]
                    d_match_IDs[n_ID] = n_ID_dest
                    arr_matches.append([n_ID_dest, f'All/{str_type}/'])
                    n_1_to_1 += 1
                else:
                    n_many_to_1 += 1
            
            elif len(idx_matches) > 1:
                n_1_to_many += 1

        print(f'{str_type}: {n_1_to_1}/{len(type_ids)} 1:1 matches found.')
        print(f'{str_type}: {n_1_to_many}/{len(type_ids)} 1:many matches found (will not use).')
        print(f'{str_type}: {n_many_to_1}/{len(type_ids)} many:1 matches found (will not use).\n')
        
        
    mapper.d_match_IDs = d_match_IDs
    
    return d_match_IDs, arr_matches


def save_and_remap_typing_data(str_new_class_txt: str, str_mapping_txt: str, 
                               mapper: MapAcrossChunk, data: so.SpikeOutputs,
                       ls_types: list, n_thresh: float=0.8):
    d_match_IDs, arr_matches = get_match_IDs(mapper, data, ls_types, n_thresh)
    np.savetxt(str_new_class_txt, arr_matches, fmt='%s', delimiter='  ')
    print(f'Saved matched typing file to {str_new_class_txt}')
    mapper.save_mapping(str_mapping_txt)

    data.reload_types(str_new_class_txt)
    data.remap_sta_vcd(d_match_IDs)
    print('Updated typing data and remapped SpikeOutputs sta data.')

def save_qc_class_txt(str_qc_src_txt: str, str_qc_dest_txt: str,
                      mapper: MapAcrossChunk,
                      data: so.SpikeOutputs, ls_types: list):
    arr_src_matches = []
    arr_dest_matches = []
    # Get the key of the matched source ID from mapper.d_match_IDs
    dest_to_src = {v: k for k, v in mapper.d_match_IDs.items()}
    for str_type in ls_types:
        dest_type_ids = data.get_type_ids(str_type)
        for dest_id in dest_type_ids:
            src_id = dest_to_src[dest_id]
            arr_src_matches.append([src_id, f'All/{str_type}/'])
            arr_dest_matches.append([dest_id, f'All/{str_type}/'])
    
    np.savetxt(str_qc_src_txt, arr_src_matches, fmt='%s', delimiter='  ')
    np.savetxt(str_qc_dest_txt, arr_dest_matches, fmt='%s', delimiter='  ')
    print(f'Saved source IDs QC typing file to {str_qc_src_txt}')
    print(f'Saved dest IDs QC typing file to {str_qc_dest_txt}')

def load_and_remap_typing_data(str_new_class_txt: str, str_mapping_txt: str, 
                               data: so.SpikeOutputs):
    # Load mapping txt
    d_match_IDs = {}
    with open(str_mapping_txt, 'r') as f:
        # Skip header
        f.readline()
        for line in f:
            src_id, dest_id = line.strip().split()
            d_match_IDs[int(src_id)] = int(dest_id)

    data.reload_types(str_new_class_txt)
    data.remap_sta_vcd(d_match_IDs)
    print('Updated typing data and remapped SpikeOutputs sta data.')

def get_top_electrodes(ei_map, n_ID, vcd, n_interval=2, n_markers=5):
    # Reshape EI timeseries
    ei = vcd.get_ei_for_cell(n_ID).ei
    sorted_electrodes = sort_electrode_map(vcd.get_electrode_map())
    ei = reshape_ei(ei, sorted_electrodes)

    ## Label top n_markers pixels spaced by n_interval in the heatmap
    # Sorted index of pixels
    ei_map_sidx = np.argsort(ei_map.flatten())[::-1]
    top_idx = ei_map_sidx[::n_interval][:n_markers]

    # Sort top_idx by argmin of EI time series
    amin_ei_ts = np.zeros(n_markers)
    for i in range(n_markers):
        y, x = np.unravel_index(top_idx[i], ei_map.shape)
        # ei_ts = ei_grid[:, y, x]
        ei_ts = ei[y, x, :]
        amin_ei_ts[i] = np.argmin(ei_ts)
    top_idx = top_idx[np.argsort(amin_ei_ts)]

    return top_idx


def plot_ei_map(ei_map, n_ID, vcd, top_idx, axs=None, label=None):
    # top_idx is the top n_markers pixels to plot, returned by get_top_electrodes
    n_markers = len(top_idx)
    if axs is None:
        f, axs = plt.subplots(nrows=n_markers+1, figsize=(5, 10),
                              gridspec_kw={'height_ratios': [1]+[1/n_markers]*n_markers})

    sample_rate = 20000.0 # Hz
    sts = vcd.get_spike_times_for_cell(n_ID)
    num_sps = len(sts)
    max_st = sts.max()/sample_rate
    avg_rate = num_sps/max_st

    ax0 = axs[0]
    ax0.imshow(ei_map, cmap='hot')
    # Get index of peak
    peak_idx = np.unravel_index(np.argmax(ei_map), ei_map.shape)
    ax0.plot(peak_idx[1], peak_idx[0], 'o', color='blue')
    ax0.axhline(peak_idx[0], color='blue')
    ax0.axvline(peak_idx[1], color='blue')

    # Interpolate every EI timeframe
    # ei_grid = compute_grid_all_ei_frames(vcd, n_ID)

    # Reshape EI timeseries
    ei = vcd.get_ei_for_cell(n_ID).ei
    sorted_electrodes = sort_electrode_map(vcd.get_electrode_map())
    ei = reshape_ei(ei, sorted_electrodes)

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
        ax.set_ylabel(str(i), rotation=0)
    
    # Set xticks for last ax
    ax.set_xticks(np.arange(0, len(ei_ts), 50))
    ax.set_xlabel('Timeframe')

    str_title = ''
    if label is not None:
        str_title += f'{label} '
    str_title += f'ID {n_ID}\nPeak: {peak_idx}\n{num_sps} sps ({avg_rate:.1f} Hz)\n'
    ax0.set_title(str_title)
    return ax0

def plot_ei_analysis(mapper: MapAcrossChunk, data: so.SpikeOutputs, 
                     idx_t_id, str_type, 
                     n_ei_maps, p_ei_maps, n_thresh = 0.8,
                     str_savedir=None):
    n_id = data.types.d_main_IDs[str_type][idx_t_id]
    n_id = int(n_id)
    n_ids = [n_id] # For many-to-one matches

    all_src_ids = np.array(mapper.vcd_src.get_cell_ids())
    idx_n_id = np.where(all_src_ids == n_id)[0][0]
    ls_n_eis = [n_ei_maps[idx_n_id]]
    
    # Find matches above threshold
    idx_matches = np.where(mapper.ei_corr[idx_n_id, :] > n_thresh)[0]
    
    # Define category of match.
    n_matches = len(idx_matches)
    if n_matches == 0:
        CATEGORY = '1_no_match'
    elif n_matches > 1:
        CATEGORY = '2_one_to_many_matches'
    elif n_matches == 1:
        # Check for multiple src matches
        idx_match = idx_matches[0]

        # Check if there is many-to-one mapping
        idx_src_matches = np.where(mapper.ei_corr[:, idx_match] > n_thresh)[0]
        if len(idx_src_matches)>1:
            CATEGORY = '3_many_to_one_match'
            ls_n_eis = [n_ei_maps[i] for i in idx_src_matches]
            n_ids = [all_src_ids[i] for i in idx_src_matches]
        else:
            CATEGORY = '4_one_to_one_match'
        

    # If no matches above threshold, get the highest correlation
    if n_matches == 0:
        idx_matches = [np.argsort(mapper.ei_corr[idx_n_id, :])[-1]]
    
    prot_ids = np.array(mapper.vcd_dest.get_cell_ids())
    ls_p_eis = [p_ei_maps[i] for i in idx_matches]

    ncols = len(ls_p_eis)+len(ls_n_eis)
    n_ei_markers = 5
    f, axs = plt.subplots(ncols=ncols, nrows=2+n_ei_markers, figsize=(ncols*5, 12),
                          gridspec_kw={'height_ratios': [1] + [1/n_ei_markers]*n_ei_markers + [1]},
                          layout='constrained')

    # Get top electrodes for first noise EI map
    top_idx = get_top_electrodes(ls_n_eis[0], n_ids[0], mapper.vcd_src, 
                                 n_interval=2, n_markers=n_ei_markers)
    
    
    # Plot noise and protocol EI maps
    for idx, ei in enumerate(ls_n_eis):
        ax = axs[:6,idx]
        n_ei = ls_n_eis[idx]
        n_id = n_ids[idx]
        tmp_type = data.types.d_types[n_id]
        plot_ei_map(n_ei, n_id, mapper.vcd_src, top_idx=top_idx, axs=ax, label=f'Noise {tmp_type}')
    for idx, ei in enumerate(ls_p_eis):
        ax = axs[:6,idx+len(ls_n_eis)]
        idx_p_id = idx_matches[idx]
        p_id = prot_ids[idx_p_id]
        plot_ei_map(ei, p_id, mapper.vcd_dest, top_idx=top_idx, axs=ax, label='Prot')

    # Plot cell type RFs
    ax = axs[-1,0]
    ax,ells=sp.plot_rfs(data, data.types.d_main_IDs[str_type], ax=ax, sd_mult=0.8, ell_color='k')
    ax.invert_yaxis()
    # Set idx_t_id ell color to red
    ells[idx_t_id].set_facecolor('red')
    ax.set_title(f'{str_type} RFs')
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot EI correlation distribution for source ID
    ax = axs[-1,1]
    sorted_idx = np.argsort(mapper.ei_corr[idx_n_id, :])[::-1]
    ax.plot(mapper.ei_corr[idx_n_id, sorted_idx], '-o')
    ax.set_ylim(-0.5, 1)
    ax.axhline(n_thresh, color='red')
    ax.set_title(f'EI correlations with Noise ID {n_id}')
    ax.set_xlabel('Protocol clusters')

    # Set axis off for those not used
    for i in range(2, ncols):
        ax = axs[-1, i]
        ax.axis('off')

    if str_savedir is not None:
        str_savedir = os.path.join(str_savedir, CATEGORY)
        if not os.path.exists(str_savedir):
            os.makedirs(str_savedir)
        str_match_ids= '_'.join([str(prot_ids[i]) for i in idx_matches])
        str_save_img = os.path.join(str_savedir, f'{str_type}_{n_id}_matching_{str_match_ids}.png')
        plt.savefig(str_save_img, bbox_inches='tight')
        plt.close()

def plot_ei_analysis_all(eic: MapAcrossChunk, data: so.SpikeOutputs, n_thresh: float=0.8,
                         str_savedir: str=None, ls_types: list=None):
    # TODO highlight near threshold misses
    if ls_types is None:
        ls_types = data.ls_RGC_labels

    if str_savedir is None:
        str_savedir = f'./{data.str_experiment}_ei_matches'
    
    if not os.path.exists(str_savedir):
        os.makedirs(str_savedir)
    
    for str_type in ls_types:
        print(f'Processing {str_type}')
        type_ids = data.types.d_main_IDs[str_type]
        str_typedir = os.path.join(str_savedir, str_type)
        if not os.path.exists(str_typedir):
            os.makedirs(str_typedir)
        for idx_t_id in tqdm.trange(len(type_ids)):
            plot_ei_analysis(eic, data, idx_t_id, str_type, 
                             eic.ei_maps_src, eic.ei_maps_dest,
                              n_thresh=n_thresh, str_savedir=str_typedir)