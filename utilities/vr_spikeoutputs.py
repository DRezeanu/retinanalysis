"""
General purpose class for:
Parsing spike outputs of a given protocol
Parsing receptive field parameters in .params and {algo}_params.mat
"""

import numpy as np
import os
import visionloader as vl
import hdf5storage
import pickle
from utilities import vr_celltype_io as ctio
import symphony_data as sd
import matplotlib.pyplot as plt
import config as cfg
from scipy.io import loadmat, savemat

def dict_list_to_array(d):
    # Convert dictionary of lists to dictionary of arrays
    d_array = {}
    for key in d.keys():
        d_array[key] = np.array(d[key])
    return d_array

class SpikeOutputs(object):
    def __init__(self, str_experiment, str_protocol=None, str_datafile=None, str_algo=None,
                 paramsfile=None, dataset_name=None, paramsmatfile=None,
                 str_classification=None, ls_RGC_labels=['OffP', 'OffM', 'OnP', 'OnM', 'SBC'],
                 ls_filenames=None, str_noise_protocol='manookinlab.protocols.SpatialNoise',
                 ls_noise_filenames=None):
        """
        How input params are used:
        
        paramsfile         : Path to .params file. Used by load_sta_from_params
            eg-'/path/to/kilosort2.params'
        dataset_name       : Name of the vision data files, typically same as algo. Used by load_sta_from_params
            eg-'kilosort2'
        paramsmatfile      : Path to params.mat file. Optional use by load_sta_from_params for spatial maps and convex hulls
            eg-'/path/to/kilosort2_params.mat'
        str_classification : Path to classification .txt file. Used to init ctio.CellTypes
            eg-'/path/to/kilosort2.classification.txt'
        ls_RGC_labels      : List of RGC labels. Used by ctio.CellTypes to parse the .txt file
            eg-['OffP', 'OffM', 'OnP', 'OnM', 'SBC']
        ls_filenames       : List of data file names used by load_psth and load_isi. 
            eg-['data001', 'data002']
        ls_noise_filenames : List of noise file names used by load_isi. 
            eg-['data001', 'data002']
        """
        self.str_experiment = str_experiment
        self.str_protocol = str_protocol
        self.str_datafile = str_datafile
        self.str_algo = str_algo
        self.str_noise_protocol = str_noise_protocol # TODO: better method for setting this from metadata
        if int(str_experiment[:8]) < 20230926:
            self.str_noise_protocol = 'manookinlab.protocols.FastNoise'

        self.paramsfile = paramsfile
        if dataset_name is None:
            dataset_name = str_algo
        self.dataset_name = dataset_name
        self.paramsmatfile = paramsmatfile
        self.ls_filenames = ls_filenames
        self.ls_noise_filenames = ls_noise_filenames
        self.ls_RGC_labels = ls_RGC_labels

        self.stim = {}
        self.spikes = {}
        self.isi = {}

        self.ARR_CELL_IDS = np.array([], dtype=int)

        # Load classifications if provided
        if str_classification is not None:
            self.str_classification = str_classification
            self.types = ctio.CellTypes(str_classification, ls_RGC_labels=ls_RGC_labels)
            self.ls_RGC_labels = list(self.types.d_main_IDs.keys())

        # Load datafile if provided
        if str_datafile is not None:
            ls_data_keys = ['spike_dict', 'isi', 'cluster_id', 'acf',
                            'isi_bin_edges']
            if '.mat' in str_datafile:
                self.stim = hdf5storage.loadmat(str_datafile)
            elif '.p' in str_datafile:
                with open(str_datafile, 'rb') as f:
                    self.stim = pickle.load(f)
                    self.spikes = {}

                    # Populate data dict with keys from ls_data_keys if present
                    for key in ls_data_keys:
                        if key in self.stim.keys():
                            self.spikes[key] = self.stim[key]
                            
                            # Remove from stim
                            self.stim.pop(key)

            else:
                raise ValueError('Data file must be .mat or .p')
            ids = np.array(self.spikes['cluster_id']).flatten().astype(int)
            self.ARR_CELL_IDS = np.union1d(ids, self.ARR_CELL_IDS)
            self.GOOD_CELL_IDS = self.ARR_CELL_IDS.copy()
            self.N_CELLS = len(self.ARR_CELL_IDS)
    
    def get_type_ids(self, str_type):
        return self.types.d_main_IDs[str_type]
    
    def load_sta_from_params(self, paramsfile: str=None, dataset_name: str=None, paramsmatfile: str=None,
                             isi_bin_edges=None, load_ei=False, load_neurons=True, load_sta=False,
                             b_flip_y=True):
        # b_flip_y is for flipping y location of RFs, to get in matrix space (0,0) at top left.
        if not paramsfile:
            paramsfile = self.paramsfile
            dataset_name = self.dataset_name
            paramsmatfile = self.paramsmatfile
        # Check that globals file exists
        str_globals = os.path.join(os.path.dirname(paramsfile), f'{dataset_name}.globals')
        if not os.path.exists(str_globals):
            raise ValueError(f'{str_globals} does not exist.')
        
        print(f'Loading STA RF fits from {paramsfile}...')
        self.vcd = vl.load_vision_data(analysis_path=os.path.dirname(paramsfile), dataset_name=dataset_name, 
                                       include_params=True, include_runtimemovie_params=True, include_ei=load_ei,
                                       include_neurons=load_neurons, include_sta=load_sta)

        self.N_WIDTH = self.vcd.runtimemovie_params.width
        self.N_HEIGHT = self.vcd.runtimemovie_params.height
    
        # This is used to translate noise stixel space to microns.
        self.NOISE_GRID_SIZE = self.vcd.runtimemovie_params.micronsPerStixelX # Typically 30 microns. 


        # Load RF fit parameters from .params file
        # Get only IDs of cells that have RF fits
        d_sta = {}
        for n_id in self.vcd.main_datatable.keys():
            if 'x0' in self.vcd.main_datatable[n_id].keys():
                d_sta[n_id] = self.vcd.main_datatable[n_id]
        self.d_sta = d_sta
        sta_cell_ids = list(self.d_sta.keys())
        print(f'Loaded STA RF fits for {len(sta_cell_ids)} cells.')

        if b_flip_y:
            for n_ID in self.d_sta.keys():
                self.d_sta[n_ID]['y0'] = self.N_HEIGHT-self.d_sta[n_ID]['y0']
            print('Flipped y0 values, so RFs are in sta matrix space with (0,0) in top left.')

        # Load _params.mat. 
        if paramsmatfile:
            print(f'Loading STA params from {paramsmatfile}...')
            self.d_params = hdf5storage.loadmat(paramsmatfile)
                
            # Get spatial maps of cells
            self.d_sta_spatial = {}
            for idx_ID, n_ID in enumerate(sta_cell_ids):
                # Load red channel spatial map. Cell ID index in vcd should be same as in _params.mat
                self.d_sta_spatial[n_ID] = self.d_params['spatial_maps'][idx_ID, :, :, 0]
            
            # Get convex hull fits of cells
            self.d_sta_convex_hull = {}
            for idx_ID, n_ID in enumerate(sta_cell_ids):
                self.d_sta_convex_hull[n_ID] = self.d_params['hull_vertices'][idx_ID, :,:]

            print(f'Loaded STA params for {len(self.d_sta_spatial.keys())} cells.')

            if b_flip_y:
                for n_ID in self.d_sta_spatial.keys():
                    self.d_sta_spatial[n_ID] = self.d_sta_spatial[n_ID][::-1, :]
                    self.d_sta_convex_hull[n_ID] = self.d_sta_convex_hull[n_ID][::-1, :]
                
        ids = np.array(sta_cell_ids).astype(int)
        self.ARR_CELL_IDS = np.union1d(ids, self.ARR_CELL_IDS)
        self.GOOD_CELL_IDS = self.ARR_CELL_IDS.copy()
        self.N_CELLS = len(self.ARR_CELL_IDS)
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)

        # Load STA ISI
        if isi_bin_edges is not None:
            print(f'Loading WN ISI...')
            self.load_isi(self.str_noise_protocol, file_names=self.ls_noise_filenames, bin_edges=isi_bin_edges)
        
    def load_protocol_vcd(self, chunk_dir, dataset_name):
        """Load protocol VCD. 
        Typically for cases when protocol chunk is different from noise chunk.
        So self.vcd has noise data and self.p_vcd has protocol data.
        Can pass to eicorr.MapAcrossChunk to map EIs across noise and protocol chunks.
        """
        
        # Check that globals file exists
        str_globals = os.path.join(chunk_dir, f'{dataset_name}.globals')
        if not os.path.exists(str_globals):
            raise ValueError(f'{str_globals} does not exist.')
        
        print(f'Loading protocol VCD from {chunk_dir}...')
        self.p_vcd = vl.load_vision_data(analysis_path=chunk_dir, dataset_name=dataset_name, 
                                         include_ei=True, include_neurons=True)

    def reload_types(self, str_classification):
        self.types = ctio.CellTypes(str_classification, ls_RGC_labels=self.ls_RGC_labels)
        self.ls_RGC_labels = list(self.types.d_main_IDs.keys())
        print(f'Reloaded types from {str_classification}.')
    
    def remap_sta_vcd(self, d_ID_map: dict):
        # Remap STA and VCD data to new IDs
        d_new_sta = {}
        for n_ID in d_ID_map.keys():
            d_new_sta[d_ID_map[n_ID]] = self.d_sta[n_ID]
        self.d_sta = d_new_sta

        # Remap VCD data
        d_new_vcd = {}
        for n_ID in d_ID_map.keys():
            d_new_vcd[d_ID_map[n_ID]] = self.vcd.main_datatable[n_ID]
        self.vcd.main_datatable = d_new_vcd

        # Remap STA spatial maps
        if hasattr(self, 'd_sta_spatial'):
            d_new_spatial = {}
            for n_ID in d_ID_map.keys():
                d_new_spatial[d_ID_map[n_ID]] = self.d_sta_spatial[n_ID]
            self.d_sta_spatial = d_new_spatial
        
        # Remap STA convex hulls
        if hasattr(self, 'd_sta_convex_hull'):
            d_new_hull = {}
            for n_ID in d_ID_map.keys():
                d_new_hull[d_ID_map[n_ID]] = self.d_sta_convex_hull[n_ID]
            self.d_sta_convex_hull = d_new_hull

        # Update cell IDs
        self.ARR_CELL_IDS = np.array(list(self.d_sta.keys()))
        self.GOOD_CELL_IDS = self.ARR_CELL_IDS.copy()
        self.N_CELLS = len(self.ARR_CELL_IDS)
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)
        print(f'Remapped to {self.N_CELLS} cells.')

    
    def load_psth(self, str_protocol, ls_param_names, 
                  bin_rate=100.0, isi_bin_edges=np.linspace(0,300,601),
                  b_load_isi=True, b_load_ei=False, ls_filenames=None):
        c_data = sd.Dataset(self.str_experiment)
        self.param_names = ls_param_names
        self.str_protocol = str_protocol

        if ls_filenames is not None:
            self.ls_filenames = ls_filenames

        spike_dict, cluster_id, params, unique_params, pre_pts, stim_pts, tail_pts = c_data.get_spike_rate_and_parameters(
        str_protocol, None, self.param_names, sort_algorithm=self.str_algo, bin_rate=bin_rate, sample_rate=20000,
        file_name=self.ls_filenames)

        params = dict_list_to_array(params)
        unique_params = dict_list_to_array(unique_params)

        n_epochs = spike_dict[cluster_id[0]].shape[0]
        n_bin_dt = 1/bin_rate * 1000 # in ms
        
        # Check that pre_pts, stim_pts, tail_pts all have a uniform value
        for pts in [pre_pts, stim_pts, tail_pts]:
            if len(np.unique(pts)) != 1:
                raise ValueError(f'{pts} has more than one unique value.')
            
        n_pre_pts = int(pre_pts[0])
        n_stim_pts = int(stim_pts[0])
        n_tail_pts = int(tail_pts[0])
        
        n_total_pts = n_pre_pts + n_stim_pts + n_tail_pts

        # Get total spike count for each cell
        spike_counts = np.zeros(len(cluster_id))
        for idx, n_id in enumerate(cluster_id):
            # Multiple Sp/s by bin_dt in s to get total spikes
            spike_counts[idx] = np.sum(spike_dict[n_id]) * n_bin_dt / 1000

        self.stim = {'params': params, 'unique_params': unique_params, 
             'n_epochs': n_epochs, 'n_pre_pts': n_pre_pts, 'n_stim_pts': n_stim_pts, 'n_tail_pts': n_tail_pts,
             'n_total_pts': n_total_pts, 'bin_rate': bin_rate, 'n_bin_dt': n_bin_dt,
             'ls_param_names': ls_param_names, 'str_protocol': str_protocol}
        self.spikes = {'spike_dict': spike_dict, 'cluster_id': cluster_id, 
                       'bin_rate': bin_rate, 'n_bin_dt': n_bin_dt,
                       'total_spike_counts': spike_counts}
        
        ids = np.array(cluster_id).astype(int)
        self.ARR_CELL_IDs = np.union1d(self.ARR_CELL_IDS, ids)
        self.GOOD_CELL_IDS = np.intersect1d(self.GOOD_CELL_IDS, ids)
        # self.update_ids(self.GOOD_CELL_IDS)
        self.N_CELLS = len(self.ARR_CELL_IDS)
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)

        if b_load_isi:
            # Load protocol ISI
            self.load_isi(str_protocol, bin_edges=isi_bin_edges, c_data=c_data, file_names=self.ls_filenames)

        if b_load_ei:
            # Load EI vcd
            t_paths = cfg.get_data_paths()
            str_sort_dir = t_paths[0]
            # For now only first data file. TODO Work in avg Ei across all data files
            str_p_vcd = os.path.join(str_sort_dir, self.str_experiment,self.ls_filenames[0], self.str_algo)
            self.p_vcd = vl.load_vision_data(analysis_path=str_p_vcd, 
                                             dataset_name=self.ls_filenames[0], 
                                             include_ei=True)

    
    def load_psth_at_frame_multiple(self, str_protocol, ls_param_names, 
                                    stride=2, ls_filenames=None):
        # stride: Number of bins per frame. 
        # Eg-for 60Hz framerate, stride=2 gets 120Hz bin rate.
        if int(self.str_experiment[:-1]) < 20230926:
            marginal_frame_rate = 60.31807657 # Upper bound on the frame rate to make sure that we don't miss any frames.
        else:
            marginal_frame_rate = 59.941548817817917 # Upper bound on the frame rate to make sure that we don't miss any frames.

        if ls_filenames is not None:
            self.ls_filenames = ls_filenames

        c_data = sd.Dataset(self.str_experiment)
        self.param_names = ls_param_names
        self.str_protocol = str_protocol

        
        print(f'Getting frame multiple spike counts for {str_protocol} {self.ls_filenames}...')
        print(f'Using frame rate {marginal_frame_rate:.2f} Hz. * {stride} = {marginal_frame_rate*stride:.2f} Hz bin rate.')

        spike_dict, cluster_id, params, unique_params, pre_pts, stim_pts, tail_pts, mean_frame_rate = c_data.get_count_at_frame_multiple(
            protocolStr=self.str_protocol, param_names=ls_param_names, 
            sort_algorithm=self.str_algo, file_name=self.ls_filenames, 
            frame_rate=marginal_frame_rate, stride=stride)
        
        print(f'Found mean frame rate: {mean_frame_rate:.2f} Hz.')
        print(f'Using frame rate {marginal_frame_rate:.2f} Hz. * {stride} = {marginal_frame_rate*stride:.2f} Hz bin rate.')
        params['mean_frame_rate'] = marginal_frame_rate

        params = dict_list_to_array(params)
        unique_params = dict_list_to_array(unique_params)

        n_epochs = spike_dict[cluster_id[0]].shape[0]
        n_bin_dt = 1/(marginal_frame_rate*stride) * 1000 # in ms
        bin_rate = marginal_frame_rate * stride
        
        # Check that pre_pts, stim_pts, tail_pts all have a uniform value
        for pts in [pre_pts, stim_pts, tail_pts]:
            if len(np.unique(pts)) != 1:
                raise ValueError(f'{pts} has more than one unique value.')
            
        # n_pre_pts = int(pre_pts[0])
        # n_stim_pts = int(stim_pts[0])
        # n_tail_pts = int(tail_pts[0])
        n_pre_pts = round(params['preTime'][0]*1e-3 * marginal_frame_rate) * stride
        n_stim_pts = round(params['stimTime'][0]*1e-3 * marginal_frame_rate) * stride
        n_tail_pts = round(params['tailTime'][0]*1e-3 * marginal_frame_rate) * stride
        
        n_total_pts = n_pre_pts + n_stim_pts + n_tail_pts
        print(f'Epochs have {n_pre_pts} pre, {n_stim_pts} stim, {n_tail_pts} tail points.')
        print(f'Corresponding to {params["preTime"][0]} ms pre, {params["stimTime"][0]} ms stim, {params["tailTime"][0]} ms tail.')
        print(f'Bin rate: {bin_rate:.2f} Hz; bin dt: {n_bin_dt:.2f} ms')

        # Get PSTH matrix and total spike count for each cell
        n_cells = len(cluster_id)
        n_epochs = spike_dict[cluster_id[0]].shape[0]
        # n_timepts = spike_dict[cluster_id[0]].shape[1]
        psth = np.zeros((n_epochs, n_cells, n_total_pts))
        total_sc = np.zeros(n_cells)
        for idx, n_id in enumerate(cluster_id):
            total_sc[idx] = np.sum(spike_dict[n_id])

            # Convert sc to firing rate in Hz
            spike_dict[n_id] = spike_dict[n_id] * bin_rate
            for n_epoch in range(n_epochs):
                fr = spike_dict[n_id][n_epoch]
                psth[n_epoch, idx, :] = fr[:n_total_pts]

        self.stim = {'params': params, 'unique_params': unique_params, 
                'n_epochs': n_epochs, 'n_pre_pts': n_pre_pts, 'n_stim_pts': n_stim_pts, 'n_tail_pts': n_tail_pts,
                'n_total_pts': n_total_pts, 'bin_rate': bin_rate, 'n_bin_dt': n_bin_dt,
                'ls_param_names': ls_param_names, 'str_protocol': str_protocol}
        self.spikes = {'spike_dict': spike_dict, 'cluster_id': cluster_id, 
                        'bin_rate': bin_rate, 'n_bin_dt': n_bin_dt,
                        'total_spike_counts': total_sc, 'psth': psth}
        
        ids = np.array(cluster_id).astype(int)
        self.ARR_CELL_IDs = np.union1d(self.ARR_CELL_IDS, ids)
        self.GOOD_CELL_IDS = np.intersect1d(self.GOOD_CELL_IDS, ids)
        # self.update_ids(self.GOOD_CELL_IDS)
        self.N_CELLS = len(self.ARR_CELL_IDS)
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)

    def load_spike_times(self, str_protocol, ls_param_names, 
                         bin_rate=1000.0, isi_bin_edges=np.linspace(0, 300, 601),
                         b_load_isi=True, ls_filenames=None):
        """
        Load spike times and associated parameters for a given protocol.
    
        Parameters:
            str_protocol: Protocol name (string).
            ls_param_names: List of parameter names to extract.
            bin_rate: Bin rate for spike times in Hz (default=1000.0).
            isi_bin_edges: Bin edges for ISI histogram (default=np.linspace(0, 300, 601)).
            b_load_isi: Whether to load ISI data (default=True).
            ls_filenames: List of filenames to process (default=None).
        """
        c_data = sd.Dataset(self.str_experiment)
        self.param_names = ls_param_names
        self.str_protocol = str_protocol
    
        if ls_filenames is not None:
            self.ls_filenames = ls_filenames
    
        # Call get_spike_times_and_parameters from symphony_data.Dataset
        spike_times, cluster_id, params, unique_params, pre_pts, stim_pts, tail_pts = c_data.get_spike_times_and_parameters(
            protocolStr=str_protocol, groupStr=None, param_names=ls_param_names, 
            sort_algorithm=self.str_algo, file_name=self.ls_filenames, 
            bin_rate=bin_rate, sample_rate=20000)
    
        params = dict_list_to_array(params)
        unique_params = dict_list_to_array(unique_params)
    
        n_epochs = spike_times.shape[1]
        n_bin_dt = 1 / bin_rate * 1000  # in ms
    
        # Check that pre_pts, stim_pts, tail_pts all have a uniform value
        for pts in [pre_pts, stim_pts, tail_pts]:
            if len(np.unique(pts)) != 1:
                raise ValueError(f'{pts} has more than one unique value.')
    
        n_pre_pts = int(pre_pts[0])
        n_stim_pts = int(stim_pts[0])
        n_tail_pts = int(tail_pts[0])
        n_total_pts = n_pre_pts + n_stim_pts + n_tail_pts
    
        self.stim = {'params': params, 'unique_params': unique_params, 
                     'n_epochs': n_epochs, 'n_pre_pts': n_pre_pts, 'n_stim_pts': n_stim_pts, 'n_tail_pts': n_tail_pts,
                     'n_total_pts': n_total_pts, 'bin_rate': bin_rate, 'n_bin_dt': n_bin_dt,
                     'ls_param_names': ls_param_names, 'str_protocol': str_protocol}
        self.spikes = {'spike_dict': spike_times, 'cluster_id': cluster_id, 
                       'bin_rate': bin_rate, 'n_bin_dt': n_bin_dt}
    
        ids = np.array(cluster_id).astype(int)
        self.ARR_CELL_IDs = np.union1d(self.ARR_CELL_IDS, ids)
        self.GOOD_CELL_IDS = np.intersect1d(self.GOOD_CELL_IDS, ids)
        # self.update_ids(self.GOOD_CELL_IDS)
        self.N_CELLS = len(self.ARR_CELL_IDS)
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)
    
        if b_load_isi:
            # Load protocol ISI
            self.load_isi(str_protocol, bin_edges=isi_bin_edges, c_data=c_data, file_names=self.ls_filenames)
    
    
    def load_isi(self, str_protocol, bin_edges=np.linspace(0,300,601), c_data=None, file_names=None):
        # Get ISI
        if c_data is None:
            c_data = sd.Dataset(self.str_experiment)
        if str_protocol not in self.isi.keys():
            print(f'Loading ISI for {str_protocol} {file_names}...')
            acf, isi, isi_cluster_id = c_data.get_interspike_interval(str_protocol, None, self.str_algo, 
                                                        file_name=file_names, bin_edges=bin_edges)
            self.isi[str_protocol] = {'acf': acf, 'isi': isi, 'isi_cluster_id': isi_cluster_id, 'isi_bin_edges': bin_edges}
            print(f'Loaded ISI for {len(isi_cluster_id)} cells.')

            self.ARR_CELL_IDS = np.union1d(self.ARR_CELL_IDS, np.array(isi_cluster_id))
        else:
            print(f'ISI for {str_protocol} already loaded.')

    def print_stim_summary(self):
        # Print stim summary from stim dictionary
        n_bin_dt = self.stim['n_bin_dt']
        print(f'Epoch length: {self.stim["n_total_pts"] * n_bin_dt:.2f} ms')
        print('Total epochs: ' + str(self.stim['n_epochs']))
        print(f'pre: {self.stim["n_pre_pts"] * n_bin_dt:.2f} ms; stim: {self.stim["n_stim_pts"] * n_bin_dt:.2f} ms; tail: {self.stim["n_tail_pts"] * n_bin_dt:.2f} ms')
        print('pre pts: ' + str(self.stim['n_pre_pts']) + '; stim pts: ' + str(self.stim['n_stim_pts']) + '; tail pts: ' + str(self.stim['n_tail_pts']))
        print(f'bin rate: {self.spikes["bin_rate"]:.2f} Hz; bin dt: {n_bin_dt:.2f} ms')
        
    def save_pkl(self, str_path: str=None):
        if not str_path:
            str_path = os.path.join(self.str_experiment, self.str_datafile)
        d_save = {'stim': self.stim, 'spikes': self.spikes, 
                  'isi': self.isi, 'ARR_CELL_IDS': self.ARR_CELL_IDS, 'GOOD_CELL_IDS': self.GOOD_CELL_IDS,
                  'str_protocol': self.str_protocol, 'param_names': self.param_names}
                  
        
        # Check if d_sta is present
        if hasattr(self, 'd_sta'):
            d_save['d_sta'] = self.d_sta
            d_save['N_HEIGHT'] = self.N_HEIGHT
            d_save['N_WIDTH'] = self.N_WIDTH
            d_save['NOISE_GRID_SIZE'] = self.NOISE_GRID_SIZE
        if hasattr(self, 'd_sta_spatial'):
            d_save['d_sta_spatial'] = self.d_sta_spatial
        if hasattr(self, 'd_sta_convex_hull'):
            d_save['d_sta_convex_hull'] = self.d_sta_convex_hull
        with open(str_path, 'wb') as f:
            pickle.dump(d_save, f)
        print('Saved to ' + str_path)

    def load_pkl(self, str_path: str):
        with open(str_path, 'rb') as f:
            d_load = pickle.load(f)
        self.stim = d_load['stim']
        self.spikes = d_load['spikes']
        self.isi = d_load['isi']
        self.ARR_CELL_IDS = d_load['ARR_CELL_IDS']
        self.N_CELLS = len(self.ARR_CELL_IDS)
        # self.str_protocol = d_load['str_protocol']
        # self.param_names = d_load['param_names']
        # self.update_ids(d_load['GOOD_CELL_IDS'])        

        # Check if d_sta is present
        if 'd_sta' in d_load.keys():
            self.d_sta = d_load['d_sta']
            self.N_HEIGHT = d_load['N_HEIGHT']
            self.N_WIDTH = d_load['N_WIDTH']
            self.NOISE_GRID_SIZE = d_load['NOISE_GRID_SIZE']
        if 'd_sta_spatial' in d_load.keys():
            self.d_sta_spatial = d_load['d_sta_spatial']
        if 'd_sta_convex_hull' in d_load.keys():
            self.d_sta_convex_hull = d_load['d_sta_convex_hull']
        print('Loaded from ' + str_path)

    def save_to_mat(self, str_path: str):
        print(f'Saving to {str_path}...')


    def update_ids(self, good_ids: np.ndarray):
        self.GOOD_CELL_IDS = good_ids
        self.N_GOOD_CELLS = len(good_ids)
        print(f'Updating GOOD_CELL_IDS to {self.N_GOOD_CELLS} cells.')
        
        # Update d_main_IDs
        for str_type in self.types.d_main_IDs.keys():
            type_ids = self.types.d_main_IDs[str_type]
            new_ids = np.intersect1d(type_ids, good_ids)
            self.types.d_main_IDs[str_type] = new_ids
            print(f'{str_type}: {len(new_ids)}/{len(type_ids)}')
    
    def remove_ids(self, bad_ids: np.ndarray):
        # Remove bad_ids from GOOD_CELL_IDS
        good_ids = np.setdiff1d(self.GOOD_CELL_IDS, bad_ids)
        self.update_ids(good_ids)