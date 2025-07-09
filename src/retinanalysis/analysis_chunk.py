import retinanalysis.schema as schema
import os
from retinanalysis.settings import NAS_ANALYSIS_DIR, NAS_DATA_DIR
import pandas as pd
import retinanalysis.vision_utils as vu
from hdf5storage import loadmat
import pickle
import numpy as np
from typing import List
import matplotlib.pyplot as plt

def get_noise_name_by_exp(exp_name):
    # Pull appropriate noise protocol for cell typing
    if int(exp_name[:8]) < 20230926:
            noise_protocol_name = 'manookinlab.protocols.FastNoise'
    else:
        noise_protocol_name = 'manookinlab.protocols.SpatialNoise'
    return noise_protocol_name

class AnalysisChunk:

    def __init__(self, exp_name: str=None, chunk_name: str=None, 
                 ss_version: str = 'kilosort2.5', pkl_file: str=None, 
                 b_load_spatial_maps: bool=True, **vu_kwargs):
        if pkl_file is None:
            if exp_name is None or chunk_name is None:
                raise ValueError("Either exp_name and chunk_name or pkl_file must be provided.")
        else:
            # Load from pickle file if string, otherwise must be a dict
            if isinstance(pkl_file, str):
                with open(pkl_file, 'rb') as f:
                    d_out = pickle.load(f)
            else:
                d_out = pkl_file
                pkl_file = "input dict."
            self.__dict__.update(d_out)
            self.vcd = vu.get_analysis_vcd(self.exp_name, self.chunk_name, self.ss_version, **vu_kwargs)
            print(f"AnalysisChunk loaded from {pkl_file}")
            return
        
        self.exp_name = exp_name
        self.chunk_name = chunk_name
        self.ss_version = ss_version
        
        # Pull Experiment ID
        exp_id = schema.Experiment() & {'exp_name': self.exp_name}
        self.exp_id = exp_id.fetch('id')[0]

        # Pull chunk id
        chunk_id = schema.SortingChunk() & {'experiment_id' : self.exp_id, 'chunk_name' : self.chunk_name}
        self.chunk_id = chunk_id.fetch('id')[0]

        self.noise_protocol = get_noise_name_by_exp(exp_name)

        # Pull protocol id
        protocol_id = schema.Protocol() & {'name' : self.noise_protocol}
        self.protocol_id = protocol_id.fetch('protocol_id')[0]

        self.vcd = vu.get_analysis_vcd(self.exp_name, self.chunk_name, self.ss_version, **vu_kwargs)
        self.get_noise_params()
        self.cell_ids = self.vcd.get_cell_ids()
        self.get_rf_params()
        self.get_df()
        if b_load_spatial_maps:
            self.get_spatial_maps()

    def get_noise_params(self):
        self.staXChecks = int(self.vcd.runtimemovie_params.width)
        self.staYChecks = int(self.vcd.runtimemovie_params.height)

        # Pull epoch block and epoch to get num X and num Y checks used in noise
        epoch_blocks = schema.EpochBlock() & {'experiment_id' : self.exp_id, 'chunk_id' : self.chunk_id, 'protocol_id' : self.protocol_id}
        epoch_block_ids = [epoch_blocks.fetch('id')[idx] for idx in range(len(epoch_blocks))]
        epochs = [schema.Epoch() & {'experiment_id' : self.exp_id, 'parent_id' : block_id} for block_id in epoch_block_ids]

        self.numXChecks = [epoch.fetch('parameters')[0]['numXChecks'] for epoch in epochs]
        self.numYChecks = [epoch.fetch('parameters')[0]['numYChecks'] for epoch in epochs]

        assert all(element == self.numXChecks[0] for element in self.numXChecks), "Not all epoch blocks used same number of X checks"
        assert all(element == self.numYChecks[0] for element in self.numYChecks), "Not all epoch blocks used same number of Y checks"

        self.numXChecks = self.numXChecks[0]
        self.numYChecks = self.numYChecks[0]

        self.deltaXChecks = int((self.numXChecks - self.staXChecks)/2)
        self.deltaYChecks = int((self.numYChecks - self.staYChecks)/2)
        
        self.microns_per_pixel = epochs[0].fetch('parameters')[0]['micronsPerPixel']
        self.canvas_size = epochs[0].fetch('parameters')[0]['canvasSize']

        # Pull noise data file names
        noise_data_dirs = epoch_blocks.fetch('data_dir')
        self.data_files = [os.path.basename(path) for path in noise_data_dirs]

        typing_files = schema.CellTypeFile() & {'chunk_id' : self.chunk_id, 'algorithm': self.ss_version}
        self.typing_files = [file_name for file_name in typing_files.fetch('file_name')] 

        self.pixels_per_stixel = self.canvas_size[0]/self.numXChecks
        self.microns_per_stixel = self.microns_per_pixel * self.pixels_per_stixel

    def get_rf_params(self):
        self.rf_params = dict()
        for id in self.cell_ids:
            center_x = self.vcd.main_datatable[id]['x0']
            center_y = self.vcd.main_datatable[id]['y0']
            self.rf_params[id] = {'center_x' : center_x + self.deltaXChecks,
                                'center_y' : (self.staYChecks - center_y) + self.deltaYChecks,
                                'std_x' : self.vcd.main_datatable[id]['SigmaX'],
                                'std_y' : self.vcd.main_datatable[id]['SigmaY'],
                                'rot' : self.vcd.main_datatable[id]['Theta']}
              
    def get_df(self):
        center_x = [self.rf_params[id]['center_x'] for id in self.cell_ids]
        center_y = [self.rf_params[id]['center_y'] for id in self.cell_ids]
        std_x = [self.rf_params[id]['std_x'] for id in self.cell_ids]
        std_y = [self.rf_params[id]['std_y'] for id in self.cell_ids]
        rot = [self.rf_params[id]['rot'] for id in self.cell_ids]

        df_dict = {'cell_id': self.cell_ids, 'center_x' : center_x,
                   'center_y': center_y, 'std_x' : std_x,
                   'std_y' : std_y, 'rot' : rot} 


        current_file_path = os.path.dirname(os.path.abspath(__file__))
        src_folder_path = os.path.split(current_file_path)[0]
        root_folder = os.path.split(src_folder_path)[0]
        cell_types_list = pd.read_csv(os.path.join(root_folder, 'assets/cell_types.csv'))
        
        cell_types = cell_types_list['cell_types'].values

        for idx, typing_file in enumerate(self.typing_files):
            file_path = os.path.join(NAS_ANALYSIS_DIR, self.exp_name, self.chunk_name, self.ss_version, typing_file)
            d_result = dict()
            
            with open(file_path, 'r') as file:
                for line in file:
                    # Split each line into key and value using the specified delimiter
                    key, value = map(str.strip, line.split(' ', 1))
                                
                    # Add key-value pair to the dictionary
                    d_result[int(key)] = value

            for cell in self.cell_ids:
                if cell in d_result.keys():

                    for type in cell_types:
                        if type in d_result[cell]:
                            d_result[cell] = type
                            break
                else:
                    d_result[cell] = 'Unknown'
                
                if 'All' in d_result[cell]:
                    d_result[cell] = 'Unknown'
                
            
            
            classification = [d_result[cell] for cell in self.cell_ids]
            df_dict[f'typing_file_{idx}'] = classification
        
        self.df_cell_params = pd.DataFrame(df_dict)

    def get_spatial_maps(self, ls_channels=[0,2]):
        # By default load red and blue channel spatial maps. 
        mat_file = os.path.join(NAS_DATA_DIR, self.exp_name, self.chunk_name, self.ss_version, f'{self.ss_version}_params.mat')
        if not os.path.exists(mat_file):
            print(f'_params.mat file not found: {mat_file}')
            return
        
        d_params = loadmat(mat_file)
        d_spatial_maps = {}
        for idx_ID, n_ID in enumerate(self.cell_ids):
            # TODO pad spatial maps to match N_HEIGHT and N_WIDTH @roaksleaf pls help 
            # Cell ID index in vcd should be same as in _params.mat
            spat_mat = d_params['spatial_maps'][idx_ID][:, :, ls_channels]
            left_pad = self.deltaXChecks
            right_pad = self.numXChecks - self.staXChecks - self.deltaXChecks
            top_pad = self.deltaYChecks
            bottom_pad = self.numYChecks - self.staYChecks - self.deltaYChecks
            padded = np.pad(spat_mat, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)
            d_spatial_maps[n_ID] = padded
            
        self.d_spatial_maps = d_spatial_maps
        print(f'Loaded spatial maps for channels {ls_channels} and {len(self.cell_ids)} cells of shape {d_spatial_maps[self.cell_ids[0]].shape}')# from:\n{mat_file}')
        print(f'Spatial maps have been padded to align with RF parameters.')
        # TODO could also load convex hull fits too under 'hull_vertices'

    def plot_rfs(self, noise_ids: List[int] = None, cell_types: List[str] = None,
                 typing_file: str = None, units: str = 'pixels', std_scaling: float = 1.6,
                 b_zoom: bool = False, n_pad = 6):

        if typing_file is None:
            typing_file = self.typing_files[0]
        
        typing_file_idx = self.typing_files.index(typing_file)

        if typing_file not in self.typing_files:
            raise FileNotFoundError("Given Typing File Doesn't Exist in Analysis Chunk")
        
        if noise_ids is None and cell_types is None:
            filtered_df = self.df_cell_params
            noise_ids = filtered_df['cell_id'].values
            cell_types = filtered_df[f'typing_file_{typing_file_idx}'].unique()
        elif noise_ids is None:
            filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} == @cell_types')
            noise_ids = filtered_df['cell_id'].values
        elif cell_types is None:
            filtered_df = self.df_cell_params.query(f'cell_id  == @noise_ids')
            cell_types = filtered_df[f'typing_file_{typing_file_idx}'].unique()
        else:
            filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} == @cell_types and cell_id == @noise_ids')

        if len(filtered_df) == 0:
            print("No data found for the given noise_ids and cell_types.")
            return

        d_noise_ids_by_type = {ct : filtered_df.query(f'typing_file_{typing_file_idx} == @ct')['cell_id'].values for ct in cell_types}

        d_ells_by_type, scale_factor = vu.get_ells(self, d_noise_ids_by_type, std_scaling = std_scaling, units = units)

        rows = int(np.ceil(len(cell_types)/4))
        cols = np.min([(len(cell_types)-1 % 4)+1, 4])
        size = (3*cols, int(3*rows))

        fig, axs = plt.subplots(nrows = rows, ncols = cols, figsize = size)

        if cols != 1:
            axs = axs.flatten()
        else:
            axs = np.array([axs])

        for idx, ct in enumerate(cell_types):
            ax = axs[idx]
            for id in d_ells_by_type[ct]:
                ax.add_patch(d_ells_by_type[ct][id])

            ax.set_xlim(0,self.numXChecks * scale_factor)
            ax.set_ylim(0,self.numYChecks * scale_factor)

            ax.set_ylabel(units.lower())
            ax.set_xlabel(units.lower())

            n_cells = len(d_ells_by_type[ct])
            ax.set_title(f"{ct}, (n = {n_cells})")

        # Remove extra empty axes 
        num_axes = (rows-1)*4 + cols
        empty_axes = num_axes - len(cell_types)

        for i in range(empty_axes):
            fig.delaxes(axs[num_axes - 1 - i])

        if b_zoom:
            x_min, x_max = filtered_df['center_x'].min(), filtered_df['center_x'].max()
            y_min, y_max = filtered_df['center_y'].min(), filtered_df['center_y'].max()
            
            for ax in axs:
                ax.set_xlim((x_min - n_pad)*scale_factor, (x_max + n_pad)*scale_factor)
                ax.set_ylim((y_min - n_pad)*scale_factor, (y_max + n_pad)*scale_factor)
        
        fig.suptitle("RFs by Cell Type", fontsize = 15)
        fig.tight_layout()

        return axs

        
    def plot_timecourses(self, noise_ids: List[int]=None, cell_types: List[int]=None,
                         typing_file: str = None, units: str = 'ms', std_scaling: float = 2) -> np.ndarray:
        
        if 'ms' in units.lower() or 'milliseconds' in units.lower():
            scale_factor = 1
        elif 's' in units.lower() or 'seconds' in units.lower():
            scale_factor = 1e-3
        else:
            raise NameError("Units string must be 'ms', 'milliseconds', 's' or 'seconds'")

        if typing_file is None:
            typing_file = self.typing_files[0]
        
        typing_file_idx = self.typing_files.index(typing_file)

        if typing_file not in self.typing_files:
            raise FileNotFoundError("Given Typing File Doesn't Exist in Analysis Chunk")

        if noise_ids is None and cell_types is None:
            filtered_df = self.df_cell_params
            noise_ids = filtered_df['cell_id'].values
            cell_types = filtered_df[f'typing_file_{typing_file_idx}'].unique()
        elif noise_ids is None:
            filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} == @cell_types')
            noise_ids = filtered_df['cell_id'].values
        elif cell_types is None:
            filtered_df = self.df_cell_params.query(f'cell_id  == @noise_ids')
            cell_types = filtered_df[f'typing_file_{typing_file_idx}'].unique()
        else:
            filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} == @cell_types and cell_id == @noise_ids')

        d_noise_ids_by_type = {ct : filtered_df.query(f'typing_file_{typing_file_idx} == @ct')['cell_id'].values for ct in cell_types}

        d_timecourses_by_type = vu.get_timecourses(self, d_noise_ids_by_type)

        rows = int(np.ceil(len(cell_types)/4))
        cols = np.min([(len(cell_types)-1 % 4)+1, 4])
        size = (4.5*cols, int(3*rows))

        fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = size)

        if cols != 1:
            ax = ax.flatten()

        for idx, ct in enumerate(cell_types):

            time_vals = np.linspace(-491.66,8.33,len(d_timecourses_by_type[ct]['rg_mean']))*scale_factor
            if cols != 1:
                rg_err_top = d_timecourses_by_type[ct]['rg_mean'] + d_timecourses_by_type[ct]['rg_std']*std_scaling
                rg_err_bottom = d_timecourses_by_type[ct]['rg_mean'] - d_timecourses_by_type[ct]['rg_std']*std_scaling
                ax[idx].plot(time_vals, d_timecourses_by_type[ct]['rg_mean'], '-g')
                ax[idx].fill_between(time_vals, rg_err_bottom, rg_err_top, alpha = 0.4, color = 'g')

                b_err_top = d_timecourses_by_type[ct]['b_mean'] + d_timecourses_by_type[ct]['b_std']*std_scaling
                b_err_bottom = d_timecourses_by_type[ct]['b_mean'] - d_timecourses_by_type[ct]['b_std']*std_scaling
                ax[idx].plot(time_vals, d_timecourses_by_type[ct]['b_mean'], '-b')
                ax[idx].fill_between(time_vals, b_err_bottom, b_err_top, alpha = 0.4, color = 'b') 

                ax[idx].set_xlim([time_vals[0], time_vals[-1]])

                ax[idx].set_ylabel(f"STA (arb. units)")
                ax[idx].set_xlabel(f"Time ({units})")
                
                ax[idx].set_title(f"{ct}, (n = {d_timecourses_by_type[ct]['rg_timecourses'].shape[0]})")

            else: 
                rg_err_top = d_timecourses_by_type[ct]['rg_mean'] + d_timecourses_by_type[ct]['rg_std']*std_scaling
                rg_err_bottom = d_timecourses_by_type[ct]['rg_mean'] - d_timecourses_by_type[ct]['rg_std']*std_scaling
                ax.plot(time_vals, d_timecourses_by_type[ct]['rg_mean'], '-g')
                ax.fill_between(time_vals, rg_err_bottom, rg_err_top, alpha = 0.4, color = 'g')

                b_err_top = d_timecourses_by_type[ct]['b_mean'] + d_timecourses_by_type[ct]['b_std']*std_scaling
                b_err_bottom = d_timecourses_by_type[ct]['b_mean'] - d_timecourses_by_type[ct]['b_std']*std_scaling
                ax.plot(time_vals, d_timecourses_by_type[ct]['b_mean'], '-b')
                ax.fill_between(time_vals, b_err_bottom, b_err_top, alpha = 0.4, color = 'b') 

                ax.set_xlim([time_vals[0], time_vals[-1]])

                ax.set_ylabel(f"STA (arb. units)")
                ax.set_xlabel(f"Time ({units})")
                
                ax.set_title(f"{ct}, (n = {d_timecourses_by_type[ct]['rg_timecourses'].shape[0]})")
        
        # Remove extra empty axes 
        num_axes = (rows-1)*4 + cols
        empty_axes = num_axes - len(cell_types)

        for i in range(empty_axes):
            fig.delaxes(ax[num_axes - 1 - i])

        fig.suptitle("Timecourse by Cell Type", fontsize = 15)
        fig.tight_layout()

        return ax



    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  chunk_name: {self.chunk_name}\n"
        str_self += f"  ss_version: {self.ss_version}\n"
        str_self += f"  noise_protocol: {self.noise_protocol}\n"
        str_self += f"  data_files: {self.data_files}\n"
        str_self += f"  typing_files: {self.typing_files}\n"
        str_self += f"  numXChecks: {self.numXChecks}\n"
        str_self += f"  numYChecks: {self.numYChecks}\n"
        str_self += f"  staXChecks: {self.staXChecks}\n"
        str_self += f"  staYChecks: {self.staYChecks}\n"
        str_self += f"  canvas_size: {self.canvas_size}\n"
        str_self += f"  microns_per_pixel: {self.microns_per_pixel}\n"
        str_self += f"  cell_ids of length: {len(self.cell_ids)}\n"
        str_self += f"  rf_params with fiels: {list(self.rf_params[self.cell_ids[0]].keys())}\n"
        str_self += f"  df_cell_params of shape: {self.df_cell_params.shape}\n"
        if hasattr(self, 'd_spatial_maps'):
            str_self += f"  d_spatial_maps with {len(self.d_spatial_maps)} cells\n"
        else:
            str_self += "  d_spatial_maps not loaded\n"
        return str_self

    def export_to_pkl(self, file_path: str):
        d_out = self.__dict__.copy()
        # Pop out vcd
        d_out.pop('vcd')
        with open(file_path, 'wb') as f:
            pickle.dump(d_out, f)
        print(f"AnalysisChunk exported to {file_path}")

