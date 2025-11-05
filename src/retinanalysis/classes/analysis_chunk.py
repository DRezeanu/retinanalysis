import retinanalysis 
import retinanalysis.config.schema as schema
import os
from retinanalysis.config.settings import (ANALYSIS_DIR,
                                           DATA_DIR)
import pandas as pd
from retinanalysis.utils.vision_utils import (get_analysis_vcd,
                                              get_ells,
                                              get_timecourses)
from hdf5storage import loadmat
import pickle
import numpy as np
from typing import (cast,
                    List,
                    Dict,
                    Optional,
                    Any)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import visionloader as vl
from scipy.ndimage import zoom

try:
    import importlib.resources as ir
except:
    import importlib_resources as ir # type: ignore

from retinanalysis.utils.datajoint_utils import get_noise_name_by_exp

class AnalysisChunk:
    """
    Class that contains data from an MEA sorting chunk created primarily from spatial noise.

    This is unique to spatial noise chunks because these chunks contain '.sta'. and '.params'
    files while regular sorting chunks and data files do not.

    Init Parameters:
    exp_name (str): The name of the experiment as seen in the 'exp_name' entry of the datajoint database

    chunk_name (str): The name of the sorting chunk (e.g. 'chunk2'). This chunk must be findable in the
    analysis directory defined by the config.ini file in the retinanalysis_root/config folder.

    ss_version (str): spike sorting version, default is 'kilosort2.5'. This is mostly used to find the
    right folder. Relevant files should be located at: 'analysis_directory/chunk_name/ss_version/'

    b_load_spatial_maps (bool): Whether or not to load the spatial maps for the cells in this chunk,
    default value is True.

    pkl_file (dict | str): Optional. If you have exported an analysis chunk to a pickle file using
    the export_to_pkl() method, you can give only this input to reload the object from the pickle file.

    Init Returns:
    AnalysisChunk object for experiment name, chunk, and ss_version given in the initializer.

    Properties:
    Use the print command on a AnalysisChunk instance to get a list of all properties contained in
    the object
    """
    def __init__(self, exp_name: Optional[str]=None, chunk_name: Optional[str]=None, 
                 ss_version: str = 'kilosort2.5', pkl_file: Optional[dict | str]=None, 
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
            self.vcd = get_analysis_vcd(self.exp_name, self.chunk_name, self.ss_version, **vu_kwargs)
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

        self.vcd = get_analysis_vcd(self.exp_name, self.chunk_name, self.ss_version, **vu_kwargs)
        self.get_noise_params()
        self.cell_ids = self.vcd.get_cell_ids()
        self.get_rf_params()
        self.get_df()
        if b_load_spatial_maps:
            self.get_spatial_maps()

    def get_noise_params(self):
        """
        Method for accessing spatial noise and STA parameters, and correcting for any
        discrepancy due to cropping.
        """
        self.staXChecks = int(self.vcd.runtimemovie_params.width)
        self.staYChecks = int(self.vcd.runtimemovie_params.height)

        # Pull epoch block and epoch to get num X and num Y checks used in noise
        epoch_blocks = schema.EpochBlock() & {'experiment_id' : self.exp_id, 'chunk_id' : self.chunk_id, 'protocol_id' : self.protocol_id}
        epoch_block_ids = [epoch_blocks.fetch('id')[idx] for idx in range(len(epoch_blocks))]
        epochs = [schema.Epoch() & {'experiment_id' : self.exp_id, 'parent_id' : block_id} for block_id in epoch_block_ids]

        numXChecks = np.array([epoch.fetch('parameters')[0]['numXChecks'] for epoch in epochs])
        numYChecks = np.array([epoch.fetch('parameters')[0]['numYChecks'] for epoch in epochs])
        
        if (not all(element == numXChecks[0] for element in numXChecks) and
            not all(element == numYChecks[0] for element in numYChecks)):
            print('WARNING: Not all epoch blocks used the same number of X and Y checks\n')

            vision_micronsPerStixel = self.vcd.runtimemovie_params.micronsPerStixelX
            gridSizes = np.array([epoch.fetch('parameters')[0]['gridSize'] for epoch in epochs])
            
            self.numXChecks = int(numXChecks[gridSizes == vision_micronsPerStixel])
            self.numYChecks = int(numYChecks[gridSizes == vision_micronsPerStixel])

        else:
            self.numXChecks = int(numXChecks[0])
            self.numYChecks = int(numYChecks[0])

        self.deltaXChecks = (self.numXChecks - self.staXChecks)/2
        self.deltaYChecks = (self.numYChecks - self.staYChecks)/2
        
        self.microns_per_pixel = epochs[0].fetch('parameters')[0]['micronsPerPixel']
        self.canvas_size = epochs[0].fetch('parameters')[0]['canvasSize']

        # Pull noise data file names
        noise_data_dirs = epoch_blocks.fetch('data_dir')
        self.data_files = [os.path.basename(path) for path in noise_data_dirs]

        # Pull typing files directly from available Analysis Directory... avoids issues with datajoint
        # not updating typing files on existing experiments
        typing_file_dir = os.path.join(ANALYSIS_DIR, self.exp_name, self.chunk_name, self.ss_version)
        self.typing_files = [file for file in os.listdir(typing_file_dir) if 'txt' in os.path.splitext(file)[1]]

        # typing_files = schema.CellTypeFile() & {'chunk_id' : self.chunk_id, 'algorithm': self.ss_version}
        # self.typing_files = [file_name for file_name in typing_files.fetch('file_name')] 

        self.pixels_per_stixel = self.canvas_size[0]/self.numXChecks
        self.microns_per_stixel = self.microns_per_pixel * self.pixels_per_stixel

    def get_rf_params(self):
        """
        Method for pulling the receptive field parameters stored in the vision cell data table (VCD).
        
        This method also corrects for Y-flipping and any crop discrepancies between the size of the
        spatial noise and the size of the STA.
        """
        self.rf_params = dict()
        broken_ids = []
        for id in self.cell_ids:
            try: 
                center_x = self.vcd.main_datatable[id]['x0']
                center_y = self.vcd.main_datatable[id]['y0']
                self.rf_params[id] = {'center_x' : center_x + self.deltaXChecks,
                                    'center_y' : (self.staYChecks - center_y) + self.deltaYChecks,
                                    'std_x' : self.vcd.main_datatable[id]['SigmaX'],
                                    'std_y' : self.vcd.main_datatable[id]['SigmaY'],
                                    'rot' : self.vcd.main_datatable[id]['Theta']}
            except:
                print(f"Issue with id {id}...\nWill remove from cell_ids list.")
                broken_ids.append(id)
            
        for id in broken_ids:
            self.cell_ids.remove(id)

    def get_cells_by_region(self, roi: Dict[str, float], units: str = 'pixels'):
        """
        Method for pulling cell_ids by region of interest.
        
        Parameters:
        roi (dict):                     roi definition as a dictionary with 4 values. 'x_min',
                                        'x_max', 'y_min', 'y_max'. These define the vertical and
                                        horizontal lines that define the region of interest. Units
                                        of ROI definition must match the units parameter!
                                        
        units (str):                    units to use when defining the roi. Must be either 'pixels',
                                        'microns', or 'stixels'. Default 'pixels'.

        Returns:
        arr_ids (ndarray):              returns a 1D array of cell ids whose center_x and center_y fall within the
                                        defined roi.                                
        """

        if 'pixels' in units.lower():
            unit_scaling = self.pixels_per_stixel
        elif 'microns' in units.lower():
            unit_scaling = self.microns_per_stixel
        elif 'stixels' in units.lower():
            unit_scaling = 1
        else:
            raise Exception("Units must be 'pixels', 'microns' or 'stixels'")
        
        bounding_box = dict()
        for key, val in roi.items():
            bounding_box[key] = val/unit_scaling
         
        x_min = bounding_box['x_min']
        x_max = bounding_box['x_max']
        y_min = bounding_box['y_min']
        y_max = bounding_box['y_max']

        df_cell_params_filtered = self.df_cell_params.query("center_x > @x_min and center_x < @x_max and center_y > @y_min and center_y < @y_max")
        arr_ids = df_cell_params_filtered['cell_id'].values

        return arr_ids
              
    def get_df(self):
        """
        Internal method for generating the cell params dataframe accessible
        as self.df_cell_params
        """
        center_x = [self.rf_params[id]['center_x'] for id in self.cell_ids]
        center_y = [self.rf_params[id]['center_y'] for id in self.cell_ids]
        std_x = [self.rf_params[id]['std_x'] for id in self.cell_ids]
        std_y = [self.rf_params[id]['std_y'] for id in self.cell_ids]
        rot = [self.rf_params[id]['rot'] for id in self.cell_ids]

        df_dict = {'cell_id': self.cell_ids, 'center_x' : center_x,
                   'center_y': center_y, 'std_x' : std_x,
                   'std_y' : std_y, 'rot' : rot} 

        cell_types_list_path = str(ir.files(retinanalysis) / "assets/cell_types.csv")
        cell_types_list = pd.read_csv(cell_types_list_path)
        cell_types = cell_types_list['cell_types'].values

        for idx, typing_file in enumerate(self.typing_files):
            file_path = os.path.join(ANALYSIS_DIR, self.exp_name, self.chunk_name, self.ss_version, typing_file)
            d_result = dict()
            
            with open(file_path, 'r') as file:
               for line in file:
                    # Split each line into key and value using the specified delimiter
                    key, value = map(str.strip, line.split(' ', 1))
                    sub_values = value.split('/')
                    
                    # Add key-value pair to the dictionary
                    d_result[int(key)] = sub_values[:-1]

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
        mat_file = os.path.join(DATA_DIR, self.exp_name, self.chunk_name, self.ss_version, f'{self.ss_version}_params.mat')
        if not os.path.exists(mat_file):
            print(f'_params.mat file not found: {mat_file}')
            return
        
        d_params = loadmat(mat_file)
        d_spatial_maps = {}
        for idx_ID, n_ID in enumerate(self.cell_ids):
            # TODO pad spatial maps to match N_HEIGHT and N_WIDTH @roaksleaf pls help 
            # Cell ID index in vcd should be same as in _params.mat
            spat_mat = d_params['spatial_maps'][idx_ID][:, :, ls_channels]
            left_pad = int(self.deltaXChecks)
            right_pad = int(self.numXChecks - self.staXChecks - self.deltaXChecks)
            top_pad = int(self.deltaYChecks)
            bottom_pad = int(self.numYChecks - self.staYChecks - self.deltaYChecks)

            padded = np.pad(spat_mat, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)
            d_spatial_maps[n_ID] = padded
            
        self.d_spatial_maps = d_spatial_maps
        print(f'Loaded spatial maps for channels {ls_channels} and {len(self.cell_ids)} cells of shape {d_spatial_maps[self.cell_ids[0]].shape}')# from:\n{mat_file}')
        print(f'Spatial maps have been padded to align with RF parameters.\n')
        # TODO could also load convex hull fits too under 'hull_vertices'

    def plot_rfs(self, noise_ids: Optional[List[int]] = None, cell_types: Optional[List[str]] = None,
                 typing_file: Optional[str] = None, units: str = 'pixels', std_scaling: float = 1.6,
                 b_zoom: bool = False, n_pad: int = 6, minimum_n: int = 1,
                 roi: Optional[Dict[str, float]] = None, label_cells: bool = False) -> Optional[np.ndarray[Any, np.dtype[np.object_]]]:
        """
        Method for plotting the receptive fields for a given list of cell ids, cell types, 
        or a union of both. If no cell_ids or cell types are given, all cells in the
        analysis chunk are plotted by type.
        
        Parameters:
        noise_ids (List[int]):  A list of cell_ids to plot. Default None.
        cell_types (List[str]): A list of cell_type strings, (e.g. ['OnP', 'OffP']). Default None.
        typing_fyle (str):      A typing file name which is used to determine the cell types for any
                                given cell_ids. If none is given, the 0th typing file associated
                                with the analysis chunk is used. Default None.
        units (str):            Units to use when plotting the receptive fields. Must be either
                                'pixels', 'microns', or 'stixels'. Default 'pixels'.
        std_scaling (float):    Factor used to scale the standard deviation of the plotted
                                receptive fields. Default 1.6
        b_zoom (bool):          Boolean value indicating whether or not to zoom the plots in on 
                                the cell mosaic. Default False
        n_pad (int):            Padding value (in stixels) used with b_zoom. B_zoom will zoom
                                into the min and max center_x and center_y values in the mosaic,
                                and n_pad will zoom back out by the given number of stixels. Default 6
        minimum_n (int):        min number of cells required to actually plot the output
        roi (dict):            roi definition as a dictionary with 4 values. 'x_min',
                                'x_max', 'y_min', 'y_max'. These define the vertical and
                                horizontal lines that define the region of interest
        
        Returns:
        axs (axes):             Axes object that contains all of the axes used in the receptive field
                                figure. There will be as many axes as there are cell_types represented
                                in the plot.

        """
        if isinstance(cell_types, str):
            cell_types = [cell_types]
        
        if isinstance(noise_ids, int) or isinstance(noise_ids, float):
            noise_ids = [int(noise_ids)]

        if typing_file is None:
            try:
                typing_file = self.typing_files[0]
            except:
                print(f'No typing files for {self.exp_name} {self.chunk_name}')
                return

        if typing_file not in self.typing_files:
            print(f"{typing_file} Doesn't Exist in {self.exp_name} {self.chunk_name}")
            return

        typing_file_idx = self.typing_files.index(typing_file)
        
        if noise_ids is None and cell_types is None:
            filtered_df = self.df_cell_params
            noise_ids = list(filtered_df['cell_id'].values)
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
        elif noise_ids is None:
            filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} in @cell_types')
            noise_ids = list(filtered_df['cell_id'].values)
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
        elif cell_types is None:
            filtered_df = self.df_cell_params.query(f'cell_id  in @noise_ids')
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
        else:
            filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} in @cell_types and cell_id in @noise_ids')
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())

        if roi is not None:
            roi_cell_ids = self.get_cells_by_region(roi = roi, units = units)
            filtered_df = filtered_df.query('cell_id in @roi_cell_ids')
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())

        if len(filtered_df) == 0:
            print("No data found for the given noise_ids and cell_types.")
            return
        

        # Remove cells velow minimum threshold
        too_few_cells = [ct for ct in cell_types if len(filtered_df.query(f"typing_file_{typing_file_idx} == @ct")['cell_id'].values) < minimum_n]
        
        for ct in too_few_cells:
            cell_types.remove(ct) 

                
        d_noise_ids_by_type = {ct : list(filtered_df.query(f'typing_file_{typing_file_idx} == @ct')['cell_id'].values) for ct in cell_types}
        d_ells_by_type, scale_factor = get_ells(self, d_noise_ids_by_type, std_scaling = std_scaling, units = units)


        rows = int(np.ceil(len(cell_types)/4))
        cols = np.min([(len(cell_types)-1 % 4)+1, 4])
        size = (4*cols, int(3*rows))

        fig, axs = plt.subplots(nrows = rows, ncols = cols, figsize = size)

        if cols != 1:
            axs = np.array(axs).flatten()
        else:
            axs = np.array([axs])

        for idx, ct in enumerate(cell_types):
            ax = axs[idx]
            for id in d_ells_by_type[ct]:
                ax.add_patch(d_ells_by_type[ct][id])
                if label_cells:
                    ax.text(d_ells_by_type[ct][id].center[0], d_ells_by_type[ct][id].center[1], str(id),
                            horizontalalignment = 'center', verticalalignment = 'center')

            ax.set_xlim(0,self.numXChecks * scale_factor)
            ax.set_ylim(0,self.numYChecks * scale_factor)

            ax.set_ylabel(units.lower())
            ax.set_xlabel(units.lower())

            n_cells = len(d_ells_by_type[ct])
            ax.set_title(f"{ct}, (n = {n_cells})")

        # Remove extra empty axes 
        num_axes = rows * cols
        empty_axes = num_axes - len(cell_types)

        for i in range(empty_axes):
            fig.delaxes(cast(Axes, axs[num_axes - 1 - i]))

        if b_zoom:
            x_min, x_max = filtered_df['center_x'].min(), filtered_df['center_x'].max()
            y_min, y_max = filtered_df['center_y'].min(), filtered_df['center_y'].max()
            
            for ax in axs:
                ax.set_xlim((x_min - n_pad)*scale_factor, (x_max + n_pad)*scale_factor)
                ax.set_ylim((y_min - n_pad)*scale_factor, (y_max + n_pad)*scale_factor)
        
        fig.suptitle("RFs by Cell Type", fontsize = 15)
        fig.tight_layout()

        return axs
        
    def plot_timecourses(self, noise_ids: Optional[List[int]]=None, cell_types: Optional[List[str]]=None,
                         typing_file: Optional[str] = None, units: str = 'ms', std_scaling: float = 2, minimum_n: int = 1,
                         roi: Optional[Dict[str, float]] = None, roi_units: str = 'pixels') -> Optional[np.ndarray[Any, np.dtype[np.object_]]]:
        """
        Method for plotting the timecourses for a given list of cell ids, cell types, 
        or a union of both. If no cell_ids or cell types are given, the timecourses for
        all cells in the analysis chunk are plotted by type. The mean is plotted as a line
        with a shaded region defined by the standard deviation * std_scaling.
        
        Parameters:
        noise_ids (List[int]):  A list of cell_ids to plot. Default None.
        cell_types (List[str]): A list of cell_type strings, (e.g. ['OnP', 'OffP']). Default None.
        typing_fyle (str):      A typing file name which is used to determine the cell types for any
                                given cell_ids. If none is given, the 0th typing file associated
                                with the analysis chunk is used. Default None.
        units (str):            Units to use when plotting the timecourse. Must be either
                                'ms', 'milliseconds', 's', or 'seconds'. Default 'mss'.
        std_scaling (float):    Factor used to scale the standard deviation used for plotting the
                                shaded region around each timecourse. Default 2
        roi (dict):             roi definition as a dictionary with 4 values. 'x_min',
                                'x_max', 'y_min', 'y_max'. These define the vertical and
                                horizontal lines that define the region of interest
        roi_units (str):        Units to use when defining the region of interest. Must be 'pixels',
                                'microns', or 'stixels'. Default 'pixels'.
        
        Returns:
        axs (axes):             Axes object that contains all of the axes used in the timecourses
                                figure. There will be as many axes as there are cell_types represented
                                in the plot.

        """
        
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
            noise_ids = list(filtered_df['cell_id'].values)
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
        elif noise_ids is None:
            filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} in @cell_types')
            noise_ids = list(filtered_df['cell_id'].values)
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
        elif cell_types is None:
            filtered_df = self.df_cell_params.query(f'cell_id in @noise_ids')
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
        else:
            filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} in @cell_types and cell_id in @noise_ids')
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
        if roi is not None:
            roi_cell_ids = self.get_cells_by_region(roi = roi, units = roi_units)
            filtered_df = filtered_df.query('cell_id in @roi_cell_ids')
            cell_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())

        # Check that we actually have cells to plot
        if len(filtered_df) == 0:
            print("No data found for the given noise_ids and cell_types.")
            return

        # Remove cells velow minimum threshold
        too_few_cells = [ct for ct in cell_types if len(filtered_df.query(f"typing_file_{typing_file_idx} == @ct")['cell_id'].values) < minimum_n]
        
        for ct in too_few_cells:
            cell_types.remove(ct) 


        d_noise_ids_by_type = {ct : filtered_df.query(f'typing_file_{typing_file_idx} == @ct')['cell_id'].values for ct in cell_types}
        d_timecourses_by_type = get_timecourses(self, d_noise_ids_by_type)


        rows = np.ceil(len(cell_types)/4).astype(int)
        cols = np.min([(len(cell_types)-1 % 4)+1, 4])
        size = (4*cols, int(3*rows))

        fig, axs = plt.subplots(nrows = rows, ncols = cols, figsize = size)

        if cols != 1:
            axs = np.array(axs).flatten()
        else:
            axs = np.array([axs])

        for idx, ct in enumerate(cell_types):
            ax = axs[idx]

            time_vals = np.linspace(-491.66,8.33,len(d_timecourses_by_type[ct]['rg_mean']))*scale_factor

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
        num_axes = rows*cols
        empty_axes = num_axes - len(cell_types)

        for i in range(empty_axes):
            fig.delaxes(cast(Axes, axs[num_axes - 1 - i]))

        fig.suptitle("Timecourse by Cell Type", fontsize = 15)
        fig.tight_layout()

        return axs

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
    
    def get_stas(self, noise_ids: Optional[int | List[int]] = None, cell_types: Optional[str | List[str]] = None,
                  typing_file: Optional[str] = None, padded: bool = True, units: str = 'stixels') -> dict:
        """
        Function for loading the STAs of a given list of cell types and/or noise ids. If both are given, will only
        pull the union of the two.
        
        Parameters:
        noise_ids (int or List[int]): list of cell ids, optional, default is None

        cell_types (str or List[str]): list of cell types, optional, default is None
        
        typing_file (str): name of a typing file to use for cell type classification

        padded (bool): Boolean value to indicate if any crop should be removed relative to the actual size of the
        noise frame. Most STAs are cropped for the sake of memory, but it makes them inaccurate relative to the
        stimulus. Default is True.

        units (str): Either 'stixels', 'pixels', or 'microns'. This will scale the STAs to the appropriate units
        using nearest neighbor scaling.
        
        Returns:
        all_stas (numpy array or dict of numpy arrays): numpy array that contains all STAs. If a typing file is given and/or 
        cell type info is available, the output will have cell type information. Otherwise it 
        will not. 
        """

        if isinstance(noise_ids, int):
            noise_ids = [noise_ids]

        if isinstance(cell_types, str):
            cell_types = [cell_types]

        if 'pixels' in units.lower():
            unit_scaling = self.pixels_per_stixel
        elif 'microns' in units.lower():
            unit_scaling = self.microns_per_stixel
        elif 'stixels' in units.lower():
            unit_scaling = 1
        else:
            raise Exception("Units must be 'pixels', 'microns' or 'stixels'")

        
        if noise_ids is None:
            if cell_types is None:
                if not self.typing_files:
                    print('Warning, no typing files exist for this chunk, will not organize cells by type')
                    filtered_df = self.df_cell_params
                    cell_ids = filtered_df['cell_id'].to_numpy()
                    available_types = None
                    typing_file_idx = None
                else:
                    print('WARNING: Loading all STAs... this will take up a huge amount of memory')
                    if typing_file is None:
                        typing_file = self.typing_files[0]

                    typing_file_idx = self.typing_files.index(typing_file)

                    filtered_df = self.df_cell_params
                    cell_ids = filtered_df['cell_id'].to_numpy()
                    available_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
            else:
                if not self.typing_files:
                    raise ValueError('No typing files exist for this chunk, try again without cell type argument')
                else:
                    if typing_file is None:
                        typing_file = self.typing_files[0]
                    
                    typing_file_idx = self.typing_files.index(typing_file)

                    filtered_df = self.df_cell_params.query(f'typing_file_{typing_file_idx} in @cell_types')
                    cell_ids = filtered_df['cell_id'].to_numpy()
                    available_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
        
        else:
            if cell_types is None:
                if not self.typing_files:
                    print('Warning, no typing files exist for this chunk, will not organize cells by type')

                    filtered_df = self.df_cell_params.query('cell_id in @noise_ids')
                    cell_ids = filtered_df['cell_id'].to_numpy()
                    available_types = None
                    typing_file_idx = None
                else:
                    if typing_file is None:
                        typing_file = self.typing_files[0]

                    typing_file_idx = self.typing_files.index(typing_file)

                    filtered_df = self.df_cell_params.query('cell_id in @noise_ids')
                    cell_ids = filtered_df['cell_id'].to_numpy()
                    available_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique())
            else:
                if not self.typing_files:
                    raise ValueError('No typing files exist for this chunk, try again without the cell type')
                else:
                    if typing_file is None:
                        typing_file = self.typing_files[0]
                    
                    typing_file_idx = self.typing_files.index(typing_file)

                    filtered_df = self.df_cell_params.query(f'cell_id in @noise_ids and typing_file_{typing_file_idx} in @cell_types')
                    cell_ids = filtered_df['cell_id'].to_numpy()
                    available_types = sorted(filtered_df[f'typing_file_{typing_file_idx}'].unique()) 

        # Pull STAs and organize by cell id alone, or nested inside a dictionary organized by cell id
        sta_reader = vl.STAReader(os.path.join(ANALYSIS_DIR, self.exp_name, self.chunk_name, self.ss_version), self.ss_version)

        if available_types is None:
            id_dict = dict()
            for cell_id in cell_ids:
                # Pull Raw STA
                data = sta_reader.get_sta_for_cell_id(cell_id)
                sta = np.stack([data.red, data.green, data.blue])
                sta = np.transpose(sta, (3,1,2,0))
                if unit_scaling > 1:
                    sta = zoom(sta, zoom = [1.0, unit_scaling, unit_scaling, 1.0], order = 0)
                id_dict[cell_id] = sta

            d_stas = id_dict
        else:
            ct_dict = dict()
            for ct in available_types:
                ct_ids = filtered_df.query(f'typing_file_{typing_file_idx} == @ct')['cell_id'].to_numpy()
                id_dict = dict()
                for cell_id in ct_ids:
                    data = sta_reader.get_sta_for_cell_id(cell_id)
                    sta = np.stack([data.red, data.green, data.blue])
                    sta = np.transpose(sta, (3,1,2,0))

                    if padded:
                        left_pad = int(self.deltaXChecks)
                        right_pad = int(self.numXChecks - self.staXChecks - self.deltaXChecks)
                        top_pad = int(self.deltaYChecks)
                        bottom_pad = int(self.numYChecks - self.staYChecks - self.deltaYChecks)

                        pad_width_config = [(0, 0), (bottom_pad, top_pad), (left_pad, right_pad), (0, 0)]
                        sta = np.pad(sta, pad_width_config, mode = 'constant', constant_values = 0)

                    if unit_scaling > 1:
                        sta = zoom(sta, zoom = [1.0, unit_scaling, unit_scaling, 1.0], order = 0)
                    id_dict[cell_id] = sta

                ct_dict[ct] = id_dict

            d_stas = ct_dict
                    

        return d_stas

    def plot_stas(self, noise_ids: Optional[int | List[int]] = None, cell_types: Optional[str | List[str]] = None,
                  typing_file: Optional[str] = None, cols: int = 4, padded: bool = False,
                  units: str = 'stixels') -> List[np.ndarray] | List[Axes]:
        """
        Method for plotting STAs for a list of cell ids, a list of cell types, or the union of the two. The
        user also has the option of providing a typing file to use. If there exists a typing file or one
        is provided, the function will plot STAs in separate figures organized by cell type. One figure per
        cell type, one STA per axis, number of axes = number of cells of that cell type.

        Parameters:
        noise_ids (int or List[int]) a single cell id or list of cell ids to be plotted. Optional, default None.

        cell_types (str or List[str]) a single cell type or list of cell types to be plotted. Optional, default None. 

        typing_file (str): The name of a typing file to use for linking cell types to cell ids. If no typing file is given
        typing file [0] from the analysis chunk is used. If there are no typing files, only one figure is plotted, with no
        cell type information.

        cols (int): number of columns to use in the resulting figure(s).

        padded (bool): Boolean value to indicate if any crop should be removed relative to the actual size of the
        noise frame. Most STAs are cropped for the sake of memory, but it makes them inaccurate relative to the
        stimulus. Default value for plotting purposes is False.

        units (str): units to plot the STAs in. Must be stixels, pixels, or microns. Default is stixels. If other units
        are used, the STA is scaled using nearest neighbor interpolation. 

        Returns:
        sta_axes: will return a list of Axes objects (multiple single cells of different types) or a list of numpy arrays
        of axes (multiple cells of multiple types). 

        The function will also plot the results automatically if you're in a jupyter notebook, but it does not call
        plt.show() on the figure.
        """

        d_stas = self.get_stas(noise_ids = noise_ids, cell_types = cell_types,
                                typing_file = typing_file, padded = padded, units = units)

        all_axes = []
        if isinstance(list(d_stas.keys())[0], str):
            # This indicates that the dictionary is organized by cell type
            available_types: List[str] = sorted(list(d_stas.keys()))
            
            for ct_idx, ct in enumerate(available_types):
                cell_ids = list(d_stas[ct].keys())
                
                rows = np.ceil(len(cell_ids)/cols).astype(int)
                if len(cell_ids) > 1:
                    fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (5*cols, 3.5*rows),
                                           layout = 'constrained')
                    ax = ax.flatten()
                else:
                    fig, ax = plt.subplots(figsize = (5, 3.5), layout = 'constrained')
                    ax = [ax]

                for c_idx, cell_id in enumerate(cell_ids):
                    sta = d_stas[ct][cell_id]
                    min_index = np.unravel_index(np.argmin(sta), sta.shape) 
                    max_index = np.unravel_index(np.argmax(sta), sta.shape) 

                    if np.abs(np.min(sta)) > np.abs(np.max(sta)):
                        timebin_to_plot = min_index[0]
                    else:
                        timebin_to_plot = max_index[0]

                    sta_img = (sta[timebin_to_plot, :, :, :]+1)/2
                    ax[c_idx].imshow(sta_img)
                    ax[c_idx].set_title(f'Cell ID #{cell_id}')
                    ax[c_idx].set_xlabel(f'{units}')
                    ax[c_idx].set_ylabel(f'{units}')


                if self.deltaXChecks > 0 and not padded:
                    fig.suptitle(f'Cropped {ct} STAs')
                else:
                    fig.suptitle(f'{ct} STAs')

                if len(cell_ids) > 1:
                    # Delete unused axes
                    num_axes = rows*cols
                    empty_axes = num_axes - len(cell_ids)

                    for i in range(empty_axes):
                        fig.delaxes(ax[num_axes-1-i])

                all_axes.append(ax)

        else:
            assert (isinstance(list(d_stas.keys())[0], int)),\
            (f'The keys in the sta dict should either be ints or strs, yours is a {type(list(d_stas.keys())[0])}')
            # This indicates the sta dictionary is organized by cell id and has no types associated with it.
            cell_ids = list(d_stas.keys())

            rows = np.ceil(len(cell_ids)/cols).astype(int)
            fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (5*cols, 3.5*rows),
                                   layout = 'constrained')
            
            if rows*cols > 1:
                ax = ax.flatten()
            else:
                ax = [ax]

            for c_idx, cell_id in enumerate(cell_ids):
                sta = d_stas[cell_id]
                min_index = np.unravel_index(np.argmin(sta), sta.shape) 
                max_index = np.unravel_index(np.argmax(sta), sta.shape) 

                if np.abs(np.min(sta)) > np.abs(np.max(sta)):
                    timebin_to_plot = min_index[0]
                else:
                    timebin_to_plot = max_index[0]

                sta_img = (sta[timebin_to_plot, :, :, :]+1)/2
                ax[c_idx].imshow(sta_img)
                ax[c_idx].set_title(f'Cell ID #{cell_id}')
                ax[c_idx].set_xlabel('Stixels X')
                ax[c_idx].set_ylabel('Stixels Y')


            if self.deltaXChecks > 0 and not padded:
                fig.suptitle(f'Cropped STAs by Cell ID')
            else:
                fig.suptitle(f'STAs by Cell ID')

            if len(cell_ids)>1:
                # Delete unused axes
                num_axes = rows*cols
                empty_axes = num_axes - len(cell_ids)

                for i in range(empty_axes):
                    fig.delaxes(ax[num_axes-1-i])

            all_axes.append(ax)

        return all_axes

    def export_to_pkl(self, file_path: str):
        d_out = self.__dict__.copy()
        # Pop out vcd
        d_out.pop('vcd')
        with open(file_path, 'wb') as f:
            pickle.dump(d_out, f)
        print(f"AnalysisChunk exported to {file_path}")

