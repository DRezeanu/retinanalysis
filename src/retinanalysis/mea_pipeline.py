import numpy as np
from retinanalysis.response import ResponseBlock
from retinanalysis.stim import StimBlock
from retinanalysis.analysis_chunk import AnalysisChunk
import visionloader as vl
import retinanalysis.vision_utils as vu
import os
from retinanalysis.settings import NAS_ANALYSIS_DIR
from matplotlib.patches import Ellipse 
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class MEAPipeline:

    def __init__(self, stim_block: StimBlock=None, response_block: ResponseBlock=None, analysis_chunk: AnalysisChunk=None, pkl_file: str = None):
        if pkl_file is None:
            if stim_block is None or response_block is None or analysis_chunk is None:
                raise ValueError("Either stim_block, response_block, and analysis_chunk must be provided or pkl_file.")
        else:
            with open(pkl_file, 'rb') as f:
                d_out = pickle.load(f)
            self.__dict__.update(d_out)
            self.stim_block = StimBlock(pkl_file=self.stim_block)
            self.response_block = ResponseBlock(pkl_file=self.response_block)
            self.analysis_chunk = AnalysisChunk(pkl_file=self.analysis_chunk)
            print(f"MEAPipeline loaded from {pkl_file}")
            return
        self.stim_block = stim_block
        self.response_block = response_block
        self.analysis_chunk = analysis_chunk

        self.match_dict = vu.cluster_match(self.analysis_chunk.vcd, self.response_block.vcd)
        
        self.add_matches_to_protocol()
        self.add_types_to_protocol()
    
    def classification_transfer(self, target_chunk: str, ss_version: str = None, input_typing_file: str = None, 
                                output_typing_file: str = 'RA_autoClassification.txt', **kwargs) -> dict:
        
        """Transfer classification between analysis chunk and another noise chunk
        Inputs:
            target_chunk: str such as 'chunk2'
            ss_version: str such as 'kilosort2.5', if None, uses same ss_version as analysis_chunk
            input_typing_file: str, filename of classification file to use, if None will use
                               the first typing file in analysis_chunk.typing_files
            output_typing_file: str, filename of classification file to export, default is
                                RA_autoClassification.txt

        Kwargs to pass to cluster_match:
            use_isi: bool, default = false
            use_timecourse: bool, default = false
            corr_cutoff: float, default = 0.8
            method: str, default = 'full'"""
        
        # Flag if there are no typing files available for this analysis chunk
        if len(self.analysis_chunk.typing_files) == 0:
            raise FileNotFoundError("No typing files available for this analysis chunk")
        
        if target_chunk == self.analysis_chunk.chunk_name:
            raise Exception(f"Target chunk ({target_chunk}) cannot be the same as analysis chunk")

        # If no input typing file is specified, use typing_file_0
        if input_typing_file is None:
            input_typing_file = self.analysis_chunk.typing_files[0] 

        # Flag if input typing file is not actually part of the current analysis chunk
        if input_typing_file not in self.analysis_chunk.typing_files:
            raise FileNotFoundError("Input typing file not found in current chunk")         

        # If no spike sorting version is given, use same ss_version as analysis chunk
        if ss_version is None:
            ss_version = self.analysis_chunk.ss_version
        
        print(f"Cluster matching {self.analysis_chunk.chunk_name} with {target_chunk}\n")
        
        # Cluster Match
        target_vcd = vu.get_analysis_vcd(self.analysis_chunk.exp_name, target_chunk, ss_version)
        target_ids = target_vcd.get_cell_ids()

        match_dict = vu.cluster_match(self.analysis_chunk.vcd, target_vcd, **kwargs)
        
        # Create classification file and drop it in the destination path
        input_file_path = os.path.join(NAS_ANALYSIS_DIR, self.analysis_chunk.exp_name,
                                       self.analysis_chunk.chunk_name, self.analysis_chunk.ss_version,
                                       input_typing_file)
        
        destination_file_path = os.path.join(NAS_ANALYSIS_DIR, self.analysis_chunk.exp_name,
                                             target_chunk, ss_version, output_typing_file)

        
        matched_count = 0
        unmatched_count = 0
        input_classification_dict = vu.create_dictionary_from_file(input_file_path, delimiter = ' ')

        with open(destination_file_path, mode='w') as output_file:
            for key in match_dict.keys():
                matched_count += 1
                print(match_dict[key], input_classification_dict[key], file = output_file)

        partial_output = vu.create_dictionary_from_file(destination_file_path, delimiter = ' ')

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

    def add_matches_to_protocol(self) -> None:
        inverse_match_dict = {val : key for key, val in self.match_dict.items()}
        for id in self.response_block.df_spike_times.index:
            if id in inverse_match_dict:
                pass
            else:
                inverse_match_dict[id] = 0

        self.response_block.df_spike_times['noise_ids'] = inverse_match_dict

    def add_types_to_protocol(self, typing_file: str = None) -> None:

        if typing_file is None:
            typing_file = 0
        else:
            try:
                typing_file = self.analysis_chunk.typing_files.index(typing_file)
            except:
                raise FileNotFoundError(f"{typing_file} Not Found in Analysis Chunk")
        
        type_dict = dict()
        for id in self.analysis_chunk.df_cell_params['cell_id']:
            if id in self.match_dict:
                type_dict[self.match_dict[id]] = self.analysis_chunk.df_cell_params.query('cell_id == @id')[f'typing_file_{typing_file}'].values[0]
        
        for id in self.response_block.df_spike_times.index:
            if id in type_dict:
                pass
            else:
                type_dict[id] = "Unmatched"

        self.response_block.df_spike_times['cell_type'] = type_dict

    def plot_rfs(self, protocol_ids: List[int] = None, cell_types: List[str] = None,
                 std_scaling: float = 1.6, units: str = 'pixels') -> np.ndarray:
        
        d_ells_by_type = self.get_noise_ids(protocol_ids, cell_types)
        d_ells_by_type, scale_factor = self.get_ells(d_ells_by_type, std_scaling = std_scaling, units = units)

        rows = int(np.ceil(len(d_ells_by_type.keys())/4))
        cols = np.min([(len(d_ells_by_type.keys())-1 % 4)+1, 4])
        size = (4.5*cols, int(3*rows))

        fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = size)

        if cols != 1:
            ax = ax.flatten()

        for idx, ct in enumerate(d_ells_by_type.keys()):

            if cols != 1:
                for id in d_ells_by_type[ct]:
                    ax[idx].add_patch(d_ells_by_type[ct][id])

                ax[idx].set_xlim(0,self.analysis_chunk.numXChecks * scale_factor)
                ax[idx].set_ylim(0,self.analysis_chunk.numYChecks * scale_factor)

                ax[idx].set_ylabel(units.lower())
                ax[idx].set_xlabel(units.lower())
                
                ax[idx].set_title(ct)

            else: 
                for id in d_ells_by_type[ct]:
                    ax.add_patch(d_ells_by_type[ct][id])

                ax.set_xlim(0,self.analysis_chunk.numXChecks * scale_factor)
                ax.set_ylim(0,self.analysis_chunk.numYChecks * scale_factor)

                ax.set_ylabel(units.lower())
                ax.set_xlabel(units.lower())

                ax.set_title(ct)

        # Remove extra empty axes 
        num_axes = (rows-1)*4 + cols
        empty_axes = num_axes - len(d_ells_by_type.keys())

        for i in range(empty_axes):
            fig.delaxes(ax[num_axes - 1 - i])

        fig.suptitle("RFs by Cell Type", fontsize = 15)
        fig.tight_layout()

        return ax
        
    def get_ells(self, d_cells_by_type: dict, std_scaling: float = 1.6, units: str = 'pixels') -> Tuple[Dict[str, dict], int]:

        if 'microns' in units.lower():
            scale_factor = self.analysis_chunk.microns_per_stixel
        elif 'pixels' in units.lower():
            scale_factor = self.analysis_chunk.pixels_per_stixel
        elif 'stixels' in units.lower():
            scale_factor = 1
        else:
            raise NameError("Units string must be 'microns', 'pixels' or 'stixels'.")
        
        rf_params = self.analysis_chunk.rf_params

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

    def plot_timecourses(self, protocol_ids: List[int] = None, cell_types: List[str] = None, 
                        units: str = 'ms', std_scaling: float = 2) -> np.ndarray:
        
        if 'ms' in units.lower() or 'milliseconds' in units.lower():
            scale_factor = 1
        elif 's' in units.lower() or 'seconds' in units.lower():
            scale_factor = 1e-3
        else:
            raise NameError("Units string must be 'ms', 'milliseconds', 's' or 'seconds'")

        d_noise_ids_by_type = self.get_noise_ids(protocol_ids, cell_types)
        d_timecourses_by_type = self.get_timecourses(d_noise_ids_by_type)


        rows = int(np.ceil(len(d_timecourses_by_type.keys())/4))
        cols = np.min([(len(d_timecourses_by_type.keys())-1 % 4)+1, 4])
        size = (4.5*cols, int(3*rows))

        fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = size)

        if cols != 1:
            ax = ax.flatten()

        for idx, ct in enumerate(d_timecourses_by_type.keys()):

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
        empty_axes = num_axes - len(d_timecourses_by_type.keys())

        for i in range(empty_axes):
            fig.delaxes(ax[num_axes - 1 - i])

        fig.suptitle("Timecourse by Cell Type", fontsize = 15)
        fig.tight_layout()

        return ax

    def get_timecourses(self, d_cells_by_type: dict) -> Dict[str, dict]: 

        d_timecourses_by_type = dict()

        for ct in d_cells_by_type.keys():

            rg_timecourses = [self.analysis_chunk.vcd.main_datatable[cell]['GreenTimeCourse'] for cell in d_cells_by_type[ct]]
            rg_timecourses = np.array(rg_timecourses)
            rg_mean = np.mean(rg_timecourses, axis = 0)
            rg_std = np.std(rg_timecourses, axis = 0)

            b_timecourses = [self.analysis_chunk.vcd.main_datatable[cell]['BlueTimeCourse'] for cell in d_cells_by_type[ct]]
            b_timecourses = np.array(b_timecourses)
            b_mean = np.mean(b_timecourses, axis = 0)
            b_std = np.std(b_timecourses, axis = 0)

            d_timecourses_by_type[ct] = {'rg_timecourses' : rg_timecourses, 'rg_mean' : rg_mean, 'rg_std' : rg_std,
                                'b_timecourses' : b_timecourses, 'b_mean' : b_mean, 'b_std' : b_std}

        return d_timecourses_by_type

    # Helper function for pulling noise ids for plotting and organizing them into a dictionary
    # by type. IDs can be pulled by list of protocol ids, list of cell types, or both. Used
    # in plot_rfs and plot_timecourse
    def get_noise_ids(self, protocol_ids: List[int] = None, cell_types: List[int] = None) -> Dict[str,list]:

        d_cells_by_type = dict()
        # Pull analysis_block ids that match the input cell_ids and cell_types
        # If neither is given, plot all matched ids
        if protocol_ids is None and cell_types is None:
            cell_types = list(self.response_block.df_spike_times['cell_type'].unique())
            for ct in cell_types:
                type_ids = self.response_block.df_spike_times.query('cell_type == @ct').index.values
                d_cells_by_type[ct] = [key for key, val in self.match_dict.items() if val in type_ids]

                # remove empty keys
                if not d_cells_by_type[ct]:
                    d_cells_by_type.pop(ct, None)

        # If only type is given, pull only ids that correspond to that type
        elif protocol_ids is None:
            d_cells_by_type = dict()
            for ct in cell_types:
                protocol_ids = self.response_block.df_spike_times.query('cell_type == @ct').index.values
                d_cells_by_type[ct] = [key for key, val in self.match_dict.items() if val in protocol_ids]

        # If only ids are given, pull all ids regardless of type
        elif cell_types is None:
            cell_types = list(self.response_block.df_spike_times['cell_type'].unique())
            for ct in cell_types:
                type_ids = self.response_block.df_spike_times.query('cell_type == @ct').index.values
                d_cells_by_type[ct] = [key for key, val in self.match_dict.items() if (val in protocol_ids
                                                                            and val in type_ids)]
                # remove empty keys
                if not d_cells_by_type[ct]:
                    d_cells_by_type.pop(ct, None) 

        # If both are given, pull only ids that match both the cell types and the cell ids given
        else:
            for ct in cell_types:
                protocol_ids = self.response_block.df_spike_times.query('cell_type ==  @ct').index.values
                d_cells_by_type[ct] = [key for key, val in self.match_dict.items() if (val in protocol_ids
                                                                    and val in protocol_ids)]


        if not d_cells_by_type:
            raise Exception("No cluster matched ids found for given list of cell ids and/or cell types") 

        return d_cells_by_type


    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  stim_block and response_block from: {os.path.splitext(self.stim_block.protocol_name)[1][1:]}\n"
        str_self += f"  analysis_chunk: {self.analysis_chunk.chunk_name}\n"
        str_self += f"  match_dict: with {self.analysis_chunk.chunk_name}_id : {os.path.splitext(self.stim_block.protocol_name)[1][1:]}_id"
        return str_self

    def export_to_pkl(self, file_path: str):
        """
        Export the MEAPipeline to a pickle file.
        """
        d_out = self.__dict__.copy()
        # For StimBlock, ResponseBlock, and AnalysisChunk, get only the __dict__ attribute
        d_out['stim_block'] = self.stim_block.__dict__
        d_out['response_block'] = self.response_block.__dict__
        d_out['analysis_chunk'] = self.analysis_chunk.__dict__
        # Pop out vcd from response_block and analysis_chunk
        d_out['response_block'].pop('vcd', None)
        d_out['analysis_chunk'].pop('vcd', None)
        with open(file_path, 'wb') as f:
            import pickle
            pickle.dump(d_out, f)
        print(f"MEAPipeline exported to {file_path}")

