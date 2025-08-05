import numpy as np
from retinanalysis.classes.response import MEAResponseBlock
from retinanalysis.classes.stim import MEAStimBlock
from retinanalysis.classes.analysis_chunk import AnalysisChunk
from retinanalysis.utils.vision_utils import cluster_match
import os
from typing import (List,
                    Dict)
import pickle


class MEAPipeline:

    def __init__(self, stim_block: MEAStimBlock=None, response_block: MEAResponseBlock=None, analysis_chunk: AnalysisChunk=None, pkl_file: str = None):
        if pkl_file is None:
            if stim_block is None or response_block is None or analysis_chunk is None:
                raise ValueError("Either stim_block, response_block, and analysis_chunk must be provided or pkl_file.")
        else:
            with open(pkl_file, 'rb') as f:
                d_out = pickle.load(f)
            self.__dict__.update(d_out)
            self.stim_block = MEAStimBlock(pkl_file=self.stim_block)
            self.response_block = MEAResponseBlock(pkl_file=self.response_block)
            self.analysis_chunk = AnalysisChunk(pkl_file=self.analysis_chunk)
            print(f"MEAPipeline loaded from {pkl_file}")
            return
        
        self.stim_block = stim_block
        self.response_block = response_block
        self.analysis_chunk = analysis_chunk

        self.match_dict, self.corr_dict = cluster_match(self.analysis_chunk, self.response_block)
        
        self.add_matches_to_protocol()
        self.add_types_to_protocol()
    

    def add_matches_to_protocol(self) -> None:
        inverse_match_dict = {val : key for key, val in self.match_dict.items()}
        for id in self.response_block.df_spike_times['cell_id']:
            if id in inverse_match_dict:
                pass
            else:
                inverse_match_dict[id] = 0

        for idx, id in enumerate(self.response_block.df_spike_times['cell_id'].values):
            self.response_block.df_spike_times.at[idx, 'noise_id'] = inverse_match_dict[id]
        
        self.response_block.df_spike_times['noise_id'] = self.response_block.df_spike_times['noise_id'].astype(int)

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
        
        for id in self.response_block.df_spike_times['cell_id']:
            if id in type_dict:
                pass
            else:
                type_dict[id] = "Unmatched"

        for idx, id in enumerate(self.response_block.df_spike_times['cell_id'].values):
            self.response_block.df_spike_times.at[idx, 'cell_type'] = type_dict[id]

    def plot_rfs(self, protocol_ids: List[int] = None, cell_types: List[str] = None,
                 **kwargs) -> np.ndarray:
        
        noise_ids = self.get_noise_ids(protocol_ids, cell_types)
        ax = self.analysis_chunk.plot_rfs(noise_ids, cell_types = cell_types,
                                          **kwargs)

        return ax
    
    def get_cells_by_region(self, roi: Dict[str, float], units: str = 'pixels'):

        noise_ids = self.analysis_chunk.get_cells_by_region(roi = roi, units = units)
        protocol_ids = [val for key, val in self.match_dict.items() if key in noise_ids]
        arr_ids = np.array(protocol_ids)
        
        return arr_ids


    def plot_timecourses(self, protocol_ids: List[int] = None, cell_types: List[str] = None, 
                        **kwargs) -> np.ndarray:

        noise_ids = self.get_noise_ids(protocol_ids, cell_types)
        ax = self.analysis_chunk.plot_timecourses(noise_ids, cell_types = cell_types, 
                                             **kwargs)
        
        return ax

    

    # Helper function for pulling noise ids for plotting and organizing them into a dictionary
    # by type. IDs can be pulled by list of protocol ids, list of cell types, or both. Used
    # in plot_rfs and plot_timecourse
    def get_noise_ids(self, protocol_ids: List[int] = None, cell_types: List[int] = None) -> List[int]:

        # Pull analysis_block ids that match the input cell_ids and cell_types
        # If neither is given, plot all matched ids
        if protocol_ids is None and cell_types is None:
            protocol_ids = self.response_block.df_spike_times['cell_id'].values
            noise_ids = [key for key, val in self.match_dict.items() if val in protocol_ids]

        # If only type is given, pull only ids that correspond to that type
        elif protocol_ids is None:
            protocol_ids = self.response_block.df_spike_times.query('cell_type in @cell_types')['cell_id'].values
            noise_ids = [key for key, val in self.match_dict.items() if val in protocol_ids]

        # If only ids are given, pull all ids regardless of type
        elif cell_types is None:
            noise_ids = [key for key, val in self.match_dict.items() if val in protocol_ids]

        # If both are given, pull only ids that match both the cell types and the cell ids given
        else:
            protocol_ids = self.response_block.df_spike_times.query('cell_type in @cell_types')['cell_id'].values
            noise_ids = [key for key, val in self.match_dict.items() if val in protocol_ids]

        if len(noise_ids) == 0:
            raise Exception("No cluster matched ids found for given list of cell ids and/or cell types") 

        return noise_ids

    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  stim_block and response_block from: {os.path.splitext(self.stim_block.protocol_name)[1][1:]}\n"
        str_self += f"  analysis_chunk: {self.analysis_chunk.chunk_name}\n"
        str_self += f"  match_dict: with {self.analysis_chunk.chunk_name}_id : {os.path.splitext(self.stim_block.protocol_name)[1][1:]}_id"
        str_self += f"  corr_dict: with {self.analysis_chunk.chunk_name}_id : calculated ei correlations"
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

def create_mea_pipeline(exp_name: str, datafile_name: str, analysis_chunk_name: str=None,
                    ss_version: str='kilosort2.5', ls_params: list=None):
    # Helper function for initializing MEAPipeline from metadata
    # TODO StimGroup and ResponseGroup functionality
    s = MEAStimBlock(exp_name, datafile_name, ls_params)
    r = MEAResponseBlock(exp_name, datafile_name, ss_version)
    if analysis_chunk_name is None:
        analysis_chunk_name = s.nearest_noise_chunk
        print(f'Using {analysis_chunk_name} for AnalysisChunk\n')
    ac = AnalysisChunk(exp_name, analysis_chunk_name, ss_version)
    mp = MEAPipeline(s, r, ac)
    return mp