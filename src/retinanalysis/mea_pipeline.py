import numpy as np
from retinanalysis.response import ResponseBlock
from retinanalysis.stim import StimBlock
from retinanalysis.analysis_chunk import AnalysisChunk
import visionloader as vl
import retinanalysis.vision_utils as vu
import os
from retinanalysis.settings import NAS_ANALYSIS_DIR


class MEAPipeline:

    def __init__(self, stim_block: StimBlock, response_block: ResponseBlock, analysis_chunk: AnalysisChunk):
        self.stim_block = stim_block
        self.response_block = response_block
        self.analysis_chunk = analysis_chunk

        self.match_dict = vu.cluster_match(self.analysis_chunk.vcd, self.response_block.vcd,
                                           use_isi = False)
        
        self.add_types_to_protocol()
    
    def classification_transfer(self, target_chunk: str, ss_version: str = None, input_typing_file: str = None, 
                                output_typing_file: str = 'RA_autoClassification.txt', **kwargs):
        
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
            use_rgb: bool, default = false
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
                    print(id, 'All/unmatched', file = output_file)
                    unmatched_count += 1

        print(f"\nTarget clusters matched: {matched_count}\nTarget clusters unmatched: {unmatched_count}\n")
        print(f"Classification file {output_typing_file} created at: {destination_file_path}")

        return match_dict

    def add_types_to_protocol(self, typing_file: str = None):

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

        type_dict_sorted = dict(sorted(type_dict.items()))

        self.response_block.df_spike_times['cell_type'] = type_dict_sorted

    def plot_rfs(self, cell_ids: list[int] = None, cell_types: list[str] = None) -> dict:
        ids_to_plot = dict()
        # Pull analysis_block ids that match the input cell_ids and cell_types
        # If neither is given, plot all matched ids
        if cell_ids is None and cell_types is None:
            cell_types = list(self.response_block.df_spike_times['cell_type'].unique())
            cell_types.remove('Unmatched')
            for ct in cell_types:
                type_ids = self.response_block.df_spike_times.query('cell_type == @ct').index.values
                ids_to_plot[ct] = [key for key, val in self.match_dict.items() if val in type_ids]
        
        # If only type is given, pull only ids that correspond to that type
        elif cell_ids is None:
            ids_to_plot = dict()
            for ct in cell_types:
                protocol_ids = self.response_block.df_spike_times.query('cell_type == @ct').index.values
                ids_to_plot[ct] = [key for key, val in self.match_dict.items() if val in protocol_ids]
        
        # If only ids are given, pull all ids regardless of type
        elif cell_types is None:
            cell_types = list(self.response_block.df_spike_times['cell_type'].unique())
            cell_types.remove('Unmatched')
            for ct in cell_types:
                type_ids = self.response_block.df_spike_times.query('cell_type == @ct').index.values
                ids_to_plot[ct] = [key for key, val in self.match_dict if (val in cell_ids
                                                                           and val in type_ids)]

        # If both are given, pull only ids that match both the cell types and the cell ids given
        else:
            for ct in cell_types:
                protocol_ids = self.response_block.df_spike_times.query('cell_type ==  @ct').index.values
                ids_to_plot[ct] = [key for key, val in self.match_dict.items() if (val in protocol_ids
                                                                    and val in cell_ids)]
            
    # Ideas: 
    # - Plot time courses in addition to RFs. 
    # - Add frame times to stim
    # - Function that bins spike times according to frame
    # - IN QC or in Pipeline, function to visualize the EIs/electrode maps/raw voltage traces
    # - Stim Regen
    # - Plot RFs by ROI
    # - Stim mask that's the gaussian center fit

    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  stim_block and response_block from: {os.path.splitext(self.stim_block.protocol_name)[1][1:]}\n"
        str_self += f"  analysis_chunk: {self.analysis_chunk.chunk_name}\n"
        str_self += f"  match_dict: with {self.analysis_chunk.chunk_name}_id : {os.path.splitext(self.stim_block.protocol_name)[1][1:]}_id"
        return str_self

