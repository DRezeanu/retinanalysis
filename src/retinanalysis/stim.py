import schema
import numpy as np
import src.retinanalysis.vision_utils as vu
import src.retinanalysis.datajoint_utils as dju
import datajoint as dj
import os
import pandas as pd
from typing import List

class StimBlock:
    def __init__(self, exp_name, datafile_name, ls_params: list=None):
        self.exp_name = exp_name
        self.datafile_name = datafile_name
        
        df = dju.get_mea_exp_summary(exp_name)
        self.d_block_summary = df[df['datafile_name'] == datafile_name].iloc[0].to_dict()

        df_e = dju.get_mea_epoch_data_from_exp(exp_name, datafile_name, ls_params=ls_params)
        self.df_epochs = df_e
        self.parameter_names = list(df_e.at[0,'epoch_parameters'].keys())

        # We switched from FastNoise to SpatialNoise after 20230926
        if int(exp_name[:8]) < 20230926:
            self.noise_protocol_name = 'manookinlab.protocols.FastNoise'
        else:
            self.noise_protocol_name = 'manookinlab.protocols.SpatialNoise'

        self.nearest_noise_chunk = self.get_nearest_noise()
    
    def get_nearest_noise(self):
        # pull relevant information from datajoint
        experiment_summary = dju.get_mea_exp_summary(self.exp_name)
        exp_id = schema.Experiment() & {'exp_name' : self.exp_name}
        exp_id = exp_id.fetch('id')[0]

        # Pull noise runs and target protocol run from experiment summary df
        noise_runs = experiment_summary.query('protocol_name == @self.noise_protocol_name and chunk_name.str.contains("chunk")')
        target_run = experiment_summary.query('protocol_name == @self.d_block_summary["protocol_name"] and datafile_name == @self.datafile_name')

        # Identify start and stop points for target and noise runs
        target_run_stop = target_run['minutes_since_start']
        target_run_start = target_run_stop-target_run['duration_minutes']
        
        noise_run_stop = noise_runs['minutes_since_start']
        noise_run_start = noise_run_stop-noise_runs['duration_minutes']
        
        # Calculate distance from noise start to target stop, and noise stop to target start
        protocolstop_to_noisestart = abs(noise_run_start.values - target_run_stop.values)
        protocolstart_to_noisestop = abs(noise_run_stop.values - target_run_start.values)

        # Find the minimum distance between target protocol and each chunk
        minimum_distance = np.minimum(protocolstart_to_noisestop, protocolstop_to_noisestart)
        
        # Iterate through minimum distances until we find the nearest chunk with a sorting file
        for distance in minimum_distance:

            # Use min val to pull the nearest noise chunk
            min_val = min(minimum_distance)
            nearest_noise_chunk = noise_runs[(protocolstart_to_noisestop == min_val)]
            if nearest_noise_chunk.empty:
                nearest_noise_chunk = noise_runs[(protocolstop_to_noisestart == min(minimum_distance))]

            nearest_noise_chunk = nearest_noise_chunk.reset_index(drop = True).loc[0, 'chunk_name']

            # Check if this chunk has a typing file
            noise_chunk_id = schema.SortingChunk() & {'experiment_id' : exp_id, 'chunk_name': nearest_noise_chunk}
            noise_chunk_id = noise_chunk_id.fetch('id')[0]

            typing_files = schema.CellTypeFile() & {'chunk_id' : noise_chunk_id}

            # If there's no typing file, remove the minimum value and try again
            if len(typing_files) == 0:
                min_index = np.argmin(minimum_distance)
                minimum_distance = np.delete(minimum_distance, min_index)
            # If there is a typing file, break the for loop there
            else:
                break
        
        # Check if we looped through all the values. If so, no sorting files found
        if minimum_distance.size == 0:
            print("Warning, none of the noise chunks in this experiment have typing files\n")

        return nearest_noise_chunk

    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  datafile_name: {self.datafile_name}\n"
        str_self += f"  chunk_name: {self.d_block_summary['chunk_name']}\n"
        str_self += f"  protocol_name: {self.d_block_summary['protocol_name']}\n"
        str_self += f"  noise_protocol_name: {self.noise_protocol_name}\n"
        str_self += f"  nearest_noise_chunk: {self.nearest_noise_chunk}\n"
        str_self += f"  parameter_names of length: {len(self.parameter_names)}\n"
        str_self += f"  df_epochs for {self.df_epochs.shape[0]} epochs\n"
        return str_self

class StimGroup:
    def __init__(self, ls_blocks: List[StimBlock]):
        # Check that all StimBlocks have same exp_name, protocol_name, and chunk_name
        if not all(block.exp_name == ls_blocks[0].exp_name for block in ls_blocks):
            raise ValueError("All StimBlocks must have the same exp_name")
        if not all(block.d_block_summary['protocol_name'] == ls_blocks[0].d_block_summary['protocol_name'] for block in ls_blocks):
            raise ValueError("All StimBlocks must have the same protocol_name")
        if not all(block.d_block_summary['chunk_name'] == ls_blocks[0].d_block_summary['chunk_name'] for block in ls_blocks):
            raise ValueError("All StimBlocks must have the same chunk_name")

        datafile_names = [block.datafile_name for block in ls_blocks]
        if len(set(datafile_names)) != len(datafile_names):
            raise ValueError(f"StimBlocks must have unique datafile_names, but found: {datafile_names}")
        
        self.ls_blocks = ls_blocks
        self.exp_name = ls_blocks[0].exp_name
        self.parameter_names = ls_blocks[0].parameter_names
        self.datafile_names = datafile_names
        self.df_epochs = pd.concat([block.df_epochs for block in ls_blocks], ignore_index=True)
        self.df_epochs.index = self.df_epochs.index.rename('epoch_index')
    
    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  protocol_name: {self.ls_blocks[0].d_block_summary['protocol_name']}\n"
        str_self += f"  chunk_name: {self.ls_blocks[0].d_block_summary['chunk_name']}\n"
        str_self += f"  datafile_names: {self.datafile_names}\n"
        str_self += f"  parameter_names of length: {len(self.parameter_names)}\n"
        str_self += f"  df_epochs for {self.df_epochs.shape[0]} epochs\n"
        return str_self

def make_stim_group(exp_name, ls_datafile_names, ls_params: list=None):
    ls_blocks = []
    for datafile_name in ls_datafile_names:
        block = StimBlock(exp_name, datafile_name, ls_params=ls_params)
        ls_blocks.append(block)
    return StimGroup(ls_blocks)