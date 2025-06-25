import schema
import numpy as np
import utils.vision_utils as vu
import utils.datajoint_utils as dju
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

    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  exp_name: {self.exp_name}\n"
        str_self += f"  datafile_name: {self.datafile_name}\n"
        str_self += f"  chunk_name: {self.d_block_summary['chunk_name']}\n"
        str_self += f"  protocol_name: {self.d_block_summary['protocol_name']}\n"
        str_self += f"  noise_protocol_name: {self.noise_protocol_name}\n"
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

def make_stim_group(ls_exp_names, ls_datafile_names, ls_params: list=None):
    ls_blocks = []
    for exp_name, datafile_name in zip(ls_exp_names, ls_datafile_names):
        block = StimBlock(exp_name, datafile_name, ls_params=ls_params)
        ls_blocks.append(block)
    return StimGroup(ls_blocks)

# class NoiseStim(Stim):
#     def __init__(self, exp_name: str, datafile_name: str, ss_version: str = 'kilosort2.5'):
#         super().__init__(exp_name, datafile_name)
#         self.ss_version = ss_version

#         # Get chunk ID
#         chunk_id = schema.SortingChunk() & {'experiment_id': self.exp_id, 'chunk_name' : self.chunk_name}
#         self.chunk_id = chunk_id.fetch1('id')

#         # Get protocol ID
#         protocol = schema.Protocol() & {'name' : self.noise_protocol_name}
#         self.protocol_id = protocol.fetch1('protocol_id')
#         self.get_noise_params()

#     def get_noise_params(self):

#         vcd = vu.get_vcd(self.exp_name, self.chunk_name, self.ss_version,
#                          ei = False, params = False)
#         self.staXChecks = int(vcd.runtimemovie_params.width)
#         self.staYChecks = int(vcd.runtimemovie_params.height)

#         # Pull epoch block and epoch to get num X and num Y checks used in noise
#         epoch_block = schema.EpochBlock() & {'experiment_id' : self.exp_id, 'chunk_id' : self.chunk_id, 'protocol_id' : self.protocol_id}
#         epoch_block_id = epoch_block.fetch('id')[0]
#         epoch = schema.Epoch() & {'experiment_id' : self.exp_id, 'parent_id' : epoch_block_id}

#         self.numXChecks = epoch.fetch('parameters')[0]['numXChecks']
#         self.numYChecks = epoch.fetch('parameters')[0]['numYChecks']
        
#         self.microns_per_pixel = epoch.fetch('parameters')[0]['micronsPerPixel']
#         self.canvas_size = epoch.fetch('parameters')[0]['canvasSize']

#         # Pull noise data file names
#         noise_data_dirs = epoch_block.fetch('data_dir')
#         self.data_files = [os.path.basename(path) for path in noise_data_dirs]

#         self.sorting_files = schema.CellTypeFile() & {'chunk_id' : self.chunk_id}
#         self.pixels_per_stixel = self.canvas_size[0]/self.numXChecks
#         self.microns_per_stixel = self.microns_per_pixel * self.pixels_per_stixel

#         self.deltaXChecks = int((self.numXChecks - self.staXChecks)/2)
#         self.deltaYChecks = int((self.numYChecks - self.staYChecks)/2)
    
#     def __repr__(self):
#         str_self = super().__repr__()
#         str_self += f"  staXChecks: {self.staXChecks}\n"
#         str_self += f"  staYChecks: {self.staYChecks}\n"
#         str_self += f"  numXChecks: {self.numXChecks}\n"
#         str_self += f"  numYChecks: {self.numYChecks}\n"
#         str_self += f"  microns_per_pixel: {self.microns_per_pixel}\n"
#         str_self += f"  canvas_size: {self.canvas_size}\n"
#         str_self += f"  data_files: {self.data_files}\n"
#         return str_self
