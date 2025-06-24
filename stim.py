import schema
import numpy as np
import utils.vision_utils as vu
import utils.datajoint_utils as dju
import datajoint as dj
import os
import pandas as pd
import json
import scipy.io as scipy


class StimData:

    def __init__(self, exp_name, ks_version: str = 'kilosort2.5'):
        self.exp_name = exp_name
        self.ks_version = ks_version
        exp_id = schema.Experiment() & {'exp_name' : self.exp_name}
        self.exp_id = exp_id.fetch('id')[0]

class NoiseStimData(StimData):

    def __init__(self, exp_name: str, chunk_name: str, ks_version: str = 'kilosort2.5'):
        super().__init__(exp_name, ks_version)
        
        self.chunk_name = chunk_name
        
        # Get chunk ID
        chunk_id = schema.SortingChunk() & {'experiment_id': self.exp_id, 'chunk_name' : self.chunk_name}
        self.chunk_id = chunk_id.fetch('id')[0]

        # Get protocol name and ID
        protocol = schema.Protocol() & 'name LIKE "%.SpatialNoise"'
        self.protocol_name = protocol.fetch('name')[0]
        self.protocol_id = protocol.fetch('protocol_id')[0]

        self.get_noise_params()

    def get_noise_params(self):

        vcd = vu.get_vcd(self.exp_name, self.chunk_name, self.ks_version,
                         ei = False, params = False)
        self.staXChecks = int(vcd.runtimemovie_params.width)
        self.staYChecks = int(vcd.runtimemovie_params.height)

        # Pull epoch block and epoch to get num X and num Y checks used in noise
        epoch_block = schema.EpochBlock() & {'experiment_id' : self.exp_id, 'chunk_id' : self.chunk_id, 'protocol_id' : self.protocol_id}
        epoch_block_id = epoch_block.fetch('id')[0]
        epoch = schema.Epoch() & {'experiment_id' : self.exp_id, 'parent_id' : epoch_block_id}

        self.numXChecks = epoch.fetch('parameters')[0]['numXChecks']
        self.numYChecks = epoch.fetch('parameters')[0]['numYChecks']
        
        self.microns_per_pixel = epoch.fetch('parameters')[0]['micronsPerPixel']
        self.canvas_size = epoch.fetch('parameters')[0]['canvasSize']

        # Pull noise data file names
        noise_data_dirs = epoch_block.fetch('data_dir')
        self.data_files = [os.path.basename(path) for path in noise_data_dirs]

        self.sorting_files = schema.CellTypeFile() & {'chunk_id' : self.chunk_id}
        self.pixels_per_stixel = self.canvas_size[0]/self.numXChecks
        self.microns_per_stixel = self.microns_per_pixel * self.pixels_per_stixel

        self.deltaXChecks = int((self.numXChecks - self.staXChecks)/2)
        self.deltaYChecks = int((self.numYChecks - self.staYChecks)/2)
    
