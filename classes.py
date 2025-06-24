import schema
import numpy as np
import utilities.dr_utils as dr
import utilities.vr_utils as vr
import datajoint as dj
import os
import pandas as pd
import json
import scipy.io as scipy

class SortingChunk:

    def __init__(self, exp_name, ks_version: str = 'kilosort2.5'):
        self.exp_name = exp_name
        self.ks_version = ks_version
        exp_id = schema.Experiment() & {'exp_name' : self.exp_name}
        self.exp_id = exp_id.fetch('id')[0]

class NoiseChunk(SortingChunk):

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

        # Get vision data 
        self.vcd = dr.get_vcd(self.exp_name, self.chunk_name, self.ks_version)

        # Get X and Y checks used to calculate STA
        self.staXChecks = int(self.vcd.runtimemovie_params.width)
        self.staYChecks = int(self.vcd.runtimemovie_params.height)

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

        self.cell_ids = self.vcd.get_cell_ids()
    
    def get_ells(self):
        self.ell_params = dict()

        for id in self.cell_ids:
            sta_fit = self.vcd.get_stafit_for_cell(id)
            self.ell_params[id] = {'center_x' : sta_fit.center_x + self.deltaXChecks, 
                                   'center_y' : sta_fit.center_y + self.deltaYChecks,
                                   'std_x' : sta_fit.std_x,
                                   'std_y' : sta_fit.std_y,
                                   'rot' : sta_fit.rot}
            
        return self.ell_params


class ProtocolChunk(SortingChunk):

    def __init__(self, exp_name, ks_version:str = 'kilosort2.5'):
        super().__init__(exp_name, ks_version)


        











# Analysis Class in Progress
class LetterAnalysis:

    def __init__(self, exp_name: str, ks_version: str = 'kilosort2.5'):

        self.exp_name = exp_name
        self.ks_version = ks_version

        self.cluster_matching_file = None
        self.match_dict = None
        self.classification_file_path = os.path.abspath(f'assets/classification_files/{self.exp_name}')

        self.Noise = None

        # Fetch experiment ID
        exp_id = schema.Experiment() & {'exp_name' : self.exp_name}
        self.exp_id = exp_id.fetch('id')[0]

        # Fetch protocol info
        self.protocol_name = 'edu.washington.riekelab.protocols.MovingLetters'
        protocol_id = schema.Protocol() & {'name' : self.protocol_name}
        self.protocol_id = protocol_id.fetch('protocol_id')[0]
        
        # Check if this protocol shows up more than once in this experiment
        experiment_summary = vr.mea_exp_summary(exp_name)
        exp_protocol_search = experiment_summary[(experiment_summary['protocol_name']==self.protocol_name)]
        protocol_options = exp_protocol_search.to_string(columns = ['protocol_name', 'minutes_since_start'], header=False, index = False)

        # If yes, ask for user input to select which instance of the protocol you're wanting to analyze
        if len(exp_protocol_search) > 1:
            protocol_instance_input: str = input(f'''This protocol shows up {len(exp_protocol_search)} times in this experiment.\n
            Which do you want to use?\n
            {protocol_options}\n                          
            Input a number from 1-{len(exp_protocol_search)}''')
            
            self.protocol_instance = int(protocol_instance_input)-1
        else:
            self.protocol_instance: int = 0

        # Fetch the appropriate protocol datafile path
        self.protocol_data_file = dr.get_data_path(exp_name = self.exp_name,
                                            target_protocol = self.protocol_name, protocol_index = self.protocol_instance)

        self.protocol_data_file = os.path.basename(self.protocol_data_file)

        # Fetch the protocol chunk id and chunk name
        protocol_chunk_id = schema.EpochBlock() & {'experiment_id' : self.exp_id, 'protocol_id' : self.protocol_id}
        self.protocol_chunk_id = protocol_chunk_id.fetch('chunk_id')[0]
        protocol_chunk_name = schema.SortingChunk() & {'experiment_id' : self.exp_id, 'id' : self.protocol_chunk_id}
        self.protocol_chunk_name = protocol_chunk_name.fetch('chunk_name')[0]

    def cluster_match(self, noise_chunk:str = None) -> dict:
 
        if noise_chunk is None:
            # match to nearest noise by default
            # Find Nearest Noise
            nearest_noise_chunk = dr.find_nearest_noise(self.exp_name, self.protocol_name, self.protocol_instance)
            self.Noise = NoiseChunk(self.exp_name, nearest_noise_chunk, self.ks_version)
        else:
            self.Noise = NoiseChunk(self.exp_name, noise_chunk, self.ks_version)

        # If no, fetch all noise chunks and associated classification files
        if len(self.Noise.sorting_files) == 0:
            all_chunks = schema.SortingChunk() & dj.AndList([f'experiment_id={self.exp_id}', 'chunk_name LIKE "chunk%"'])
            all_chunk_ids = all_chunks.fetch('id')
            all_chunk_names = all_chunks.fetch('chunk_name')

            all_sorting_files = dict()
            for jj in range(len(all_chunk_ids)):
                sorting_file = schema.CellTypeFile() & {'chunk_id' : all_chunk_ids[jj]}
                try:
                    all_sorting_files[all_chunk_names[jj]] = sorting_file.fetch('file_name')[0]
                except:
                    all_sorting_files[all_chunk_names[jj]] = None
                
            all_sorting_files = dict(sorted(all_sorting_files.items()))

            # Throw an error if no classification files exist for this experiment yet
            if all(vals is None for vals in all_sorting_files.values()):
                raise Exception("No classification files found for this experiment")
            else:
                valid_sorting_files = {key: value for key, value in all_sorting_files.items() if value is not None}

            # Ask user to select the sorting file they would like to use
            noise_chunk_to_use = input(f'''No sorting file found for {self.Noise.chunk_name}.\n
            Which chunk's sorting file would you like to use instead?\n
            {valid_sorting_files}\n
            Enter number between 1-{len(valid_sorting_files)}: ''')

            noise_chunk_to_use = int(noise_chunk_to_use)-1
            noise_chunk = list(valid_sorting_files.keys())[noise_chunk_to_use]
            self.Noise = NoiseChunk(self.exp_name, noise_chunk, self.ks_version)

        # Check if there's mor than one sorting file
        if len(self.Noise.sorting_files) > 1:
            sorting_files_available = self.Noise.sorting_files.fetch('file_name')
            sorting_file_instance: str = input(f'''This chunk has {len(self.Noise.sorting_files)} sorting files.\n
                                                Which do you want to use?\n
                                                {sorting_files_available}\n
                                                Input a number from 1-{len(self.Noise.sorting_files)}''')
            sorting_file_instance = int(sorting_file_instance)-1
            self.cluster_matching_file = sorting_files_available[sorting_file_instance]
        else:
            self.cluster_matching_file = self.Noise.sorting_files.fetch('file_name')[0]

        # Create classification file path
        if os.path.exists(self.classification_file_path):
            pass    
        else:
            os.mkdir(self.classification_file_path)
        
        protocol_classification_file = os.path.join('assets', 'classification_files', str(self.exp_name), f'movingLetters_{self.Noise.chunk_name}_autoClassification.txt')
         
        self.match_dict = dr.auto_classification(self.exp_name, self.Noise.chunk_name, self.protocol_name,
                        self.cluster_matching_file, match_index = self.protocol_instance, output_filename = protocol_classification_file)

        return self.match_dict
        

    def get_stim_params(self) -> pd.DataFrame:

        epoch_block = schema.EpochBlock & {'experiment_id' : self.exp_id, 'protocol_id' : self.protocol_id}
        epoch_block = epoch_block.fetch(format='frame').reset_index()
        epoch_block = epoch_block.loc[self.protocol_instance]
        epoch_block_id = epoch_block['id']

        epoch = schema.Epoch & {'experiment_id': self.exp_id, 'parent_id': epoch_block_id}
        epoch = epoch.fetch(format='frame').reset_index()
        epoch = epoch.loc[0]

        epoch_params = epoch['parameters']
        epoch_block_params = epoch_block['parameters']

        frameRate = 59
        msPerFrame = 1/frameRate*1e3
        framesPerMs = 1/msPerFrame

        if 'preFrames' in epoch_block_params:
            preFrames = epoch_block_params['preFrames']
        else:
            preFrames = int(np.floor(framesPerMs * epoch_block_params['preTime']))

        if 'flashFrames' in epoch_block_params:
            flashFrames = epoch_block_params['flashFrames']
        else:
            flashFrames = int(np.floor(framesPerMs*epoch_block_params['flashTime']))

        if 'gapFrames' in epoch_block_params:
            gapFrames = epoch_block_params['gapFrames']
        else:
            gapFrames = int(np.floor(framesPerMs*epoch_block_params['gapTime']))

        if 'tailFrames' in epoch_block_params:
            tailFrames = epoch_block_params['tailFrames']
        else:
            tailFrames = int(np.floor(framesPerMs * epoch_block_params['tailTime']))

        actualStimFrames = int(np.floor((flashFrames + gapFrames) * epoch_block_params['imagesPerEpoch']))

        # issues with old code
        if '0429' in self.exp_name or '0514' in self.exp_name:
            flashFrames = 5
            gapFrames = 22

        # issues with old code
        if '0527' in self.exp_name or '0530' in self.exp_name:
            flashFrames = 5
            gapFrames = 23

        images_per_epoch = epoch_block_params['imagesPerEpoch']
        num_epochs = epoch_block_params['numberOfAverages']
        mat_file = epoch_block_params['matFile']
        magnification_factor = epoch_params['magnificationFactor']

        # issues with old code
        if '0429' in self.exp_name or '0514' in self.exp_name:
            ls_param_names = ['imageOrder', 'magnificationFactor']
        else:
            ls_param_names = ['imageOrientation', 'imageMovement', 'magnificationFactor'] 

        all_epochs = schema.Epoch & {'experiment_id': self.exp_id, 'parent_id': epoch_block_id}
        all_epochs = all_epochs.fetch(format='frame').reset_index()

        if '0429' in self.exp_name or '0514' in self.exp_name:
            raw_orientation = [all_epochs.loc[epoch, 'parameters']['imageOrder'] for epoch in range(num_epochs)] 
            image_orientation = [json.loads(list_in) for list_in in raw_orientation]
            image_orientation = np.array(image_orientation, dtype=object)
        else:
            raw_orientation = [all_epochs.loc[epoch, 'parameters']['imageOrientation'] for epoch in range(num_epochs)]
            image_orientation = [json.loads(list_in) for list_in in raw_orientation]
            image_orientation = np.array(image_orientation, dtype = object)

            raw_movement = [all_epochs.loc[epoch, 'parameters']['imageMovemebnt'] for epoch in range(num_epochs)]
            image_movement = [json.loads(list_in) for list_in in raw_movement]
            image_movement = np.array(image_movement, dtype = object)


        epoch_block_properties = epoch_block['properties']

        frame_times = epoch_block_properties['frameTimesMs']
        frame_times = np.array(frame_times, dtype=object)

        # Put it all in a stimulus dataframe
        epochs = np.arange(1,num_epochs+1)
        image_files = [os.path.basename(mat_file) for i in range(num_epochs)]

        locations_OFF_filename = os.path.join('assets', 'resources', os.path.splitext(mat_file)[0] + '_OFF_locations.mat')
        locations_ON_filename = os.path.join('assets', 'resources', os.path.splitext(mat_file)[0] + '_ON_locations.mat')
        locations_OFF = scipy.loadmat(locations_OFF_filename)
        locations_ON = scipy.loadmat(locations_ON_filename)
        locations_OFF = locations_OFF['locations_OFF'].squeeze() 
        locations_ON = locations_ON['locations_ON'].squeeze()

        for idx in range(len(locations_OFF)):
            locations_OFF[idx] = np.flip(locations_OFF[idx]) * 1/self.Noise.pixels_per_stixel
            locations_ON[idx] = np.flip(locations_ON[idx]) * 1/self.Noise.pixels_per_stixel

        locations_OFF = np.array(locations_OFF, dtype=object)
        locations_ON = np.array(locations_ON, dtype=object)

        stim_params = {'epoch' : epochs, 'image_file': image_files}

        stim_df = pd.DataFrame(stim_params)
        stim_df['orientation'] = list(image_orientation)

        if '0429' in self.exp_name or '0514' in self.exp_name:
            pass
        else:
            stim_df['movement'] = list(image_movement)
            
        stim_df['frame_times'] = list(frame_times)
        stim_df.set_index('epoch',inplace=True)

        return stim_df

