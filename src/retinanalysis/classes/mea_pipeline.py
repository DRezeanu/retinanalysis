import numpy as np
from retinanalysis.classes.response import MEAResponseBlock
from retinanalysis.classes.stim import MEAStimBlock
from retinanalysis.classes.analysis_chunk import AnalysisChunk
from retinanalysis.utils.vision_utils import cluster_match, get_spike_xarr
import os
from typing import (List,
                    Dict,
                    Optional,
                    Any)
import pickle
from matplotlib.axes import Axes
import xarray as xr
import matplotlib.pyplot as plt

SAMPLE_RATE = 20000


class MEAPipeline:
    """
    MEA Pipeline object primarily meant as a container for the MEAStimBlock, MEAResponseBlock, and AnalysisChunk
    objects. The pipeline aggregates these objects and performs methods across them, such as clustery matching
    cell_ids from an AnalysisChunk object (which contains rf params and stas from a noise run) to an MEAResponseBlock
    object (which contains cell_ids and spike times for a particular protocol datafile).

    NOTE: that MEAPipeline objects are not usually created using the initializer. Typically one would use the 
    utility function create_mea_pipeline(), which will create each of the input objects and then feed them to the
    MEAPipeline initializer for you. 

    Init Parameters:
    stim_block (MEAStimBlock): A stimulus block object, see help(MEAStimBlock) for more details.

    response_block (MEAResponseBlock): A response block object, see help(MEAResponseBlock) for more details.

    analysis_chunk (AnalysisChunk): An AnalysisChunk object, see help(AnalysisChunk) for more details

    typing_file (str): Optional. Can specify which cell typing file to prioritize when filling in the 'cell_type'
    column of the MEAResponseBlock.df_spike_times DataFrame. By default, we will use the 0th cell typing file 
    from the provided AnalysisChunk, or none if none exists.

    pkl_file (str): Optional. Path to a pickle file containing an MEAPipeline object. Use this and leave all other
    inputs blank if you've exported a pipeline object using the export_to_pkl() method.

    Init Returns:
    MEAPipeline object for the stim_block, response_block and analysis_chunk given to the initializer.

    Properties:
    Use the print command on an instance of MEAPipeline to get a list of all properties contained in
    the object
    """

    def __init__(self, stim_block: Optional[MEAStimBlock] = None, response_block: Optional[MEAResponseBlock] = None,
                 analysis_chunk: Optional[AnalysisChunk] = None, typing_file: Optional[str] = None,
                 pkl_file: Optional[str] = None):

        if pkl_file is None:
            if stim_block is None or response_block is None or analysis_chunk is None:
                raise ValueError("Either stim_block, response_block, and analysis_chunk must be provided or pkl_file.")
        else:
            with open(pkl_file, 'rb') as f:
                d_out = pickle.load(f)
            self.__dict__.update(d_out)
            self.stim_block = MEAStimBlock(pkl_file=d_out['stim_block'])
            self.response_block = MEAResponseBlock(pkl_file=d_out['response_block'])
            self.analysis_chunk = AnalysisChunk(pkl_file=d_out['analysis_chunk'])
            print(f"MEAPipeline loaded from {pkl_file}")
            return
        
        self.stim_block = stim_block
        self.response_block = response_block
        self.analysis_chunk = analysis_chunk
        self.typing_file = typing_file

        self.match_dict, self.corr_dict = cluster_match(self.analysis_chunk, self.response_block)
        
        self.add_matches_to_protocol()
        self.add_types_to_protocol(typing_file_name = self.typing_file)
    

    def add_matches_to_protocol(self) -> None:
        """
        Built in MEAPipeline method for adding a 'noise_id' column to the df_spike_times
        dataframe in the included MEAResponseBlock. During initialization, the cluster_match()
        utility function is called to create a noise_id : protocol_id match dictionary, and then this
        dictionary is used in reverse to assign a noise_id to every protocol_id in the 
        MEAResponseBlock.df_spike_times dataframe.
        """
        inverse_match_dict = {val : key for key, val in self.match_dict.items()}
        for id in self.response_block.df_spike_times['cell_id']:
            if id in inverse_match_dict:
                pass
            else:
                inverse_match_dict[id] = 0

        for idx, id in enumerate(self.response_block.df_spike_times['cell_id'].values):
            self.response_block.df_spike_times.at[idx, 'noise_id'] = inverse_match_dict[id]
        
        self.response_block.df_spike_times['noise_id'] = self.response_block.df_spike_times['noise_id'].astype(int)

    def add_types_to_protocol(self, typing_file_name: Optional[str] = None) -> None:
        """
        Built in MEAPipeline method for adding a 'cell_type' column to the df_spike_times
        dataframe in the included MEAResponseBlock. If no 'typing_file' argument is given to the 
        MEAPipeline initializer, this function is called by default using the 0th typing file in the
        included AnalysisChunk object. If a typing file is given, that typing file will be used.

        This function can also be called after initialization to overwrite the MEAResponseBlock.df_spike_times
        'cell_type' column with cell types from a different typing file.

        Parameters:
        typing_file_name (str): Name of a typing file that exists in the analysis directory for the
        AnalysisChunk that was used to generate this MEAPipeline object. Default is the 0th typing
        file in the AnalysisChunk.typing_files list.

        Returns:
        None: This function does not return anything. It simply reassigns the values in the 
        MEAResponseBlock.df_spike_times 'cell_type' column using whichever typing file was given
        as the source of the information.
        """

        if typing_file_name is None:
            typing_file = 0
            print(f"Using {self.analysis_chunk.typing_files[typing_file]} for classification.\n")
        else:
            try:
                typing_file = self.analysis_chunk.typing_files.index(typing_file_name)
                print(f"Using {self.analysis_chunk.typing_files[typing_file]} for classification.\n")
            except:
                raise FileNotFoundError(f"{typing_file_name} Not Found in Analysis Chunk")
        
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

    def plot_rfs(self, protocol_ids: Optional[List[int] | int] = None, cell_types: Optional[List[str] | str] = None,
                 minimum_n: int = 1, **kwargs) -> Optional[np.ndarray[Any, np.dtype[np.object_]]]:
        """
        Stub method that mainly calls AnalysisChunk.plot_rfs(). This method allows you to give a list of 
        protocol_ids and/or cell types, and will plot the receptive fields of all of those cells, organized
        by cell type. The stub method here mainly converts protocol_ids to noise_ids, since all RF params 
        are contained within the AnalysisChunk.

        Parameters:
        protocol_ids (List[int]): A list of integer cell ids as assigned by spike sorting to your protocol
        datafile. 

        cell_types (List[str]): A list of cell type strings. All protocol_ids that are part of these cell
        types will be plotted, whether or not they're in the protocol_ids list.

        minimum_n (int): Optional, default is 1. This sets the lower limit on the number of cells of a given
        type that are required for the function to plot an axis for it. If there are only 2 Off Smooth cells
        and minimum_n is set to 3, there will be no OffS plot in the output.

        **kwargs: kwargs are fed to AnalysisChunk.plot_rfs. call help on that method for more details.

        Returns:
        ax (Axis or Numpy Array of Axes): A figure with one axis/plot per cell type will be plotted and
        the Axis or np.ndarray of Axes is returned in case the user wants to modify the axes or figure
        further after plotting. 
        """
        
        if isinstance(cell_types, str):
            cell_types = [cell_types]
        
        if isinstance(protocol_ids, int) or isinstance(protocol_ids, float):
            protocol_ids = [int(protocol_ids)]
        
        noise_ids = self.get_noise_ids(protocol_ids = protocol_ids, cell_types = cell_types)

        # Check if user provided a typing file. If not, use the typing file provided when pipeline
        # was initialized. This can still be None, in which case typing_file_0 will be used.
        if 'typing_file' in kwargs:
            ax = self.analysis_chunk.plot_rfs(noise_ids = noise_ids, cell_types = cell_types,
                                              minimum_n = minimum_n, **kwargs)
        else:
            ax = self.analysis_chunk.plot_rfs(noise_ids = noise_ids, cell_types = cell_types,
                                              minimum_n = minimum_n, typing_file = self.typing_file,
                                              **kwargs)
            

        return ax
    
    def get_cells_by_region(self, roi: Dict[str, float], units: str = 'pixels'):

        noise_ids = self.analysis_chunk.get_cells_by_region(roi = roi, units = units)
        protocol_ids = [val for key, val in self.match_dict.items() if key in noise_ids]
        arr_ids = np.array(protocol_ids)
        
        return arr_ids

    def plot_timecourses(self, protocol_ids: Optional[List[int]] = None, cell_types: Optional[List[str]] = None, 
                         minimum_n: int = 1, **kwargs) -> Optional[np.ndarray[Any, np.dtype[np.object_]]]:
        """
        Stub method that mainly calls AnalysisChunk.plot_timecourses(). This method allows you to give a list of 
        protocol_ids and/or cell types, and will plot the timecourses of all of those cells, organized
        by cell type. The stub method here mainly converts protocol_ids to noise_ids, since all STA timecourses 
        are contained within the AnalysisChunk.

        Parameters:
        protocol_ids (List[int]): A list of integer cell ids as assigned by spike sorting to your protocol
        datafile. 

        cell_types (List[str]): A list of cell type strings. All protocol_ids that are part of these cell
        types will be plotted, whether or not they're in the protocol_ids list.

        minimum_n (int): Optional, default is 1. This sets the lower limit on the number of cells of a given
        type that are required for the function to plot an axis for it. If there are only 2 Off Smooth cells
        and minimum_n is set to 3, there will be no OffS plot in the output.

        **kwargs: kwargs are fed to AnalysisChunk.plot_timecourses(). call help on that method for more details.

        Returns:
        ax (Axis or Numpy Array of Axes): A figure with one axis/plot per cell type will be plotted and
        the Axis or np.ndarray of Axes is returned in case the user wants to modify the axes or figure
        further after plotting. The solid lines are the mean timecourse for each color channel, and the shaded
        areas cover is one standard deviation.
        """
        
        if isinstance(cell_types, str):
            cell_types = [cell_types]
            
        if isinstance(protocol_ids, int) or isinstance(protocol_ids, float):
            protocol_ids = [int(protocol_ids)]

        noise_ids = self.get_noise_ids(protocol_ids, cell_types)

        # Check if user provided a typing file. If not, use the typing file provided when pipeline
        # was initialized. This can still be None, in which case typing_file_0 will be used.
        if 'typing_file' in kwargs:
            ax = self.analysis_chunk.plot_timecourses(noise_ids, cell_types = cell_types, 
                                                 minimum_n = minimum_n, **kwargs)
        else:
            ax = self.analysis_chunk.plot_timecourses(noise_ids, cell_types = cell_types, 
                                                 minimum_n = minimum_n, typing_file = self.typing_file,
                                                      **kwargs)
            
        
        return ax

    def get_noise_ids(self, protocol_ids: Optional[List[int]] = None, cell_types: Optional[List[str]] = None) -> List[int]:
        """
        Helper function for pulling noise ids for plotting and organizing them into a dictionary by type.
        IDs can be pulled by list of protocol ids, list of cell types, or both. This helper function is used in
        the built-in MEAPipeline plot_rfs and plot_timecourse methods to convert protocol_ids to noise_ids
        before the full AnalysisChunk versions of those functions are called.
        """

        # Pull analysis_block ids that match the input cell_ids and cell_types
        # If neither is given, plot all matched ids
        if isinstance(cell_types, str):
            cell_types = [cell_types]
        
        if isinstance(protocol_ids, int) or isinstance(protocol_ids, float):
            protocol_ids = [int(protocol_ids)]

        if protocol_ids is None and cell_types is None:
            protocol_ids = list(self.response_block.df_spike_times['cell_id'].values)
            noise_ids = [key for key, val in self.match_dict.items() if val in protocol_ids]

        # If only type is given, pull only ids that correspond to that type
        elif protocol_ids is None:
            protocol_ids = list(self.response_block.df_spike_times.query('cell_type in @cell_types')['cell_id'].values)
            noise_ids = [key for key, val in self.match_dict.items() if val in protocol_ids]

        # If only ids are given, pull all ids regardless of type
        elif cell_types is None:
            noise_ids = [key for key, val in self.match_dict.items() if val in protocol_ids]

        # If both are given, pull only ids that match both the cell types and the cell ids given
        else:
            filtered_protocol_ids = self.response_block.df_spike_times.query('cell_type in @cell_types and cell_id in @protocol_ids')['cell_id'].values
            noise_ids = [key for key, val in self.match_dict.items() if val in filtered_protocol_ids]

        if len(noise_ids) == 0:
            raise Exception("No cluster matched ids found for given list of cell ids and/or cell types") 

        return noise_ids

    def get_psth_arr(self, protocol_ids: Optional[List[int] | int] = None,
                     cell_types: Optional[List[str] | str] = None,
                     typing_file: Optional[str] = None, minimum_n: int = 1,
                     bins: Optional[np.ndarray | list | int] = None) -> xr.DataArray:
        """
        Function for creating an array of peri-stimulus time histograms (PSTHs) for a
        list of protocol_ids, a list of cell_types, or both. As with plot_rfs() and 
        plot_timecourses(), you can give a minimum_n value so that cell types with less
        than the minumum number of cells are not included int he final array. 

        Parameters:
        protocol_ids (List[int] | int): A single integer ID or list of cell IDs to include

        cell_types (List[str] | str): A single cell_type string or list of cell type strings
        to include

        typing_file (str): Optional. The name of a typing file to use. If none is given, then 
        the typing file used to intantiate the MEAPipeline object will be used.

        minimum_n (int): Optional, default 1. A minimum number of cells required for a cell type
        to be included in the output array.

        bins (np.ndarray | list | int): Optional. Frame times used by default. If an integer
        is given, the spike times will be binned in that many evently spaced bins. If a list
        is given, the values in the list are used as bin edges.

        Returns:
        psth_xarr (xr.DaraArray): an xarray DataArray with dimensions (cell_id, epoch, bin)
        and coordinates (cell_id, epoch, cell_type, bin, bin_edges).
        """

        # Bins are frame times by default
        if bins is None:
            bin_edges = np.array(self.stim_block.df_epochs.loc[0, 'frame_times_ms'])
        else:
            if isinstance(bins, int):
                all_epoch_starts = np.array(self.response_block.d_timing['epochStarts'])
                all_epoch_ends = np.array(self.response_block.d_timing['epochEnds'])

                # epoch starts and ends in milliseconds
                epoch_start = 0
                epoch_end = np.mean(all_epoch_ends - all_epoch_starts)/SAMPLE_RATE*1e3

                bin_edges = np.linspace(epoch_start, epoch_end, bins)
            else:
                bin_edges = bins


        if typing_file is not None:
            self.add_types_to_protocol(typing_file_name = typing_file)

        spike_times = get_spike_xarr(self.response_block, protocol_ids = protocol_ids,
                                     cell_types = cell_types, minimum_n = minimum_n)

        n_bins = len(bin_edges)-1

        def apply_hist(arr, bin_edges):
            output, _ = np.histogram(arr, bin_edges)
            return output

        psth_xarr = xr.apply_ufunc(apply_hist, spike_times,
                                   kwargs = {'bin_edges' : bin_edges}, 
                                   input_core_dims = [[]], output_core_dims = [['bin']],
                                   vectorize = True)

        psth_xarr = psth_xarr.assign_coords({'bin' : np.arange(0, n_bins)})
        psth_xarr = psth_xarr.assign_coords({'bin_edges' : ('bin', bin_edges[1:])})

        return psth_xarr

    def plot_psth(self, protocol_ids: Optional[List[int] | int] = None,
                  cell_types: Optional[List[str] | str] = None,
                  typing_file: Optional[str] = None, minimum_n: int = 1,
                  bins: Optional[np.ndarray | list | int] = None,
                  time_step: int = 500) -> dict:


        psth_arr = self.get_psth_arr(protocol_ids = protocol_ids,
                                     cell_types = cell_types,
                                     typing_file = typing_file,
                                     minimum_n = minimum_n,
                                     bins = bins)


        unique_types = np.unique(psth_arr.coords['cell_type'].to_numpy())
        bin_idx = psth_arr.coords['bin'].to_numpy()
        bin_edges = psth_arr.coords['bin_edges'].to_numpy()
        avg_bin_width = np.mean(np.diff(bin_edges))
        x_tick_step = np.ceil(time_step/avg_bin_width).astype(int)


        all_ax = dict() 
        for ct in unique_types:
            filtered_xarr = psth_arr.where(psth_arr.cell_type == ct, drop = True)
            cell_ids = filtered_xarr.coords['cell_id'].to_numpy()
            num_cells = len(cell_ids)

            cols = 4
            rows = np.ceil(num_cells/cols).astype(int)

            fig, ax = plt.subplots(nrows = rows, ncols = cols,
                                   figsize = (5*cols, 3*rows),
                                   layout = 'constrained')

            ax = ax.flatten()
            
            for idx, id in enumerate(cell_ids):
                noise_id = psth_arr.sel(cell_id = id).noise_id.item()
                psth = psth_arr.sel(cell_id = id).to_numpy()
                im1 = ax[idx].imshow(psth, aspect = 'auto')
                ax[idx].set_xticks(bin_idx[::x_tick_step], bin_edges[::x_tick_step])
                ax[idx].set_xlabel('Time (ms)')
                ax[idx].set_ylabel('Epoch')
                ax[idx].set_title(f'Cell ID: {id}, Noise ID: {noise_id}')
                plt.colorbar(im1)

            fig.suptitle(f'{ct} PSTHs')


            num_axes = rows * cols
            empty_axes = num_axes - len(cell_ids)
            for i in range(empty_axes):
                fig.delaxes(ax[num_axes-i-1])
            
            all_ax[ct] = ax
        
        return all_ax


    def __repr__(self):
        str_self = f"{self.__class__.__name__} with properties:\n"
        str_self += f"  stim_block and response_block from: {os.path.splitext(self.stim_block.protocol_name)[1][1:]}\n"
        str_self += f"  analysis_chunk: {self.analysis_chunk.chunk_name}\n"
        str_self += f"  match_dict: with {self.analysis_chunk.chunk_name}_id : {os.path.splitext(self.stim_block.protocol_name)[1][1:]}_id\n"
        str_self += f"  corr_dict: with {self.analysis_chunk.chunk_name}_id : calculated ei correlations\n"
        return str_self

    def export_to_pkl(self, file_path: str):
        """
        Export the MEAPipeline to a pickle file. The output of this method can be given to the 
        MEAPipeline initializer directly to reload a saved pipeline object with all its properties intact.
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

def create_mea_pipeline(exp_name: str, datafile_name: str, analysis_chunk_name: Optional[str] = None,
                    typing_file: Optional[str] = None, ss_version: str = 'kilosort2.5',
                    ls_params: Optional[list] = None, b_load_fd: bool = True):
    """
    Helper function for initializing an MEAPipeline from metadata.

    Parameters:
    exp_name (str): experiment name as found in the datajoint database (e.g. '20251022C')

    datafile_name (str): name of protocol datafile of interest (e.g. 'data006')

    analysis_chunk_name (str): Name of noise chunk to use for RF params and cell typing 
    information. This input is optional, the nearest noise chunk will be determined and used
    by default.

    ss_version (str): Kilosort version used for spike sorting. Default is 'kilosort2.5'. This
    is mostly used to locate the appropriate files, since they're usually kept in a folder
    that is named for the kilosort version. (e.g. /analysis_dir/chunk_name/ss_version/*files of interest*)
    
    ls_params (List): List of epoch parameters to pull into their own column in the MEAStimBlock.df_epochs
    DataFrame. By default parameters that change with each epoch are already pulled, but additional params
    can be specified in this list.

    Returns:
    pipeline (MEAPipeline): MEAPipeline object that contains the MEAStimBlock and MEAResponse block for the
    given datafile, and the AnalysisChunk for the given noise chunk or, if none is given, the nearest noise chunk.
    """
    # TODO StimGroup and ResponseGroup functionality
    s = MEAStimBlock(exp_name, datafile_name, ls_params)
    if rb_ss_version is None:
        rb_ss_version = ss_version

    r = MEAResponseBlock(exp_name, datafile_name, rb_ss_version, b_load_fd=b_load_fd)
    if analysis_chunk_name is None:
        analysis_chunk_name = s.nearest_noise_chunk
        print(f'Using {analysis_chunk_name} for AnalysisChunk\n')
    ac = AnalysisChunk(exp_name, analysis_chunk_name, ss_version)
    pipeline = MEAPipeline(stim_block = s, response_block = r, analysis_chunk = ac, typing_file = typing_file)
    return pipeline
