import h5py
import json
import numpy as np
from scipy.signal import butter, filtfilt, bessel
from typing import Tuple, Optional, List
import bin2py
import os, argparse
import re
from datetime import datetime, timedelta
from typing import Union

def strip_uuid(uuid: str) -> str:
    """ Strip the UUID from the UUID string. """
    return '-'.join(uuid.split('-')[:-5])

TTL_CHANNEL = 0
def get_litke_triggers(bin_path, RW_BLOCKSIZE=2000000, TTL_THRESHOLD=-1000):
    """ Get the Litke triggers from the binary file. 
    Parameters:
        bin_path: The path to the binary file (str).
        RW_BLOCKSIZE: The block size (int).
        TTL_THRESHOLD: The TTL threshold (int).
    Returns:
        epoch_starts: The epoch start times (np.ndarray).
        epoch_ends: The epoch end times (np.ndarray).
        array_id: The array ID (int).
        n_samples: The number of samples (int).
    """
    epoch_starts = []
    epoch_ends = []
    try:
        with bin2py.PyBinFileReader(bin_path, chunk_samples=RW_BLOCKSIZE, is_row_major=True) as pbfr:
            array_id = pbfr.header.array_id
            n_samples = pbfr.length
            for start_idx in range(0, n_samples, RW_BLOCKSIZE):
                n_samples_to_get = min(RW_BLOCKSIZE, n_samples - start_idx)
                samples = pbfr.get_data_for_electrode(0, start_idx, n_samples_to_get)
                # Find the threshold crossings at the beginning and end of each epoch.
                below_threshold = (samples < TTL_THRESHOLD)
                above_threshold = np.logical_not(below_threshold)
                # Epoch starts.
                above_to_below_threshold = np.logical_and.reduce([
                    above_threshold[:-1],
                    below_threshold[1:]
                ])
                # Make a Schmitt trigger at each crossing to make sure that we don't get multiple crossings within a few samples..
                trigger_indices = np.argwhere(above_to_below_threshold) + start_idx
                epoch_starts.append(trigger_indices[:, 0])
                below_to_above_threshold = np.logical_and.reduce([
                    below_threshold[:-1],
                    above_threshold[1:]
                ])
                trigger_indices = np.argwhere(below_to_above_threshold) + start_idx
                epoch_ends.append(trigger_indices[:, 0])
    except Exception as e:
        print(f'ERROR: Unable to parse bin file, error: {e}')
        array_id = None
        n_samples = None
        return epoch_starts, epoch_ends, array_id, n_samples 
    epoch_starts = np.concatenate(epoch_starts, axis=0)
    epoch_ends = np.concatenate(epoch_ends, axis=0)
    return epoch_starts, epoch_ends, array_id, n_samples

# Stopped recording frames in Labview 20220518..
def get_litke_triggers_old(binpath, transitionThreshold = 2000):
    """ Get the Litke triggers from the binary file. 
    Parameters:
        binpath: The path to the binary file (str).
        transitionThreshold: The transition threshold (int).
    Returns:
        epoch_starts: The epoch start times (np.ndarray).
        epoch_ends: The epoch end times (np.ndarray).
        array_id: The array ID (int).
        n_samples: The number of samples (int).
    """
    # print("Getting Litke triggers from binary file (old format)...")
    # In this case, the flips are frame times, not epoch boundaries.
    try:
        epoch_starts,epoch_ends,array_id,n_samples = get_litke_triggers(binpath)
        if len(epoch_ends) > 0:
            last_end = epoch_ends[-1]
            # Combined the start and end times.
            frame_times = np.sort(np.concatenate([epoch_starts,epoch_ends]))
            # Take the derivative of the frame times.
            d_frames = np.diff(frame_times)
            start_idx = np.where(d_frames >= transitionThreshold)[0]
            start_idx = np.insert(start_idx,0,0)
            end_idx = start_idx[1:]
            epoch_starts = frame_times[start_idx]
            epoch_ends = frame_times[end_idx]
            epoch_ends = np.append(epoch_ends,last_end)
        return epoch_starts,epoch_ends,array_id,n_samples
    except Exception as error:
        print('Error: ' + str(error))
        return np.array([]),np.array([]),0,0

def butter_lowpass_filter(data, cutoff, fs, order=6):
    """ Apply a low-pass Butterworth filter to the data.
    Parameters:
        data: The data to filter (np.ndarray).
        cutoff: The cutoff frequency (float).
        fs: The sample rate (float).
        order: The filter order (int).
    Returns:
        y: The filtered data (np.ndarray).
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output = 'ba') # type: ignore
    y = filtfilt(b, a, data)
    return y

def bessel_lowpass_filter(data, cutoff, fs, order=6):
    """ Apply a low-pass Bessel filter to the data. 
    Parameters:
        data: The data to filter (np.ndarray).
        cutoff: The cutoff frequency (float).
        fs: The sample rate (float).
        order: The filter order (int).
    Returns:
        y: The filtered data (np.ndarray).
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = bessel(order, normal_cutoff, 'low', analog=False, norm='phase')
    y = filtfilt(b, a, data)
    return y

def calculate_up_down_offsets(
        frame_ups: np.ndarray, 
        frame_downs: np.ndarray, 
        expected_frame_rate: float=59.94) -> float:
    """ Calculate the offset between the up and down frame times.
    Parameters:
        frame_ups: The up frame times (np.ndarray).
        frame_downs: The down frame times (np.ndarray).
        expected_frame_rate: The expected frame rate (float).
    Returns:
        delta_down: The offset between the up and down frame times (float).
    """
    try:
        d_down = np.zeros(frame_downs.shape[0])
        for ii in range(frame_ups.shape[0]):
            d_idx = np.where(frame_downs > frame_ups[ii])[0]
            if len(d_idx) > 0:
                d_down[ii] = np.min(np.abs(frame_downs[d_idx] - frame_ups[ii]))
        d_down = d_down[d_down < 30.0]
        delta_down = np.round(np.mean(d_down) - 1000.0/expected_frame_rate).astype(int)
        delta_down = np.max([delta_down, 0])
    except Exception as error:
        print('Error: ' + str(error))
        delta_down = 0
    return delta_down

def find_threshold_cross(x, threshold, direction):
    x_original = x[:-1]
    x_shift = x[1:]
    if direction > 0:
        index = np.argwhere((x_original < threshold) & (x_shift >= threshold)).ravel()
    else:
        index = np.argwhere((x_original >= threshold) & (x_shift < threshold)).ravel()
    return index

def find_peaks(x, direction):
    # Take the second derivative.
    d2x = np.diff((np.diff(x) > 0.0).astype(float))
    if direction > 0: # local max
        index = np.argwhere(d2x < 0.0).ravel()
    else: # local min
        index = np.argwhere(d2x > 0.0).ravel()
    index += 1
    return index, x[index]

def upsample_frame_times(ttl_times: np.ndarray, frames_per_ttl: int) -> np.ndarray:
    """
    Linearly interpolates frame times based on the ttls
    Parameters:
        ttl_times: ttl times
        frames_per_ttl: frames per ttl
    Returns:
        Array of interpolated frame times.
    """
    # Calculate the average frame time interval.
    avg_frame_interval = np.round(np.mean(np.diff(ttl_times)))
    ttl_times = np.append(ttl_times, ttl_times[-1] + avg_frame_interval)
    frame_times = []
    for i in range(ttl_times.shape[0]-1):
        frame_times.append(np.linspace(ttl_times[i],
                               ttl_times[i+1],
                               frames_per_ttl,
                               endpoint=False))
    return np.asarray(frame_times).ravel()

def find_optimal_threshold(f_monitor: np.ndarray, sample_rate: float=10000.0) -> float:
    thresholds = np.arange(0.05,0.5025,0.025)
    frame_violations = np.zeros(thresholds.shape[0])
    for t_idx, threshold in enumerate(thresholds):
        frame_ups = find_threshold_cross(f_monitor,threshold=threshold,direction=1)
        frame_downs = find_threshold_cross(f_monitor,threshold=threshold,direction=-1)
        d_ups = np.diff(frame_ups)
        d_downs = np.diff(frame_downs)
        d_frames = np.append(d_ups, d_downs)
        frame_violations[t_idx] = len(np.where(d_frames/sample_rate*1000.0 < 32.0)[0])
    return np.max(thresholds[np.where(frame_violations == np.min(frame_violations))]).astype(float)

def check_frame_times(frame_times: np.ndarray, expected_frame_rate: float=60.0, bin_rate: float=1000.0):
    # Compute the minimum and maximum intervals from the expected frame rate (msec).
    minimum_interval = np.floor(bin_rate/expected_frame_rate).astype(float)
    maximum_interval = np.ceil(bin_rate/expected_frame_rate).astype(float)
    drop_interval = np.ceil(bin_rate/expected_frame_rate*1.5).astype(float)
    # Take the derivative of the frame times.
    d_frames = np.diff(frame_times)
    long_idx = np.where((d_frames > maximum_interval) & (d_frames < drop_interval))[0]
    if np.any(long_idx):
        frame_times[long_idx+1] -= (d_frames[long_idx] - maximum_interval)
    d_frames = np.diff(frame_times)
    short_idx = np.where(d_frames < minimum_interval)[0]
    frame_times[short_idx+1] += (minimum_interval - d_frames[short_idx])
    return frame_times

def get_frame_times(frame_monitor, bin_rate: float=1000.0, sample_rate: float=10000.0):
    default_frame_rate = 59.994
    # Low-pass filter fMonitor = lowPassFilter(fMonitor,250,1/sampleRate)
    frame_monitor = butter_lowpass_filter(frame_monitor, 250, sample_rate, 6)
    # Normalize
    frame_monitor -= np.min(frame_monitor)
    frame_monitor /= np.max(frame_monitor)
    ups, _ = find_peaks(frame_monitor,1)
    downs, _ = find_peaks(frame_monitor,-1)
    d_ups = np.diff(ups/sample_rate*1000.0)
    d_downs = np.diff(downs/sample_rate*1000.0)
    # Find the flips that are too short.
    short_ups = np.argwhere(d_ups < 30.0).ravel() + 1
    short_downs = np.argwhere(d_downs < 30.0).ravel() + 1
    ups = np.delete(ups, short_ups)
    downs = np.delete(downs, short_downs)
    frame_times = ups
    frame_times = np.append(frame_times, downs)
    frame_times = np.sort(frame_times).astype(float)
    frame_times -= np.min(frame_times)
    frame_times = np.ceil(frame_times*bin_rate/sample_rate)
    return frame_times

def get_frame_times_lightcrafter(
        f_monitor: np.ndarray, 
        bin_rate: float=1000.0, 
        sample_rate: float=10000.0,
        low_pass_filter: bool=True) -> np.ndarray:
    """ Extract the frame times from the frame monitor signal (Lightcrafter 4500).
    Parameters:
        frame_monitor: The frame monitor signal (np.ndarray).
        bin_rate: The bin rate in Hz (float).
        sample_rate: The sample rate in Hz (float).
        down_offset: The offset for the down transitions (float).
    Returns:
        frame_times: The frame times (np.ndarray)
    """
    try:
        minimum_samples = np.round(15.0*sample_rate/1000.0).astype(int)
        f_monitor -= np.min(f_monitor)
        f_monitor /= np.max(f_monitor)
        if low_pass_filter:
            f_cutoff = sample_rate/10.0*3.0
            f_monitor = bessel_lowpass_filter(f_monitor,f_cutoff,sample_rate)
            f_monitor /= np.max(f_monitor)
        threshold = find_optimal_threshold(f_monitor=f_monitor, sample_rate=sample_rate)
        frame_ups = find_threshold_cross(f_monitor,threshold=threshold,direction=1)
        frame_downs = find_threshold_cross(f_monitor,threshold=threshold,direction=-1)
        delta_down = calculate_up_down_offsets(frame_ups, frame_downs, expected_frame_rate=59.94)
        frame_downs -= delta_down
        frame_times = np.append(frame_ups, frame_downs)
        frame_times = np.sort(frame_times).astype(float)
        frame_times = np.unique(frame_times)
        good_idx = list()
        good_idx.append(0)
        last_flip = 0.0
        for ii in range(1, frame_times.shape[0]):
            if frame_times[ii] - last_flip >= minimum_samples:
                good_idx.append(ii)
                last_flip = frame_times[ii]
        frame_times = frame_times[np.array(good_idx)]
        frame_times = np.append(frame_times,[0.0])
        frame_times = np.sort(frame_times).astype(float)
        frame_times = np.ceil(frame_times*bin_rate/sample_rate)
        frame_times = check_frame_times(frame_times, expected_frame_rate=60.0)
    except Exception as error:
        print('Error: ' + str(error))
        frame_step = bin_rate / 59.94
        frame_times = np.round(np.arange(0.0,len(f_monitor)/sample_rate*bin_rate,frame_step)).astype(float)
    return frame_times

def get_frame_times_pattern_mode(frame_monitor: np.ndarray, bin_rate: float=1000.0, sample_rate: float=10000.0, updates_per_frame: int=1):
    """ Extract the frame times from the frame monitor signal in pattern mode (Lightcrafter 4500). 
    Parameters:
        frame_monitor: The frame monitor signal (np.ndarray).
        bin_rate: The bin rate in Hz (float).
        sample_rate: The sample rate in Hz (float).
        updates_per_frame: The number of updates per frame (int)
    Returns:    
        frame_times: The frame times (np.ndarray)
    """
    f_monitor = frame_monitor.copy()
    f_monitor -= np.min(f_monitor)
    f_monitor /= np.max(f_monitor)
    frame_ups = find_threshold_cross(f_monitor,0.15,1)
    frame_downs = find_threshold_cross(f_monitor,0.15,-1)
    frame_times = np.append(frame_ups, frame_downs)
    frame_times = np.append(frame_times,[0.0])
    frame_times = np.sort(frame_times).astype(float)
    frame_times = frame_times*bin_rate/sample_rate
    frame_times = check_frame_times(frame_times, expected_frame_rate=60.0)
    # Upsample the frames by the pattern rate multiple.
    if updates_per_frame > 1:
        frame_times = upsample_frame_times(frame_times, updates_per_frame)
    return np.ceil(frame_times).ravel()

def get_frame_times_from_syncs(sync: np.ndarray, bin_rate: float=1000.0, sample_rate: float=10000.0):
    sync -= np.min(sync)
    sync /= np.max(sync)
    frame_ups = find_threshold_cross(sync,0.5,1)
    frame_ups = np.insert(frame_ups,0,0)
    frame_times = frame_ups[::4]
    frame_times = np.sort(frame_times).astype(float)
    frame_times = np.ceil(frame_times*bin_rate/sample_rate)
    return frame_times

def get_frame_times_from_pwm(pwm: np.ndarray, bin_rate: float=1000.0, sample_rate: float=10000.0):
    d_pwm = np.diff(pwm)
    d_pwm = np.insert(d_pwm,0,0)
    pk, _ = find_peaks(d_pwm, 1)
    th_idx = find_threshold_cross(d_pwm,0.01,1)
    d_th_idx = np.diff(th_idx).astype(float)
    good_idx = np.argwhere(d_th_idx > sample_rate/1000.0*3.9).ravel() + 1
    good_idx = np.insert(good_idx,0,0)
    if (len(th_idx) == 0) or (len(good_idx) == 0):
        return np.array([])
    frame_times = th_idx[good_idx]
    frame_times = np.insert(frame_times,0,0)
    frame_times = frame_times[::4]
    frame_times = np.sort(frame_times).astype(float)
    frame_times = np.ceil(frame_times*bin_rate/sample_rate)
    return frame_times

def dotnet_ticks_to_datetime(ticks):
    """ Convert .NET ticks to a datetime object. 
    Parameters:
        ticks: The .NET ticks (int).
    Returns:
        date_string: The date string (str).
        date_seconds: The date in seconds (float).
    """
    t = datetime(1, 1, 1) + timedelta(microseconds = int(ticks)//10)
    date_string = t.strftime("%m/%d/%Y %H:%M:%S:%f")
    # Get epoch start times in seconds.
    date_seconds = t.timestamp()
    return date_string, date_seconds

def descend_obj(obj,sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if isinstance(obj, h5py.Group) or isinstance(obj, h5py.File):
        for key in obj.keys():
            print(sep,'-',key,':',obj[key])
            descend_obj(obj[key],sep=sep+'\t')
    elif isinstance(obj, h5py.Dataset):
        for key in obj.attrs.keys():
            print(sep+'\t','-',key,':',obj.attrs[key])

def h5dump(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path,'r') as f:
         descend_obj(f[group])

def hdf5_to_json(input_hdf5_file, output_json_file):
    import sys
    sys.setrecursionlimit(2000)
    with h5py.File(input_hdf5_file, 'r') as hdf5_file:
        data_dict = recursive_hdf5_to_dict(hdf5_file)
        
    with open(output_json_file, 'w') as json_file:
        json.dump(data_dict, json_file, cls=NpEncoder)

def recursive_hdf5_to_dict(group):
    result = dict()
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = recursive_hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]  # Convert dataset to NumPy array
    return result

def parse_value(value):
    """ Parse the value from the HDF5 file. 
    Parameters:
        value: The value to parse.
    Returns:
        value: The parsed value.
    """
    if isinstance(value, np.bytes_):
        return value.decode('UTF-8')
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, h5py.Empty):
        return None
    else:
        return value

def order_list_by_start_time(lst: list, start_times: list) -> list:
    """ Order a list of objects by their start times. 
    Parameters:
        lst: The list of objects.
        start_times: The start times of the objects.
    Returns:
        lst: The ordered list of objects.
    """
    try:
        list_order = list(np.argsort(np.array(start_times)))
        return [lst[i] for i in list_order]
    except Exception as error:
        print('Error: ' + str(error))
        return lst

def get_electrode_pitch_by_array_id(array_id: int) -> str:
    """ Get the electrode pitch by the array ID. 
    Parameters:
        array_id: The array ID (str).
    Returns:
        pitch: The electrode pitch (float).
    """
    if array_id < 1501:
        pitch = '60um'
    elif array_id < 3501:
        pitch = '30um'
    else:
        pitch = '120um'
    return pitch

def parse_attributes(obj: h5py.Group) -> dict:
    """ Parse the attributes.
    Parameters:
        obj (h5py.Group): The attributes group.
    Returns:
        attributes (dict): The parsed attributes dictionary.
    """
    attributes = dict()
    for key, value in obj.attrs.items():
        if isinstance(value, np.bytes_):
            value = value.decode('UTF-8')
        elif isinstance(obj, h5py.Empty):
            value = None
        attributes[key] = value
    return attributes

def clean_epoch_group_for_json(e_group: dict) -> dict:
    """ Clean the epoch group for JSON serialization.
    Parameters:
        e_group (dict): The epoch group dictionary.
    Returns:
        e_group (dict): The cleaned epoch group dictionary."""
    if 'epoch_blocks' in e_group.keys():
        if len( e_group['epoch_blocks'] ) > 0:
            for ii in range( len( e_group['epoch_blocks'] ) ):
                if 'frameTimesMs' in e_group['epoch_blocks'][ii].keys():
                    e_group['epoch_blocks'][ii]['properties']['frameTimesMs'] = e_group['epoch_blocks'][ii]['frameTimesMs']
                    del e_group['epoch_blocks'][ii]['frameTimesMs']
                if 'array_id' in e_group['epoch_blocks'][ii].keys():
                    e_group['epoch_blocks'][ii]['properties']['array_id'] = e_group['epoch_blocks'][ii]['array_id']
                    del e_group['epoch_blocks'][ii]['array_id']
                if 'n_samples' in e_group['epoch_blocks'][ii].keys():
                    e_group['epoch_blocks'][ii]['properties']['n_samples'] = e_group['epoch_blocks'][ii]['n_samples']
                    del e_group['epoch_blocks'][ii]['n_samples']
                if 'epochStarts' in e_group['epoch_blocks'][ii].keys():
                    e_group['epoch_blocks'][ii]['properties']['epochStarts'] = e_group['epoch_blocks'][ii]['epochStarts']
                    del e_group['epoch_blocks'][ii]['epochStarts']
                if 'epochEnds' in e_group['epoch_blocks'][ii].keys():
                    e_group['epoch_blocks'][ii]['properties']['epochEnds'] = e_group['epoch_blocks'][ii]['epochEnds']
                    del e_group['epoch_blocks'][ii]['epochEnds']
        return e_group
    else:
        return e_group

class SourceObj(object):
    def __init__(self, d: Optional[dict]=None) -> None:
        self.label = None
        self.uuid = None
        self.notes = list()
        self.properties = dict()
        self.attributes = dict()
        self.start_time = None
        if d is not None:
            if 'label' in d['attributes'].keys():
                self.label = parse_value(d['attributes']['label'])
            if 'uuid' in d['attributes'].keys():
                self.uuid = parse_value(d['attributes']['uuid'])
            if 'creationTimeDotNetDateTimeOffsetTicks' in d['attributes'].keys():
                creationTimeDotNetDateTimeOffsetTicks = d['attributes']['creationTimeDotNetDateTimeOffsetTicks']
                self.start_time = dotnet_ticks_to_datetime(creationTimeDotNetDateTimeOffsetTicks)[0]
            elif 'startTimeDotNetDateTimeOffsetTicks' in d['attributes'].keys():
                startTimeDotNetDateTimeOffsetTicks = d['attributes']['startTimeDotNetDateTimeOffsetTicks']
                self.start_time = dotnet_ticks_to_datetime(startTimeDotNetDateTimeOffsetTicks)[0]
            # Set the properties.
            if 'properties' in d.keys():
                self.parse_properties(d = d['properties'])
            # Set the attributes.
            if 'attributes' in d.keys():
                self.parse_attributes(d = d['attributes'])
            # Set the notes.
            if 'notes' in d.keys():
                self.parse_notes(d = d['notes'])
    def parse_properties(self, d: dict):
        self.properties = dict()
        for key, value in d.items():
            value = parse_value(value)
            if value is not None:
                self.properties[key] = value
            elif key not in self.properties:
                self.properties[key] = None
    def parse_attributes(self, d: dict):
        self.attributes = dict()
        for key, value in d.items():
            value = parse_value(value)
            if value is not None:
                self.attributes[key] = value
            elif key not in self.attributes:
                self.attributes[key] = None
    def parse_notes(self, d: list):
        self.notes = list()
        for note in d:
            note = parse_value(note)
            if note:
                self.notes.append(note)
        self.notes = list()
        for note in d:
            note = parse_value(note)
            if note:
                self.notes.append(note)

class ExperimentObj(SourceObj):
    def __init__(self, d: Optional[dict]=None, rig_type: Optional[str]=None) -> None:
        super().__init__(d=d)
        self.rig_type = rig_type
        if d is not None:
            if 'experimenter' in d['properties'].keys():
                self.experimenter = parse_value(d['properties']['experimenter'])
            if 'institution' in d['properties'].keys():
                self.institution = parse_value(d['properties']['institution'])
            if 'lab' in d['properties'].keys():
                self.lab = parse_value(d['properties']['lab'])
            if 'project' in d['properties'].keys():
                self.project = parse_value(d['properties']['project'])
            if 'rig' in d['properties'].keys():
                self.rig = parse_value(d['properties']['rig'])

class AnimalObj(SourceObj):
    def __init__(self, d: Optional[dict]=None) -> None:
        super().__init__(d=d)
        self.id = None
        self.description = None
        self.sex = None
        self.age = None
        self.weight = None
        self.darkAdaptation = None
        self.species = None
        if d is not None:
            if 'id' in d['properties'].keys():
                self.id = parse_value(d['properties']['id'])
            if 'description' in d['properties'].keys():
                self.description = parse_value(d['properties']['description'])
            if 'sex' in d['properties'].keys():
                self.sex = parse_value(d['properties']['sex'])
            if 'age' in d['properties'].keys():
                self.age = parse_value(d['properties']['age'])
            if 'weight' in d['properties'].keys():
                self.weight = parse_value(d['properties']['weight'])
            if 'darkAdaptation' in d['properties'].keys():
                self.darkAdaptation = parse_value(d['properties']['darkAdaptation'])
            if 'species' in d['properties'].keys():
                self.species = parse_value(d['properties']['species'])

class CellObj(SourceObj):
    def __init__(self, d: Optional[dict]=None) -> None:
        super().__init__(d=d)
        self.type = None
        if d is not None:
            if 'type' in d['properties'].keys():
                self.type = parse_value(d['properties']['type'])

class PreparationObj(SourceObj):
    def __init__(self, d: Optional[dict]=None) -> None:
        super().__init__(d=d)
        self.bathSolution = None
        self.preparationType = None
        self.region = None
        self.arrayPitch = None
        self.cells = list()
        if d is not None:
            if 'bathSolution' in d['properties'].keys():
                self.bathSolution = parse_value(d['properties']['bathSolution'])
            if 'preparation' in d['properties'].keys():
                self.preparationType = parse_value(d['properties']['preparation'])
            if 'region' in d['properties'].keys():
                self.region = parse_value(d['properties']['region'])
    def add_cell(self, cell: CellObj):
        if self.cells is None:
            self.cells = list()
        self.cells.append(cell.__dict__)

class ProtocolObj(object):
    def __init__(self) -> None:
        self.label = None
        self.group = None

class EpochGroupObj(SourceObj):
    def __init__(self, d: Optional[dict]=None) -> None:
        super().__init__(d=d)
        self.epoch_blocks = list()
        self.end_time = None
        self.parse( d = d )
    def parse(self, d: Optional[dict]):
        if d is not None:
            if 'endTimeDotNetDateTimeOffsetTicks' in d['attributes'].keys():
                self.set_end_time_from_ticks( d['attributes']['endTimeDotNetDateTimeOffsetTicks'] )
            if 'block' in d.keys():
                self.epoch_blocks = self.check_epoch_blocks( d['block'] )
    def set_start_time(self, start_time: datetime):
        self.start_time = start_time
    def set_end_time(self, end_time: datetime):
        self.end_time = end_time
    def set_start_time_from_ticks(self, ticks: int):
        self.start_time = dotnet_ticks_to_datetime(ticks)[0]
    def set_end_time_from_ticks(self, ticks: int):  
        self.end_time = dotnet_ticks_to_datetime(ticks)[0]
    def check_epoch_blocks(self, blocks: list):
        for ii in range(len(blocks)):
            if 'epoch' in blocks[ii].keys():
                blocks[ii]['epochs'] = blocks[ii].pop('epoch')
            if 'frameTimesMs' in blocks[ii].keys():
                blocks[ii]['properties']['frameTimesMs'] = blocks[ii]['frameTimesMs']
            if 'array_id' in blocks[ii].keys():
                blocks[ii]['properties']['array_id'] = blocks[ii]['array_id']
            if 'n_samples' in blocks[ii].keys():
                blocks[ii]['properties']['n_samples'] = blocks[ii]['n_samples']
            if 'epochStarts' in blocks[ii].keys():
                blocks[ii]['properties']['epochStarts'] = blocks[ii]['epochStarts']
            if 'epochEnds' in blocks[ii].keys():
                blocks[ii]['properties']['epochEnds'] = blocks[ii]['epochEnds']
        return blocks

class DataObj(object):
    def __init__(self, d: Optional[Union[dict, h5py.Group]]=None) -> None:
        self.uuid = None
        self.protocolID = None
        self.properties = dict()
        self.attributes = dict()
        self.start_time = None
        self.end_time = None
        if isinstance(d, h5py.Group):
            d = self.h5_to_dict(obj=d)
        self.parse_dict(d=d)
    def h5_to_dict(self, obj: h5py.Group) -> dict:
        d = dict()
        attributes = parse_attributes(obj)
        d['attributes'] = attributes
        d['uuid'] = attributes['uuid']
        if 'properties' in obj.keys():
            properties = parse_attributes(obj['properties']) #type: ignore
            self.properties = properties
            d['properties'] = properties
        return d
    def parse_dict(self, d: Optional[dict]):
        if d is not None:
            if 'uuid' in d['attributes'].keys():
                self.uuid = parse_value(d['attributes']['uuid'])
            if 'creationTimeDotNetDateTimeOffsetTicks' in d['attributes'].keys():
                creationTimeDotNetDateTimeOffsetTicks = d['attributes']['creationTimeDotNetDateTimeOffsetTicks']
                self.start_time = dotnet_ticks_to_datetime(creationTimeDotNetDateTimeOffsetTicks)[0]
            elif 'startTimeDotNetDateTimeOffsetTicks' in d['attributes'].keys():
                startTimeDotNetDateTimeOffsetTicks = d['attributes']['startTimeDotNetDateTimeOffsetTicks']
                self.start_time = dotnet_ticks_to_datetime(startTimeDotNetDateTimeOffsetTicks)[0]
            if 'endTimeDotNetDateTimeOffsetTicks' in d['attributes'].keys():
                endTimeDotNetDateTimeOffsetTicks = d['attributes']['endTimeDotNetDateTimeOffsetTicks']
                self.end_time = dotnet_ticks_to_datetime(endTimeDotNetDateTimeOffsetTicks)[0]
            # Set the attributes.
            self.parse_attributes(d = d['attributes'])
            if 'protocolID' in self.attributes.keys():
                self.protocolID = self.attributes['protocolID']
    def parse_properties(self, d: Union[dict, h5py.Group]):
        if isinstance(d, h5py.Group):
            self.properties = parse_attributes(d)
        else:
            # self.properties = dict()
            for key, value in d.items():
                value = parse_value(value)
                if value is not None:
                    self.properties[key] = value
                elif key not in self.properties:
                    self.properties[key] = None
    def parse_attributes(self, d: dict):
        self.attributes = dict()
        for key, value in d.items():
            value = parse_value(value)
            if value is not None:
                self.attributes[key] = value
            elif key not in self.attributes:
                self.attributes[key] = None
    def set_start_time(self, start_time: datetime):
        self.start_time = start_time
    def set_end_time(self, end_time: datetime):
        self.end_time = end_time
    def set_start_time_from_ticks(self, ticks: int):
        self.start_time = dotnet_ticks_to_datetime(ticks)[0]
    def set_end_time_from_ticks(self, ticks: int):  
        self.end_time = dotnet_ticks_to_datetime(ticks)[0]

class EpochBlockObj(DataObj):
    def __init__(self, d: Union[dict,h5py.Group]) -> None:
        super().__init__(d=d)
        self.dataFile = None
        self.parameters = dict()
        self.arrayPitch: Optional[str] = None
        self.epochs = list()
        if 'dataFileName' in self.properties.keys():
            self.dataFile = self.properties['dataFileName']
        if 'protocolParameters' in d.keys():
            if isinstance(d, h5py.Group):
                self.parameters = parse_attributes(d['protocolParameters']) # type: ignore
            else:
                self.parameters = d['protocolParameters']

class ResponseObj(object):
    def __init__(self, label: Optional[str]=None, d: Optional[dict]=None) -> None:
        self.label = None
        self.sampleRate = None
        self.sampleRateUnits = None
        self.h5path = None
        self.inputTimeDotNetDateTimeOffsetOffsetHours = None
        self.inputTimeDotNetDateTimeOffsetTicks = None
        self.uuid = None
        if (label is not None) and (d is not None):
            self.set_from_dict(label=label, d=d)
    def set_from_dict(self, label: str, d: dict):
        self.label = label
        self.sampleRate = d['sampleRate']
        self.sampleRateUnits = d['sampleRateUnits']
        self.h5path = d['h5path']
        self.inputTimeDotNetDateTimeOffsetOffsetHours = d['inputTimeDotNetDateTimeOffsetOffsetHours']
        self.inputTimeDotNetDateTimeOffsetTicks = d['inputTimeDotNetDateTimeOffsetTicks']
        self.uuid = d['uuid']

class StageObj(object):
    def __init__(self,
                 key_name: str,
                 properties: dict,
                 frame_times_from_syncs: bool=True) -> None:
        self.frame_times_from_syncs = frame_times_from_syncs
        self.device_type = None
        self.mode = None
        self.frame_rate = None
        self.updates_per_frame = int(1)
        self.frame_data = None
        self.sync_data = None
        self.sample_rate = None
        self.parse_properties(key_name, properties)
    def parse_properties(self, key_name: str, properties: dict):
        self.updates_per_frame = 1
        if 'LightCrafter' in key_name:
            self.device_type = 'LightCrafter'
            self.mode = 'Pattern'
            self.frame_rate = 59.94
            try:
                if ('lightCrafterPatternRate' in properties.keys()) and ('monitorRefreshRate' in properties.keys()):
                    self.updates_per_frame = 1 #np.round(properties['lightCrafterPatternRate']/properties['monitorRefreshRate']).astype(int)
                else:
                    self.updates_per_frame = 1
            except:
                self.updates_per_frame = 1
        elif 'LcrVideo' in key_name:
            self.device_type = 'LightCrafter'
            self.mode = 'Video'
            self.frame_rate = 59.94
        elif 'Microdisplay' in key_name:
            self.device_type = 'Microdisplay'
            self.mode = 'Video'
            self.frame_rate = 60.31807657
        else:
            self.device_type = 'LightCrafter'
            self.mode = 'Video'
            self.frame_rate = 59.94    
    def set_frame_data(self, d: np.ndarray):
        self.frame_data = d
    def set_sync_data(self, d: np.ndarray):
        self.sync_data = d
    def set_sample_rate(self, r: float):
        self.sample_rate = r
    def get_frame_times_lcr_pattern_mode(self):
        if self.frame_data is None:
            print('No frame data found for this stage object.')
            return np.array([])
        assert self.sample_rate is not None, 'self.sample_rate is None, cannot proceed'
        return get_frame_times_pattern_mode(self.frame_data, bin_rate=1000.0, sample_rate=self.sample_rate, updates_per_frame=self.updates_per_frame)
    def get_frame_times_lcr_video_mode(self):
        if self.frame_data is None:
            print('No frame data found for this stage object.')
            return np.array([])
        assert self.sample_rate is not None, 'self.sample_rate is None, cannot proceed'
        return get_frame_times_lightcrafter(self.frame_data, bin_rate=1000.0, sample_rate=self.sample_rate)
    def get_frame_times_generic(self):
        if self.frame_data is None:
            print('No frame data found for this stage object.')
            return np.array([])
        assert self.sample_rate is not None, 'self.sample_rate is None, cannot proceed'
        return get_frame_times(self.frame_data, bin_rate=1000.0, sample_rate=self.sample_rate)
    def get_frame_times(self):
        if not self.frame_times_from_syncs:
            bin_rate=1000.0
            assert self.frame_rate is not None, 'self.frame_rate is None, cannot proceed'
            frame_step = bin_rate / self.frame_rate
            print('Frame rate: {}'.format(self.frame_rate))
            print('Sample rate: {}'.format(self.sample_rate))
            print('Frame step: {}'.format(frame_step))
            assert self.frame_data is not None, 'self.frame_data is None, cannot proceed'
            print('Frame data length: {}'.format(len(self.frame_data)))
            assert self.sample_rate is not None, 'self.sample_rate is None, cannot proceed'
            frame_times = np.round(np.arange(0.0,len(self.frame_data)/self.sample_rate*bin_rate,frame_step)).astype(float)
            return frame_times
        if self.device_type == 'LightCrafter':
            if self.mode == 'Pattern':
                return self.get_frame_times_lcr_pattern_mode()
            elif self.mode == 'Video':
                return self.get_frame_times_lcr_video_mode()
        else:
            return self.get_frame_times_generic()


class EpochObj(DataObj):
    def __init__(self, d: Union[dict, h5py.Group]) -> None:
        super().__init__(d=d)
        self.stage_obj = None
        self.label = None
        self.parameters = dict()
        self.backgrounds = dict()
        self.responses = list()
        self.stimuli = list()
        self.frameTimesMs = None
        if 'label' in self.attributes.keys():
            self.label = self.attributes['label']
    def add_response(self, response: ResponseObj):
        if self.responses is None:
            self.responses = list()
        self.responses.append( response.__dict__ )
    def parse_backgrounds(self, d: h5py.Group):
        backgrounds = dict()
        for key, value in d['backgrounds'].items(): #type: ignore
            key_name = strip_uuid( key )
            bg_attrs = parse_attributes( value )
            if 'dataConfigurationSpans' in value.keys():
                try: 
                    foo = parse_attributes(value['dataConfigurationSpans'][list(value['dataConfigurationSpans'].keys())[0]][key_name])
                    # Append the items to the background attributes.
                    for bg_key, bg_value in foo.items():
                        bg_attrs[bg_key] = bg_value
                        if bg_key not in self.properties.keys():
                            self.parameters[bg_key] = bg_value
                except Exception as error: 
                    print('Error: ' + str(error))
                    continue
                # Check for a Stage device.
                try:
                    if 'Stage' in key_name:
                        self.stage_obj = StageObj( key_name=key_name, properties=foo )
                except Exception as error:
                    print('Error: ' + str(error))
            backgrounds[ key_name ] = bg_attrs
        self.backgrounds = backgrounds
        for key, value in d.items():
            if isinstance(value, h5py.Group):
                self.backgrounds[key] = parse_attributes( value )
    #TODO: Figure out why this unfinished method exists
    def parse_responses(self, d: h5py.Group):
        pass

# TODO: Figure out why this unfinished function exists
class FrameObj(object):
    def __init__(self,
                 stage_obj: StageObj,
                 frame_monitor: np.ndarray) -> None:
        pass
    def parse_stage_properties(self):
        pass
    def parse_frame_monitor(self, stage_obj: StageObj, frame_monitor: np.ndarray):
        pass

class NpEncoder(json.JSONEncoder):
    """ Special JSON encoder for numpy types. 
    Source: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable"""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, bytes):
            return o.decode('UTF-8')
        if isinstance(o, h5py.Empty):
            return None
        return super(NpEncoder, self).default(o)

class Symphony2Reader:
    def __init__(self, 
                 h5_path: str, 
                 out_path: str, 
                 mea_raw_data_path: Optional[str]=None, 
                 stage_type: str='LightCrafter',
                 frame_times_from_syncs: bool=True,
                 save_h5_path: bool=True, 
                 sample_rate: float=20000.0):
        self.file = None
        self.h5_path = h5_path
        self.json_path = out_path
        self.mea_raw_data_path = mea_raw_data_path
        self.stage_type = stage_type
        self.frame_times_from_syncs = frame_times_from_syncs
        self.save_h5_path = save_h5_path
        self.metadata = None
        self.experiment = None
        self.sample_rate = sample_rate
        self.current_epoch = None
        self.array_pitch = None
        self.array_pitch_dict = dict() # Define the array pitch dictionary.
        try:
            splits = h5_path.split('/')
            self.experiment_name = splits[-1].split('.')[0]
        except:
            self.experiment_name = None

    def read_write(self, datajoint: bool = False):
        """ Write out the metadata as a JSON file. """
        self.metadata = self.read_file()

        # Only necessary when writing JSON for Symphony Data, but it does reassign
        # the value of self.experiment so we'll run it no matter what
        self.organize_metadata()
            
        if self.json_path is not None:
            assert self.metadata is not None, 'self.metadata variable is empty'
            experiment = self.create_symphony_dict( self.metadata )

            # Write out the MEA JSON for SymphonyData.
            if not datajoint:
                if self.mea_raw_data_path is not None:
                    self.write_json(self.experiment, self.json_path) 
            # Write the full experiment JSON for datajoint.
            else:
                self.write_json(experiment, self.json_path)

            # If this is an MEA experiment, export the text file.
            if self.mea_raw_data_path is not None:
                self.export_summary_text(experiment, self.json_path.replace('.json','.txt'))

    # Read in the hdf5 file.
    def read_file(self) -> dict:
        """ Read in the hdf5 file.
        Parameters:
            file_path (str): The path to the hdf5 file.
        Returns:
            metadata (dict): The metadata extracted from the H5 file. Metadata contains the following keys: dict_keys(['sources', 'group', 'project'])
        """
        self.file = h5py.File(self.h5_path, 'r')
        keys = list(self.file.keys())
        for key in keys:
            # Find the experiment level.
            if type(self.file[key]) is h5py.Group:
                metadata = self.parse_file(key)
                return metadata

        raise ValueError('No metadata found')

    # Write out the metadata as a JSON file.
    def write_json(self, metadata, out_path):
        """ Write out the metadata as a JSON file.
        Parameters:
            metadata (dict): The metadata to write out.
            out_path (str): The path to write the JSON file.
        """
        with open(out_path, 'w+') as outfile:
            json.dump(metadata, outfile, cls=NpEncoder)
    
    def create_symphony_dict(self, metadata: dict) -> dict:
        """ Create the Symphony dictionary from the metadata. 
        Parameters:
            metadata: The metadata dictionary reflecting the Symphony H5 file structure.
        Returns:
            experiment: The experiment dictionary.
        """
        # Set the rig type.
        if self.mea_raw_data_path is not None:
            rig_type = 'MEA'
        else:
            rig_type = 'PATCH'
        experiment = ExperimentObj( d = metadata['sources'][0], rig_type=rig_type ).__dict__
        exp_animals = metadata['sources']
        animals = list()
        for hh in range(len(exp_animals)):
            animal = AnimalObj( d = exp_animals[hh] ).__dict__
            exp_preparations = exp_animals[hh]['sources']
            preparations = list()
            for ii in range(len(exp_preparations)):
                preparation = PreparationObj( d = exp_preparations[ii] ).__dict__
                cell_list = list()
                cells = exp_preparations[ii]['sources']
                for jj in range( len(cells) ):
                    cell = CellObj( d = cells[jj] ).__dict__
                    print(f'Cell keys: {cell.keys()}')
                    cell['epoch_groups'] = list()
                    for kk in range(len(metadata['group'])):
                        if cell['uuid'] == metadata['group'][kk]['source']['uuid']:
                            e_group = EpochGroupObj( d = metadata['group'][kk] ).__dict__
                            e_group = clean_epoch_group_for_json( e_group = e_group )
                            cell['epoch_groups'].append( e_group )
                            # Check for an array pitch entry.
                            this_array_pitch = self.array_pitch_dict[ metadata['group'][kk]['uuid'] ]
                            if this_array_pitch is not None:
                                preparation['arrayPitch'] = this_array_pitch
                    cell_list.append( cell )
                preparation['cells'] = cell_list
                preparations.append( preparation )
            animal['preparations'] = preparations
            animals.append( animal )
        experiment['animals'] = animals
        return experiment

    def organize_metadata(self):                     
        """ Organize the metadata according to the hierarchy. """
        # Organize by protocol.
        assert self.metadata is not None, 'self.metadata variable is empty'
        protocol_labels = list()
        for group in self.metadata['group']:
            for block in group['block']:
                p_label = block['protocolID']
                if p_label not in protocol_labels:
                    protocol_labels.append(p_label)

        # Sort the labels.
        protocol_labels = sorted(protocol_labels)
        protocol_labels = np.array(protocol_labels)
        protocols = list()
        for p_label in protocol_labels:
            protocol = dict()
            protocol['label'] = p_label
            protocol['group'] = list()
            protocols.append(protocol)

        for group in self.metadata['group']:
            group_protocols = list()
            for block in group['block']:
                p_label = block['protocolID']
                if p_label not in group_protocols:
                    group_protocols.append(p_label)
            # Sort the labels.
            group_protocols = np.array(sorted(group_protocols))
            for g_label in group_protocols:
                p_idx = np.where(protocol_labels == g_label)[0][0]
                new_group = dict()
                new_group['attributes'] = group['attributes']
                new_group['label'] = group['label']
                new_group['properties'] = group['properties']
                new_group['source'] = group['source']
                new_group['block'] = list()
                for block in group['block']:
                    p_label = block['protocolID']
                    if (self.mea_raw_data_path is not None) and ('dataFile' in block.keys()):
                        block['dataFile'] = block['dataFile'].replace('.bin', '/')
                    if p_label == g_label:
                        new_group['block'].append(block)
                protocols[p_idx]['group'].append(new_group)

        self.experiment = dict()
        self.experiment['protocol'] = protocols
        self.experiment['sources'] = self.metadata['sources']

    # Get the name of the Litke data file saved for the EpochBlock
    def get_data_file_from_block(self, block: dict):
        """
        Extracts the data file name from the block dictionary.
        Parameters:
            block: Block dictionary (type: dict)
            Returns: Data file name (type: str)
        """
        dataFile = block['dataFile'].split('/')[-2:-1]
        dataFile = dataFile[0]
        return dataFile
    
    def export_summary_text(self, 
            experiment: dict, 
            filepath: str):
        """
        Exports the metadata to a text file.
        Parameters:
            experiment: The experiment dictionary.
            filepath: Full path to the output text file.
        """
        all_files = list()
        file_nums = list()
        with open(filepath, 'w') as file:
            for animal in experiment['animals']:
                file.write('Animal:' + animal['label'] + '\n')
                for preparation in animal['preparations']:
                    file.write('  Preparation:' + preparation['label'] + '\n')
                    for cell in preparation['cells']:
                        file.write('    Cell:' + cell['label'] + '\n')
                        for group in cell['epoch_groups']:
                            file.write('      Group:' + group['label'] + '\n')
                            # Loop through the blocks and find the unique protocols.
                            protocols = dict()
                            for block in group['epoch_blocks']:
                                if block['protocolID'] not in protocols.keys():
                                    protocols[ block['protocolID'] ] = list()
                                # Get the file name.
                                f_name = self.get_data_file_from_block(block)
                                print(f'Identified datafile as {f_name}\n')
                                protocols[ block['protocolID'] ].append(f_name)
                                all_files.append(f_name)
                                match = re.search(r'(\d+)', f_name)
                                if match:
                                    file_nums.append( int(match.group(0)) )
                                else:
                                    print(f'Warning: no numbers found in filename: {f_name}')
                            for key, value in protocols.items():
                                file.write('        Protocol:' + key + '\n')
                                for v in value:
                                    file.write('          ->' + v + '\n')
            # Sort the files.
            all_files = sorted( all_files )
            print('All files:', all_files)
            file.write( '\n' )
            file.write( 'All files:\n' )
            for f in all_files:
                file.write(f + ' ')
            # Find any missing files.
            file_nums = sorted( file_nums )
            print('File nums:', file_nums)
            try:
                missing = self.missing_elements( file_nums )
                if len(missing) > 0:
                    file.write( '\n' )
                    file.write( 'Missing files: ' )
                    for m in missing:
                        file.write( str(m) + ' ' )
                else:
                    file.write( '\n\n' )
                    file.write( 'No missing files (' + str(file_nums[0]) + '-' + str(file_nums[-1]) + ').' )
            except Exception as e:
                print('Error:', e)

    def missing_elements(self, L: list) -> list:
        """ Find the missing elements in a list.  
        Parameters:
            L: List of file names.
        Returns:    
            missing: List of missing names.
        """
        try:
            L = sorted(L)
            start, end = L[0], L[-1]
            missing = sorted(set(range(start, end + 1)).difference(L))
        except:
            return []
        return missing

    def parse_attributes(self, obj: h5py.Group) -> dict:
        """ Parse the attributes.
        Parameters:
            obj (h5py.Group): The attributes group.
        Returns:
            attributes (dict): The parsed attributes dictionary.
        """
        attributes = dict()
        for key, value in obj.attrs.items():
            if isinstance(value, np.bytes_):
                value = value.decode('UTF-8')
            elif isinstance(obj, h5py.Empty):
                value = None
            attributes[key] = value
        return attributes

    def parse_properties(self, obj: h5py.Group) -> dict:
        """ Parse the properties. 
        Parameters:
            obj (h5py.Group): The properties group.
        Returns:
            props (dict): The parsed properties dictionary.
        """
        props = dict()
        for key, value in obj['properties'].attrs.items():
            if isinstance(value, np.bytes_):
                value = value.decode('UTF-8')
            elif isinstance(obj, h5py.Empty):
                value = None
            props[key] = value
        return props

    def combine_parameters(self, block_params: dict, epoch_params: dict) -> dict:
        """ Combine the block and epoch parameters. 
        Parameters:
            block_params (dict): The block parameters.
            epoch_params (dict): The epoch parameters.
        Returns:
            parameters (dict): The combined parameters. """
        parameters = dict()
        for key, value in block_params.items():
            parameters[key] = value
        for key, value in epoch_params.items():
            parameters[key] = value
        parameters = dict(sorted(parameters.items()))
        return parameters

    def parse_epoch(self, epoch: h5py.Group, block_params: dict) -> dict:
        """ Parse the epoch. 
        Parameters:
            epoch (h5py.Group): The epoch group.
            block_params (dict): The block parameters.
        Returns:
            epoch_dict (dict): The parsed epoch dictionary.
        """
        self.current_epoch = EpochObj( epoch )
        self.current_epoch.parse_backgrounds( d=epoch )
        epoch_dict = self.current_epoch.__dict__.copy()
        # epoch_dict = epoch_dict.__dict__
        del epoch_dict['stage_obj']
        # Parse the protocol parameters.
        params = dict()
        for key, value in epoch['protocolParameters'].attrs.items():
            params[key] = value
        params = self.combine_parameters(block_params, params)
        epoch_dict['parameters'].update(params)
        # Parse the responses.
        if 'responses' in epoch.keys():
            responses, frame_times = self.parse_responses(epoch['responses']) # type: ignore
            epoch_dict['responses'] = responses
            epoch_dict['frameTimesMs'] = frame_times
            # Keep a copy down in properties for now, as this isn't currently transferred to datajoint.
            epoch_dict['properties']['frameTimesMs'] = frame_times
        # Parse the stimuli.
        if 'stimuli' in epoch.keys():
            stimuli = self.parse_stimuli(epoch['stimuli']) # type: ignore
            epoch_dict['stimuli'] = stimuli
        # epoch_dict['uuid'] = attributes['uuid']
        return epoch_dict

    def get_reference_string(self, value):
        """ Get the full reference string within the HDF5 file. 
        Parameters:
            value: H5 object
        Returns:
            full_path: Full reference string (type: str)
        """
        return value.name
    
    def parse_stimuli(self, stimuli: h5py.Group) -> dict:
        """ Parse the stimuli. 
        Parameters:
            stimuli (h5py.Group): The stimuli group.
        Returns:
            stimuli_dict (dict): The parsed stimuli dictionary.
        """
        stimuli_dict = dict()
        for key, value in stimuli.items():
            key_name = strip_uuid(key)
            stimuli_dict[key_name] = self.parse_attributes(value)
            # Get the full reference string within the HDF5 file.
            if self.save_h5_path:
                full_path = self.get_reference_string(value)
                stimuli_dict[key_name]['h5path'] = full_path
        return stimuli_dict

    def parse_responses(self, responses: h5py.Group) -> Tuple[dict, Optional[np.ndarray]]:
        """ Parse the responses. 
        Parameters:
            responses (h5py.Group): The responses group.
        Returns:
            response_dict (dict): The parsed responses dictionary.
            frame_times (np.ndarray): The frame times.
        """
        response_dict = dict()
        frame_times = None
        found_syncs = False
        for key, value in responses.items():
            key_name = strip_uuid(key)
            response_dict[key_name] = self.parse_attributes(value)

            # Get the full reference string within the HDF5 file.
            if self.save_h5_path:
                full_path = self.get_reference_string(value)
                response_dict[key_name]['h5path'] = full_path

            # Pull the frame monitor data.
            if ('Sync' in key) and ('data' in value.keys()):
                found_syncs = False
                red_sync = value['data']['quantity']
                sample_rate = response_dict[key_name]['sampleRate']
                assert self.current_epoch is not None, 'self.current_epoch is None'
                if self.current_epoch.stage_obj is None:
                    self.current_epoch.stage_obj = StageObj(key_name=key_name, properties=response_dict[key_name], frame_times_from_syncs=self.frame_times_from_syncs)
                self.current_epoch.stage_obj.set_sample_rate(sample_rate)
                self.current_epoch.stage_obj.set_sync_data(red_sync)

            if ('Frame' in key) and ('data' in value.keys()) and (not found_syncs):
                frame_monitor = value['data']['quantity']#[()]
                sample_rate = response_dict[key_name]['sampleRate']
                assert self.current_epoch is not None, 'self.current_epoch is None'
                if self.current_epoch.stage_obj is None:
                    self.current_epoch.stage_obj = StageObj(key_name=key_name, properties=response_dict[key_name], frame_times_from_syncs=self.frame_times_from_syncs)
                self.current_epoch.stage_obj.set_sample_rate(sample_rate)
                self.current_epoch.stage_obj.set_frame_data(frame_monitor)

                # Process the frame data.
                if self.frame_times_from_syncs:
                    frame_times = self.current_epoch.stage_obj.get_frame_times()

        return response_dict, frame_times

    def parse_data_file_name(self, data_file_string: str) -> str:
        """ Parse the data file name. 
        Parameters:
            data_file_string (str): The data file string.
        Returns:
            out_string (str): The parsed data file string.
        """
        exp_string, f_name = data_file_string.split('\\')
        f_name = f_name.replace('.bin', '/')
        assert self.experiment_name is not None, 'self.experiment_name is None'
        if exp_string[:8] == self.experiment_name[:8]:
            out_string = exp_string + '/' + f_name
        else:
            out_string = self.experiment_name + '/' + f_name
        return out_string
    
    def parse_epoch_block(self, epoch_block: dict) -> Optional[dict]:
        """ Parse an epoch block.
        Parameters:
            epoch_block (dict): The epoch block to parse.
        Returns:
            dict: The parsed epoch block.
        """
        # Ensure that the block contains epochs, otherwise return None.
        if 'epochs' not in epoch_block.keys():
            print('WARNING: No epochs found in block: ')
            return None
        elif len(epoch_block['epochs']) == 0:
            print('WARNING: No epochs found in block: ')
            return None
        
        block_obj = EpochBlockObj(d=epoch_block)

        #NOTE: Modified code below sets n_samples and array_id to None since there's no file name
        if self.mea_raw_data_path is not None:
            if 'dataFileName' not in block_obj.properties.keys():
                print('WARNING: No data file found in block: ')
                print(block_obj.__dict__)
                litke_starts = list()
                litke_ends = list()
                n_samples = None
                f_name = ''
                array_id = None
            else:
                f_name = block_obj.properties['dataFileName']
                f_name = self.parse_data_file_name(f_name)
                print(f'Parsing dataFileName: {f_name}')
                block_obj.dataFile = f_name
                assert self.experiment_name is not None, 'self.experiment_name is None'
                f_path = os.path.join(self.mea_raw_data_path, self.experiment_name, f_name.split('/')[-2])
                print(f'Datafile location: {f_path}\n')
                # Check for the old setup where we directly recorded the frame monitor.
                if int(self.experiment_name[:8]) < 20220518:
                    litke_starts, litke_ends, array_id, n_samples = get_litke_triggers_old(f_path)
                else:
                    litke_starts, litke_ends, array_id, n_samples = get_litke_triggers(f_path)

                if array_id is None or n_samples is None:
                    print(f"ERROR: Unable to parse data file {block_obj.__dict__['properties']['dataFileName']}, see error on previous line.\n")
                    return None
                else:
                    self.array_pitch = get_electrode_pitch_by_array_id(array_id = array_id)
                    # Check for unrecorded epochs.
                    if np.any(np.array(litke_starts) > n_samples):
                        unrecorded_epochs = np.argwhere(np.array(litke_starts) > n_samples).ravel()
                        print('WARNING: Unrecorded epochs found in file: ' + f_name)
                        for bad_idx in sorted(unrecorded_epochs, reverse=True):
                            litke_starts = np.delete(litke_starts, bad_idx)
                            litke_ends = np.delete(litke_ends, bad_idx)


        else:
            if 'dataFileName' not in block_obj.properties.keys():
                print('WARNING: No data file found in block: ')
                print(block_obj.__dict__)
                f_name = ''
                litke_starts = list()
                litke_ends = list()
                n_samples = None
                array_id = None
            else:
                f_name = block_obj.properties['dataFileName']
                f_name = self.parse_data_file_name(f_name)
                block_obj.dataFile = f_name
                litke_starts = list()
                litke_ends = list()
                n_samples = None
                array_id = None

        # NOTE: Old code. Keeping it around until we're cetrain the new more complete if/then statement
        # works correctly

        # if (self.mea_raw_data_path is not None) and ('dataFileName' not in block_obj.properties.keys()):
        #     print('WARNING: No data file found in block: ')
        #     print(block_obj.__dict__)
        #     litke_starts = list()
        #     litke_ends = list()
        #     f_name = ''
        # if 'dataFileName' in block_obj.properties.keys():
        #     f_name = block_obj.properties['dataFileName']
        #     f_name = self.parse_data_file_name(f_name)
        #     block_obj.dataFile = f_name
        #     if self.mea_raw_data_path is not None:
        #         assert self.experiment_name is not None, 'self.experiment_name is None'
        #         f_path = os.path.join(self.mea_raw_data_path, self.experiment_name, f_name.split('/')[-2])
        #         # Check for the old setup where we directly recorded the frame monitor.
        #         if int(self.experiment_name[:8]) < 20220518:
        #             litke_starts, litke_ends, array_id, n_samples = get_litke_triggers_old(f_path)
        #         else:
        #             litke_starts, litke_ends, array_id, n_samples = get_litke_triggers(f_path)
        #
        #         if array_id is None or n_samples is None:
        #             print('ERROR: unable to parse data file for block, no epochs found.')
        #             print(block_obj.__dict__)
        #             return None
        #         else:
        #             self.array_pitch = get_electrode_pitch_by_array_id(array_id = array_id)
        #             # Check for unrecorded epochs.
        #             if np.any(np.array(litke_starts) > n_samples):
        #                 unrecorded_epochs = np.argwhere(np.array(litke_starts) > n_samples).ravel()
        #                 print('WARNING: Unrecorded epochs found in file: ' + f_name)
        #                 for bad_idx in sorted(unrecorded_epochs, reverse=True):
        #                     litke_starts = np.delete(litke_starts, bad_idx)
        #                     litke_ends = np.delete(litke_ends, bad_idx)
        
        # Copy out the block parameters.
        params = block_obj.parameters 
        if self.array_pitch is not None:
            block_obj.arrayPitch = self.array_pitch
        # Parse the epochs.
        epoch_list = list()
        frame_times = list()
        for _, value in epoch_block['epochs'].items():
            my_epoch = self.parse_epoch(value, params)
            epoch_list.append(my_epoch)
            frame_times.append(my_epoch['frameTimesMs'])
        # Get the epochs ordered by start time.
        epoch_starts = np.zeros(len(epoch_list))
        for count, epoch in enumerate(epoch_list):
            epoch_starts[count] = epoch['attributes']['startTimeDotNetDateTimeOffsetTicks']
        epoch_order = np.argsort(epoch_starts)
        epoch_list = [epoch_list[i] for i in epoch_order]
        frame_times = [frame_times[i] for i in epoch_order]
        epoch_starts = epoch_starts[epoch_order]

        # Convert the epoch starts to a datetime.
        start_seconds = np.zeros(len(epoch_starts))
        epoch_datetime = list()
        for ii in range(len(epoch_starts)):
            date_string, date_seconds = dotnet_ticks_to_datetime(epoch_starts[ii])
            epoch_datetime.append(date_string)
            epoch_list[ii]['datetime'] = date_string
            # Get epoch start times in seconds.
            start_seconds[ii] = date_seconds

        block = block_obj.__dict__
        if self.mea_raw_data_path is not None:
            if len(litke_starts) == 0:
                print('WARNING: No Litke triggers found in file: ' + f_name)
                litke_starts = 46969 + np.floor((start_seconds - start_seconds[0])*self.sample_rate).astype(int)

                if n_samples is not None:
                    unrecorded_epochs = np.argwhere(np.array(litke_starts) > n_samples).ravel()
                    for bad_idx in sorted(unrecorded_epochs, reverse=True):
                        del epoch_list[bad_idx]
                        del frame_times[bad_idx]
                        del litke_starts[bad_idx]
                        
            # Calculate the number of samples between each epoch from the Symphony file.
            # !! How to deal with Windows updating the system clock in the middle of a block????
            if len(epoch_starts) < len(litke_starts):
                start_samples = litke_starts[0] + np.floor((start_seconds - start_seconds[0])*self.sample_rate).astype(int)
                n_epochs = np.min([len(litke_starts), len(epoch_starts)])
                dt = np.abs(start_samples[:n_epochs] - litke_starts[:n_epochs])
                if len(start_samples) > 1:
                    d_samps = np.mean(np.diff(start_samples))
                    if np.any(dt > 0.5*d_samps):
                        print('WARNING: Found missing epochs start pulses in file: ' + f_name)
                        litke_starts = start_samples[:n_epochs]
            if len(epoch_starts) > len(litke_starts):
                print(f'WARNING: More epochs ({len(epoch_starts)}) than Litke triggers ({len(litke_starts)}) found in file: ' + f_name)
                unrecorded_epochs = np.arange(len(litke_starts), len(epoch_starts))
                for bad_idx in sorted(unrecorded_epochs, reverse=True):
                    del epoch_list[bad_idx]
                    del frame_times[bad_idx]
            block['epochStarts'] = litke_starts
            block['epochEnds'] = litke_ends
            block['array_id'] = array_id
            block['n_samples'] = n_samples
            # Save down in the properties so it's persisted to the database.
            block['properties']['array_id'] = array_id
            block['properties']['n_samples'] = n_samples
            block['properties']['epochStarts'] = litke_starts
            block['properties']['epochEnds'] = litke_ends
        
        block['epoch'] = epoch_list
        block['frameTimesMs'] = frame_times
        # Save down in the properties so it's persisted to the database.
        block['properties']['frameTimesMs'] = frame_times
        return block

    def parse_epoch_group(self, epoch_group: h5py.Group) -> dict:
        """Parse the epoch group.
        Parameters:
            epoch_group: h5py.Group
        Returns:
            group_dict: dict
        """
        group_dict = dict()
        # Get the attributes.
        attributes = self.parse_attributes(epoch_group)
        group_dict['attributes'] = attributes
        group_dict['uuid'] = attributes['uuid']
        if 'label' in attributes.keys():
            group_dict['label'] = attributes['label']
            print('Parsing group: ' + attributes['label'])
        # Get the properties.
        props = dict()
        for key, value in epoch_group['properties'].attrs.items():
            props[key] = value
        group_dict['properties'] = props
        # Parse the epoch blocks.
        block_list = list()
        block_start_times = list()
        for key, value in epoch_group['epochBlocks'].items(): #type: ignore
            print(f'Parsing epoch block {key}')
            this_block = self.parse_epoch_block(value)
            if this_block is not None:
                block_start_times.append( this_block['attributes']['startTimeDotNetDateTimeOffsetTicks'] )
                block_list.append(this_block)
        self.array_pitch_dict[group_dict['uuid']] = self.array_pitch
        # Reorder the blocks based on start time.
        try:
            block_list = order_list_by_start_time(block_list, block_start_times)
        except Exception as error:
            print('Error: ' + str(error))
        group_dict['block'] = block_list
        # Get the group source.
        group_src = epoch_group['source'].attrs
        src = dict()
        if 'label' in group_src.keys():
            src['label'] = group_src['label'].decode('UTF-8') #type: ignore
        if 'uuid' in group_src.keys():
            src['uuid'] = group_src['uuid'].decode('UTF-8') #type: ignore
        group_dict['source'] = src
        return group_dict

    def parse_source(self, source: h5py.Group, parse_children: bool=True) -> dict:
        """ Parse the source group.
        Parameters:
            source (h5py.Group): The source group.
        Returns:
            source_dict (dict): The source dictionary.
        """
        source_dict = dict()
        # attributes = source.attrs
        # attrs = dict()
        # for key in attributes.keys():
        #     attrs[key] = attributes[key]
        attrs = self.parse_attributes(source)
        source_dict['attributes'] = attrs
        if 'label' in attrs.keys():
            source_dict['label'] = attrs['label']
        # props = dict()
        # for key, value in source['properties'].attrs.items():
        #     props[key] = value
        # source_dict['properties'] = props
        source_dict['properties'] = self.parse_properties(source)

        # Check for notes.
        if 'notes' in source.keys():
            notes = source['notes']
            source_dict['notes'] = self.parse_notes(notes) #type: ignore
        else:
            source_dict['notes'] = list()
        
        if parse_children:
            source_list = list()
            for _, value in source['sources'].items(): #type: ignore
                source_list.append(self.parse_source(value))
            source_dict['sources'] = source_list
        return source_dict

    def parse_notes(self, notes: h5py.Group) -> List[dict]:
        """ Parse the notes group.
        Parameters:
            notes (h5py.Group): The notes group.
        Returns:
            notes_dict (dict): The notes dictionary.
        """
        notes_list = list()
        note_text = notes['text']
        time_ticks = notes['time']['ticks'] #type: ignore
        time_offset = notes['time']['offsetHours'] #type: ignore
        for ii in range(len(note_text)): #type: ignore
            note_dict = dict()
            note_dict['text'] = note_text[ii].decode('UTF-8') #type: ignore
            note_dict['time_ticks'] = time_ticks[ii] #type: ignore
            note_dict['time_offsetHours'] = time_offset[ii] #type: ignore
            date_string, _ = dotnet_ticks_to_datetime(time_ticks[ii]) #type: ignore
            note_dict['datetime'] = date_string
            notes_list.append(note_dict)
        return notes_list
    
    def parse_experiment(self, experiment: h5py.Group) -> dict:
        """ Parse the experiment group.
        Parameters:
            experiment (h5py.Group): The experiment group.
        Returns:
            experiment_dict (dict): The parsed experiment group.
        """
        experiment_dict = dict()
        attributes = experiment.attrs
        attrs = dict()
        for key in attributes.keys():
            attrs[key] = attributes[key]
        experiment_dict['attributes'] = attrs
        if 'label' in attrs.keys():
            experiment_dict['label'] = attrs['label']
        # Get the properties.
        props = dict()
        for key, value in experiment['properties'].attrs.items():
            props[key] = value
        experiment_dict['properties'] = props

        # Check for notes.
        if 'notes' in experiment.keys():
            notes = experiment['notes']
            experiment_dict['notes'] = self.parse_notes(notes) #type: ignore
        else:
            experiment_dict['notes'] = list()

        source_list = list()
        for key, value in experiment['sources'].items(): #type: ignore
            source_list.append(self.parse_source(value))
        experiment_dict['sources'] = source_list
        return experiment_dict

    def parse_file(self, exp_key) -> dict:
        """ Parse the file for the experiment metadata.
        Parameters:
            file (h5py.File): The hdf5 file to parse.
            exp_key (str): The key for the experiment.
        Returns:
            metadata (dict): The metadata for the experiment.
        """
        metadata = dict()
        sources = self.file[exp_key]['sources'] #type: ignore
        # Get the top-level/purpose node.
        # project_dict = self.parse_source(self.file[exp_key], parse_children=False)
        # project_dict = self.parse_source(self.file[exp_key], parse_children=True)
        # attributes = self.parse_attributes(self.file[exp_key])
        # Get the experiment level.
        exp_list = list()
        group_list = list()
        group_start_times = list()
        for _, value in sources.items(): #type: ignore
            experiment = value['experiment']
            exp_sources = experiment['sources']
            print('Parsing experiment sources.')
            for _, s_value in exp_sources.items():
                exp_list.append( self.parse_experiment(s_value) )
            exp_groups = experiment['epochGroups']
            group_count = 0
            for _, g_value in exp_groups.items():
                print('Parsing group {}'.format( group_count+1 ) + ' of {}...'.format( len(exp_groups) ))
                this_group = self.parse_epoch_group( g_value )
                group_list.append( this_group )
                group_start_times.append( this_group['attributes']['startTimeDotNetDateTimeOffsetTicks'] )
                group_count += 1
        group_list = order_list_by_start_time(group_list, group_start_times)
        metadata['sources'] = exp_list
        metadata['group'] = group_list
        # metadata['project'] = [project_dict]
        return metadata

    def close(self):
        assert self.file is not None
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.file is not None
        self.file.close()

def h5_to_datajoint_json(h5_path: str, output_directory: str = '/Volumes/data-1/datajoint/mea/meta/',
                         raw_directory: str = '/Volumes/data-1/data/raw/', **kwargs):
    basename = os.path.basename(h5_path)
    exp_name, _ = os.path.splitext(basename)
    out_path = os.path.join(output_directory, f'{exp_name}.json') 
    reader = Symphony2Reader(h5_path = h5_path, out_path = out_path, mea_raw_data_path = raw_directory, **kwargs)
    reader.read_write(datajoint = True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse H5 experiment files into JSON format.')
    parser.add_argument('h5_path', type=str, help='Path to the Symphony HDF5 file (e.g., /data/data/h5/20230607C.h5).')
    parser.add_argument('out_path', type=str, help='Output path for the JSON and TXT files (e.g., /data/data/metadata/json/20230607C.json).')
    parser.add_argument('-r','--raw_data_path', default=None, type=str, help='Path to raw data (MEA experiments only; e.g., /data/data/raw/).')
    parser.add_argument('-s', '--stage-type', default='LightCrafter', type=str, help='Stage type (e.g., LightCrafter or Microdisplay).')
    parser.add_argument('-p','--save_h5_path', action='store_true', help='Whether to extract the full path to the stimuli/responses.')
    parser.add_argument('-f', '--frame_times_from_syncs', action='store_true', help='Whether to extract frame times from syncs (default: False).')

    args = parser.parse_args()

    h5_path = args.h5_path
    out_path = args.out_path
    raw_data_path = args.raw_data_path
    stage_type = args.stage_type
    save_h5_path = True #args.save_h5_path
    frame_times_from_syncs = True #args.frame_times_from_syncs

    if not h5_path.endswith('.h5'):
        path_split = h5_path.split('/')
        print('Input path must end with .h5. Cannot process {}.'.format(h5_path))
        import sys
        sys.exit(1)

    # Check the output path.
    if not out_path.endswith('.json'):
        path_split = h5_path.split('/')
        out_path = os.path.join(out_path, path_split[-1].split('.')[0] + '.json')

    print('Parsing {}.'.format(h5_path))
    print('Writing to {}.'.format(out_path))
    print('Raw data path: {}'.format(raw_data_path))


    with Symphony2Reader(h5_path=h5_path, out_path=out_path, mea_raw_data_path=raw_data_path, stage_type=stage_type, frame_times_from_syncs=frame_times_from_syncs, save_h5_path=save_h5_path) as reader:
        reader.read_write()

