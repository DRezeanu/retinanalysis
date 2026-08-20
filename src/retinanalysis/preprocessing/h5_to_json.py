"""H5 parsing functions for converting Symphony HDF5 files
into dictionaries and JSON files that can be used by 
retinanalysis's datajoint database. 
"""
from retinanalysis._config import config
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta, timezone
from .frame_parser import detect_flips
import json
from .raw_data_loader import load_raw_data

# Translating between convoluted names in H5 file and simple versions
# that we use in our JSON dict
TIME_DICT = {
    'creationOffset' : 'creationTimeDotNetDateTimeOffsetOffsetHours',
    'creationTime' : 'creationTimeDotNetDateTimeOffsetTicks',
    'startOffset' : 'startTimeDotNetDateTimeOffsetOffsetHours',
    'startTime' : 'startTimeDotNetDateTimeOffsetTicks',
    'endOffset' : 'endTimeDotNetDateTimeOffsetOffsetHours',
    'endTime' : 'endTimeDotNetDateTimeOffsetTicks',
}

# Required because old parse_data renames 'preparation' to
# 'preparationType' when creating the _dj JSON. To maintain
# parity with what database_pop expects and what is in the old
# JSON files, we need to do the same here.
PREP_FIELDS = {
    'bathSolution' : 'bathSolution',
    'preparationType' : 'preparation',
    'region' : 'region',
}

def parse_attrs(group: h5py.Group):
    """Helper function to clean and convert HDF5 Group attrs into
    a dictionary.
    """
    return {key: clean_dtypes(val) for key, val in group.attrs.items()}

def parse_experiment(
    h5_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    """Core function that parses a single experiment's H5 file into a dictionary
    and optionally exports that dictionary as a JSON. This function also prints out
    the number of epoch_blocks skipped for one reason or another (missing datafile,
    no epochs in the epoch_block, etc.)

    Args:
        h5_path: path to the h5 file to parse.
        output_dir: Optional output directory path. If provided, a sanitized JSON
            will be written to outpath/h5_filename.json

    Returns:
        dictionary representation of the h5 file
    """
    result = {
        'label': None,
        'uuid': None,
        'start_time': None,
        'end_time': None,
        'experimenter': None,
        'institution': None,
        'lab': None,
        'project': None,
        'rig': None,
        'rig_type': None,
        'properties': {},
        'attributes': {},
        'notes': [],
        'animals': [],
    }


    with h5py.File(h5_path, 'r') as f:
        exp_key = [k for k in f.keys() if k.startswith('experiment-')][0]
        experiment = f[exp_key]
        assert isinstance(experiment, h5py.Group)

        # Parse attributes
        attrs = parse_attrs(experiment)
        result['attributes'] = attrs
        result['uuid'] = attrs['uuid']

        # Convert dotNet ticks and offset to datetime
        start_ticks = attrs[TIME_DICT['startTime']]
        start_offset = attrs[TIME_DICT['startOffset']]
        assert isinstance(start_ticks, int)
        assert isinstance(start_offset, float)
        result['start_time'] = ticks_to_datetime(start_ticks, start_offset)


        # Parse end_time separately (because it can be None/missing)
        result['end_time'] = parse_end_time(attrs)

        # Parse properties into dictionary
        if 'properties' in experiment:
            result['properties'] = parse_attrs(experiment['properties']) #type: ignore 

        # Elevate certain properties to top level properties dict
        for field in ['experimenter', 'institution', 'lab', 'project', 'rig']:
            if field in result['properties']:
                result[field] = result['properties'][field]

        # Parse notes if there are any
        if 'notes' in experiment:
            result['notes'] = parse_notes(experiment['notes']) #type: ignore 

        animal_container = experiment['sources']
        assert isinstance(animal_container, h5py.Group)
        
        # Start recursive cascade through sources. Each level (animal, prep, cell)
        # has certain properties that datajoint expects at the top level, so we manually
        # elevate those.
        animals = []
        for key in animal_container:
            # returns a dictionary with 'sources'
            animal = parse_source(animal_container[key]) #type: ignore 

            for field in ['id', 'description', 'sex', 'age', 'weight', 'darkAdaptation', 'species']:
                if field in animal['properties']:
                    animal[field] = animal['properties'][field]
            
            # Rename sources → preparations → cells
            animal['preparations'] = animal.pop('sources')
            for prep in animal['preparations']:

                for field, prop_key in PREP_FIELDS.items():
                    if prop_key in prep['properties']:
                        prep[field] = prep['properties'][prop_key]

                prep['cells'] = prep.pop('sources')
                for cell in prep['cells']:

                    if 'type' in cell['properties']:
                        cell['type'] = cell['properties']['type']

                    del cell['sources']

            animals.append(animal)

        result['animals'] = animals
        
        # Start epoch parsing cascade. We parse each epoch_group, which triggers
        # parsing each block in that group, which triggers parsing each epoch
        # in that block.
        epoch_groups = []
        for key in experiment['epochGroups']: #type: ignore
            eg = parse_epoch_group(experiment['epochGroups'][key]) #type: ignore
            epoch_groups.append(eg)

        epoch_groups.sort(key=lambda eg: eg['attributes'][TIME_DICT['startTime']])

        # parse_epoch_block returns None for a block with no epochs, so those vanish here
        # and we print a helpful message for the use so they know which blocks were dropped
        h5_blocks = sum(len(experiment['epochGroups'][key].get('epochBlocks', {})) #type: ignore
                        for key in experiment['epochGroups']) #type: ignore
        dropped = h5_blocks - sum(len(eg['epoch_blocks']) for eg in epoch_groups)
        if dropped:
            print(f'{Path(h5_path).stem}: dropped {dropped} of {h5_blocks} epoch blocks (no epochs)')

        # Create lookup dicts that associate each cell and prep with their respective UUIDs
        cell_lookup = {}
        prep_lookup = {}
        for animal in animals:
            for prep in animal['preparations']:
                for cell in prep['cells']:
                    cell['epoch_groups'] = []
                    cell_lookup[cell['uuid']] = cell
                    prep_lookup[cell['uuid']] = prep

        # Use cell lookup dict to associate each epoch group with its cell. All
        # epoch groups should be created inside the cell source, NOT the prep source.
        # We don't check for this explicitly as of now, but if blocks were created under
        # prep by accident, open the H5 back up in symphony and manually fix it.
        for eg in epoch_groups:
            cell_uuid = eg['sourceUuid']
            cell_lookup[cell_uuid]['epoch_groups'].append(eg)

            # Use prep lookup dict to add 'arrayPitch' to the prep level
            for eb in eg['epoch_blocks']:
                if 'arrayPitch' in eb:
                    prep_lookup[cell_uuid]['arrayPitch'] = eb['arrayPitch']
                    break

        # Set rig type by looking for 'arrayPitch', which only gets set
        # if there is a 'datafile'
        if any('arrayPitch' in prep for prep in prep_lookup.values()):
            result['rig_type'] = 'MEA' 
        else:
            result['rig_type'] = 'PATCH'

    # Set label using experiment name from h5 filename. Previous parser
    # accidentally pulled all 'experiment' level data from animal source[0]
    # instead of the actual experiment attrs. Experiment level doesn't actually
    # have a 'label' attr, but datajoint expects it now.
    label = Path(h5_path).stem
    result['label'] = label

    # It output directory provided, sanitize the dict and export to JSON
    if output_dir is not None:
        sanitized = _json_sanitize(result)
        output_path = Path(output_dir) / f'{label}.json'
        with open(output_path, 'w') as json_file:
            json.dump(sanitized, json_file, allow_nan=False)

    return result


def parse_source(source: h5py.Group) -> dict:
    """Helper function that parses Symphony H5 'sources' Animal, Preparation,
    and Cell. The function parses sub-sources recursively, so only Animal level
    must be submitted.

    Args:
        source: h5py.Group object for a Symphony H5 source (animal, prep, cell).

    Returns:
        dict with all properties and attributes parset from Symphony H5 source group
    """
    result = {
        'label' : None,
        'uuid' : None,
        'attributes' : {},
        'start_time' : None,
        'keywords' : [],
        'resources' : {},
        'properties' : {},
        'notes' : [],
        'sources' : [],
    }


    # Parse attributes
    attrs = parse_attrs(source)
    result['attributes'] = attrs
    result['label'] = attrs['label']
    result['uuid'] = attrs['uuid']

    # Parse dotNet ticks and offset into datetime strings
    ticks = attrs[TIME_DICT['creationTime']]
    offset = attrs[TIME_DICT['creationOffset']]
    assert isinstance(ticks, int)
    assert isinstance(offset, float)
    creation_time = ticks_to_datetime(ticks, offset)
    result['start_time'] = creation_time


    # Parse keywords if they exist (usually no)
    if 'keywords' in attrs:
        result['keywords'] = attrs['keywords']

    # Parse properties into a dict
    if 'properties' in source:
        result['properties'] = parse_attrs(source['properties']) #type: ignore 

    # Parse notes into nested dict
    if 'notes' in source:
        result['notes'] = parse_notes(source['notes']) #type: ignore 

    # Parse sub-sources recursively. If animal, subsource is prep, if prep
    # subsource is cell. Cells have no sub-sources.
    if 'sources' in source:
        for key in source['sources']: #type: ignore 
            child = source['sources'][key] #type: ignore 
            result['sources'].append(parse_source(child)) #type: ignore 

    return result


def parse_epoch_group(epoch_group: h5py.Group) -> dict:
    """Helper function that parses Symphony H5 'epoch group' group. These groups
    should all be contained inside the Cell source, but all epoch_groups are parsed
    regardless.

    Args:
        epoch_group: h5py.Group object for a Symphony H5 epoch_group

    Returns:
        dict with all properties and attributes parset from H5 epoch
    """
    result = {
        'label': None,
        'uuid': None,
        'start_time': None,
        'end_time': None,
        'attributes': {},
        'sourceUuid': None,
        'keywords': [],
        'properties': {},
        'resources': {},
        'notes': [],
        'epoch_blocks': [],
        'epoch_groups': [],   # nested epoch groups
    }
    
    # Parse attributes
    attrs = parse_attrs(epoch_group)
    result['attributes'] = attrs
    result['label'] = attrs['label']
    result['uuid'] = attrs['uuid']

    # Parse dotNet ticks and offset into datetime strings
    start_ticks = attrs[TIME_DICT['startTime']]
    start_offset = attrs[TIME_DICT['startOffset']]
    assert isinstance(start_ticks, int)
    assert isinstance(start_offset, float)
    result['start_time'] = ticks_to_datetime(start_ticks, start_offset)

    # Parse end time separately because it can be None/missing
    result['end_time'] = parse_end_time(attrs)

    # Add source UUID from parent epoch_group
    result['sourceUuid'] = parse_attrs(epoch_group['source'])['uuid'] #type: ignore

    # Parse keywords if they exist (usually no)
    if 'keywords' in attrs:
        result['keywords'] = attrs['keywords']

    # Parse notes into nested dict
    if 'notes' in epoch_group:
        result['notes'] = parse_notes(epoch_group['notes']) #type: ignore

    # Parse propeties into dict
    if 'properties' in epoch_group:
        result['properties'] = parse_attrs(epoch_group['properties']) #type: ignore

    # Recurse into epoch blocks
    if 'epochBlocks' in epoch_group:
        for key in epoch_group['epochBlocks']: #type: ignore
            eb = epoch_group['epochBlocks'][key] #type: ignore
            block = parse_epoch_block(eb) #type: ignore
            if block is not None:
                result['epoch_blocks'].append(block)

    # Sort epoch_blocks by start time
    result['epoch_blocks'].sort(key=lambda eb: eb['attributes'][TIME_DICT['startTime']])

    # Recurse into nested Epoch Groups (should always be empty)
    if 'epochGroups' in epoch_group:
        for key in epoch_group['epochGroups']: #type: ignore
            eg = epoch_group['epochGroups'][key] #type: ignore
            result['epoch_groups'].append(parse_epoch_group(eg)) #type: ignore

    return result


def parse_epoch_block(epoch_block: h5py.Group):
    """Helper function that parses Symphony H5 'epoch block' group. Every epoch group
    will have one or more epoch blocks inside it. 

    Args:
        epoch_block: h5py.Group object for a Symphony H5 epoch block

    Return:
        dict with all properties and attributes parset from H5 epoch block
        or None if the block contains zero epochs.
    """
    if len(epoch_block.get('epochs', {})) == 0: #type: ignore
        return None

    result = {
        'label' : None,
        'protocolID' : None,
        'uuid': None,
        'start_time': None,
        'end_time': None,
        'dataFile': None,
        'attributes': {},
        'keywords' : [],
        'properties': {},
        'parameters': {},
        'resources': {},
        'notes': [],
        'epochs' : [],
    }

    # Parse attributes
    attrs = parse_attrs(epoch_block)
    result['attributes'] = attrs
    result['protocolID'] = attrs['protocolID']
    result['uuid'] = attrs['uuid']

    # Parse dotNet ticks and offset into datetime string
    start_ticks = attrs[TIME_DICT['startTime']]
    start_offset = attrs[TIME_DICT['startOffset']]
    assert isinstance(start_ticks, int)
    assert isinstance(start_offset, float)
    result['start_time'] = ticks_to_datetime(start_ticks, start_offset)

    # Parse end_time separately because it can be None/missing
    result['end_time'] = parse_end_time(attrs)

    # Parse keywords if they exist (usually no)
    if 'keywords' in attrs:
        result['keywords'] = attrs['keywords']

    # Parse properties into dict
    if 'properties' in epoch_block:
        result['properties'] = parse_attrs(epoch_block['properties']) #type: ignore

    # Parse datafile name and convert from ('exp\\datafile.bin') format into
    #('exp/datafile/') that describes the stub of its path.
    if 'dataFileName' in result['properties']:
        raw_name = result['properties']['dataFileName']
        assert isinstance(raw_name, str)
        exp_name, bin_file = raw_name.split('\\')
        result['dataFile'] = exp_name + '/' + bin_file.replace('.bin', '/')

    # Parse protocolParameters into 'parameters' dict
    if 'protocolParameters' in epoch_block:
        result['parameters'] = parse_attrs(epoch_block['protocolParameters']) #type: ignore

    # Parse notes to nested dict
    if 'notes' in epoch_block:
        result['notes'] = parse_notes(epoch_block['notes']) #type: ignore

    # Parse all epochs inside this epoch_block
    if 'epochs' in epoch_block:
        for key in epoch_block['epochs']: #type: ignore
            epoch = epoch_block['epochs'][key] #type: ignore
            result['epochs'].append(parse_epoch(epoch, result['parameters'])) #type: ignore

    # Sort epochs dict by start_time
    result['epochs'].sort(key=lambda ep: ep['attributes'][TIME_DICT['startTime']])

    # Pull frame times from each epoch into a list of lists inside the
    # epoch_block properties
    result['properties']['frameTimesMs'] = [
        ep['properties'].get('frameTimesMs', []) for ep in result['epochs']
    ]

    # If there's a datafile, parse the raw bin files for array_id, n_samples,
    # epoch starts and epoch ends. Far and away the slowest step because 
    # epoch starts and ends need the TTL signal which requires parsing the full
    # 400-500GB set of raw .bin files
    if result['dataFile'] is not None:
        raw_file_path = Path(config.RAW_DIR) / result['dataFile']
        raw_data_container = load_raw_data(str(raw_file_path), ttl_only = True)
        result['arrayPitch'] = _get_array_pitch(raw_data_container.array_id)
        result['properties']['array_id'] = raw_data_container.array_id
        result['properties']['n_samples'] = raw_data_container.n_samples
        result['properties']['epochStarts'] = raw_data_container.epoch_starts.tolist()
        result['properties']['epochEnds'] = raw_data_container.epoch_ends.tolist()

    return result

def parse_data_config_spans(epoch: h5py.Group) -> dict:
    """Parse device configuration settings contained in the dataConfigurationSpans
    property of the background property inside each epoch. We keep devices in backgrounds
    and dataConfigurationSpans holds important info like 'micronsPerPixel' and 'ndfs'.

    Only the span node matching the background's device name is taken here, controller
    and port nodes (HEKA/Nidac, ao0, doport1, etc.) are ignored.

    Only the first span is read. A device cannot be reconfigured mid-epoch, so any
    additional spans would describe the same configuration. This matches parse_data.

    Config is kept namespaced by device because keys collide across devices: 'ndfs' and
    'lightPath' exist on both an LED and the LightCrafter Stage, and which NDFs are
    installed where is real experimental metadata. Callers that need the flat _dj-style
    parameter dict should merge these in device order and accept that later devices win.

    Args:
        epoch: h5py.Group object for an individual Symphony H5 epoch

    Returns:
        dict mapping device name to that device's configuration dict
    """
    result = {}
    if 'backgrounds' not in epoch:
        return result

    for bg_key in epoch['backgrounds']: #type: ignore
        background = epoch['backgrounds'][bg_key] #type: ignore
        device_name = strip_uuid(bg_key)
        if 'dataConfigurationSpans' not in background: #type: ignore
            continue
        spans = background['dataConfigurationSpans'] #type: ignore

        first_span = spans[list(spans.keys())[0]] #type: ignore
        if device_name in first_span: #type: ignore
            result[device_name] = parse_attrs(first_span[device_name]) #type: ignore

    return result


def parse_epoch(
    epoch: h5py.Group,
    epoch_block_params: dict,
) -> dict:
    """Helper function that parses Symphony H5 'epoch' group. Every epoch block 
    will have one or more epochs inside it. 

    Args:
        epoch: h5py.Group object for a Symphony H5 epoch

    Return:
        dict with all properties and attributes parset from H5 epoch
    """
    result = {
        'label': None,
        'uuid': None,
        'start_time' : None,
        'end_time' : None,
        'frameTimesMs': [],
        'attributes': {},
        'keywords': [],
        'properties': {},
        'parameters': {},
        'responses': {},
        'backgrounds': {},
        'stimuli': {},
        'notes': [],
    }

    # Parse attributes
    attrs = parse_attrs(epoch)
    result['attributes'] = attrs
    result['uuid'] = attrs['uuid']

    # Parse dotNet ticks and offset into datetime string
    start_ticks = attrs[TIME_DICT['startTime']]
    start_offset = attrs[TIME_DICT['startOffset']]
    assert isinstance(start_ticks, int)
    assert isinstance(start_offset, float)
    result['start_time'] = ticks_to_datetime(start_ticks, start_offset)

    # Parse end_time separately because it can be None/missing
    result['end_time'] = parse_end_time(attrs)

    # Parse keywords if they exist (usually no)
    if 'keywords' in attrs:
        result['keywords'] = attrs['keywords']

    # Parse epoch properties into a dict
    if 'properties' in epoch:
        result['properties'] = parse_attrs(epoch['properties']) #type: ignore

    # Parse protocol Parameters into a 'properties' dict inside the epoch
    if 'protocolParameters' in epoch:
        result['parameters'] = parse_attrs(epoch['protocolParameters']) #type: ignore

    # Parse device configurations
    span_config = parse_data_config_spans(epoch)

    # Merge in the epoch_block parameters like the old code did
    merged_params = epoch_block_params.copy()
    merged_params.update(result['parameters'])

    # Device config is merged last, flattened across devices so later devices win on
    # colliding keys ('ndfs', 'lightPath'). In practice this means that devices with 
    # the first name alphabetically (Frame Monitor comes before Green LED) have these
    # properties written to the epoch at the top epoch['properties'] level,  while
    # the accurate per-device values are stored at the result['backgrounds'] level, which
    # is NOT imported by Datajoint. If you want access to more than one device's 
    # ndf configurator setting, we will need to change datajoint_pop.py to ingest 'backgrounds'
    for device_config in span_config.values():
        for key, value in device_config.items():
            if key not in result['properties']:
                merged_params[key] = value

    # Sort parameter dict alphabetically
    result['parameters'] = dict(sorted(merged_params.items()))

    # Prase the 'responses' group
    fm_group = None
    fm_device = None
    if 'responses' in epoch:
        for key in epoch['responses']: #type: ignore
            response = epoch['responses'][key] #type: ignore
            device_name = strip_uuid(key)
            result['responses'][device_name] = parse_attrs(response) #type: ignore
            result['responses'][device_name]['h5path'] = response.name

            # Pull frame monitor group for parsing
            if ('Frame' in key) and ('data' in response): #type: ignore
                fm_group = response
                fm_device = device_name

    # Parse frame monitor flips
    if fm_group is not None:
        fm_trace = fm_group['data']['quantity'][:] #type: ignore
        sample_rate = result['responses'][fm_device]['sampleRate']
        flip_times, _, _ = detect_flips(fm_trace, sample_rate, f_cutoff = 120.0)
        result['frameTimesMs'] = flip_times.tolist()
        result['properties']['frameTimesMs'] = flip_times.tolist()

    # Parse 'backgrounds'
    if 'backgrounds' in epoch:
        for key in epoch['backgrounds']: #type: ignore
            background = epoch['backgrounds'][key] #type: ignore
            device_name = strip_uuid(key)
            bg_attrs = parse_attrs(background) #type: ignore
            bg_attrs.update(span_config.get(device_name, {}))
            result['backgrounds'][device_name] = bg_attrs

    # Parse 'stimuli' group
    if 'stimuli' in epoch:
        for key in epoch['stimuli']: #type: ignore
            stimulus = epoch['stimuli'][key] #type: ignore
            result['stimuli'][strip_uuid(key)] = parse_attrs(stimulus) #type: ignore
            result['stimuli'][strip_uuid(key)]['h5path'] = stimulus.name

    return result


def parse_notes(notes_dataset: h5py.Dataset) -> list:
    """Helper function for parsing the 'notes' Dataset contained in all levels
    of the Symphony H5. Each entry in the notes dataset has 'text' and 'time' 
    fields, and the time field has 'ticks' (dotNet ticks) and 'offsetHours' also
    dotNet, indicating time zone offset).

    Args:
        notes_dataset: 'notes' h5py.Dataset object from inside a Symphony H5 group. 

    Returns:
        list of dicts, with one dict per note in the notes Dataset
    """
    notes_list = []
    for entry in notes_dataset[:]:
        note = {
            'text': entry['text'].decode('UTF-8'),
            'time_ticks' : int(entry['time']['ticks']),
            'time_offsetHours' : float(entry['time']['offsetHours']),
            'datetime': ticks_to_datetime(
                int(entry['time']['ticks']),
                float(entry['time']['offsetHours'])
            ),
        }
        notes_list.append(note)
    return notes_list

def clean_dtypes(val):
    """Helper function for decoding and sanitizing the data types contained in
    the Symphony H5. The outputs are meant to prevent issues writing the final
    data to a JSON.

    Args:
        val: data of type np.bytes_, np.floating, np.integer, np.ndarray or 
            h5py.Empty

    Returns:
        val: data sanitized into UTF-8, float, int, list, or None, respectively
    """
    if isinstance(val, np.bytes_):
        return val.decode('UTF-8')

    elif isinstance(val, np.floating):
        return float(val)

    elif isinstance(val, np.integer):
        return int(val)

    elif isinstance(val, np.ndarray):
        return [clean_dtypes(d) for d in val]

    elif isinstance(val, h5py.Empty):
        return None

    else:
        return val

def parse_end_time(attrs: dict) -> str | None:
    """End time for a node, or None if Symphony never closed it.

    A session that was aborted or crashed leaves a node with no endTime attrs at all --
    ticks and offset go missing together. Seen at epoch group and epoch block level
    (2026-07-02_E has one of each); start times are never absent, so those stay a hard
    requirement. parse_data records None for these, so matching it keeps _dj parity.

    Args:
        attrs: parsed attributes of an experiment, epoch group, epoch block or epoch.
    """
    end_ticks = attrs.get(TIME_DICT['endTime'])
    if end_ticks is None:
        return None

    end_offset = attrs[TIME_DICT['endOffset']]
    assert isinstance(end_ticks, int)
    assert isinstance(end_offset, float)
    return ticks_to_datetime(end_ticks, end_offset)

def ticks_to_datetime(ticks: int, offset_hours:float=0.0) -> str:
    """Convert dotNetTicks + dotNetOffset values into a datetime object
    with the appropriate format used by retinanalysis datajoint parser.

    Args:
        ticks: integer dotNet ticks
        offset_hours: floating point dotNet offset in hours

    Returns:
        datetime string with format "%m/%d/%Y %H:%M:%S:%f"
    """
    # Start with .NET base datetime (Jan 1, 0001) and add ticks
    date_time = datetime(1, 1, 1) + timedelta(microseconds=ticks // 10)

    # Apply timezone offset in hours
    time_zone = timezone(timedelta(hours=offset_hours))
    date_time = date_time.replace(tzinfo=time_zone)

    return date_time.strftime("%m/%d/%Y %H:%M:%S:%f")

def get_h5_path(exp_name: str):
    """Helper function for pulling h5 path from experiment name
    using currently active retinanalysis.config.H5_DIR. File must be
    named 'exp_name.h5' and must exist in the H5_DIR.

    Args:
        exp_name: string representing an experiment such as '20260506C'

    Raises:
        FileNotFoundError if H5_DIR/exp_name.h5 doesn't exist
    """
    path = Path(config.H5_DIR) / f'{exp_name}.h5'
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} does not exist.'
        )

    return path

def strip_uuid(string:str):
    """Helper function for stripping the 5-section alphanumeric UUID 
    from a device or group name in the H5.
    """
    return '-'.join(string.split('-')[:-5])

def _get_array_pitch(array_id: int) -> str:
    """Hardcoded helper function for getting electrode pitch from array id.
    This is apparently standard Litke numbering (all < 1501 = 60um, etc.)

    Args:
        array_id: integer array ID. In Rieke lab we mostly use ID 504, which
            has 512 electrodes and a 60um pitch.

    Returns:
        electrode pitch as a string
    """
    if array_id < 1501:
        pitch = '60um'
    elif array_id < 3501:
        pitch = '30um'
    else:
        pitch = '120um'
    return pitch

def _json_sanitize(val):
    """Recursively sanitize all values inside a dictionary so they don't choke JSON.dump.
    This essentially involves replacing any 'NaN' or 'Inf' values with None.
    """
    if isinstance(val, dict):
        return {k: _json_sanitize(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_json_sanitize(v) for v in val]
    elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    else:
        return val
