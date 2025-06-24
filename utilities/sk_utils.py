import datajoint as dj
import os
import platform

if platform.system() == 'Darwin':
    if os.path.exists('/Volumes/data-1'):
        NAS_DATA_DIR = os.path.abspath('/Volumes/data-1/data/sorted')
        NAS_ANALYSIS_DIR = os.path.abspath('/Volumes/data-1/analysis')
    elif os.path.exists('/Volumes/ExternalM2/mea_ssd/'):
        NAS_DATA_DIR = os.path.abspath('/Volumes/ExternalM2/mea_ssd/data/sorted')
        NAS_ANALYSIS_DIR = os.path.abspath('/Volumes/ExternalM2/mea_ssd/analysis')
    else:
        raise Exception("No data or analysis paths found. Is the NAS or SSD connected?")
else:
    if os.path.exists('Z:/data'):
        NAS_DATA_DIR = os.path.abspath('Z:/data/sorted')
        NAS_ANALYSIS_DIR = os.path.abspath('Z:/analysis')
    elif os.path.exists('E:/mea_ssd'):
        NAS_DATA_DIR = os.path.abspath('E:/mea_ssd/data/sorted')
        NAS_ANALYSIS_DIR = os.path.abspath('E:/mea_ssd/analysis')
    else:
        raise Exception("No data or analysis paths found. Is the NAS or SSD connected?")

fields = {
    'experiment': [
        ('h5_uuid', 'uuid'),
        ('label', 'label'),
        ('properties','properties'),
        ('attributes', 'attributes'),
        ('start_time', 'start_time'),
        ('experimenter', 'experimenter'),
        ('institution', 'institution'),
        ('lab', 'lab'),
        ('project', 'project'),
        ('rig', 'rig'),
        ('rig_type', 'rig_type')
    ],
    'animal': [
        ('h5_uuid', 'uuid'),
        ('label', 'label'),
        ('properties','properties'),
        ('attributes', 'attributes'),
        ('start_time', 'start_time'),
        ('props_id', 'id'),
        ('description', 'description'),
        ('sex', 'sex'),
        ('age', 'age'),
        ('weight', 'weight'),
        ('dark_adaptation', 'darkAdaptation'),
        ('species', 'species')
    ],
    'preparation': [
        ('h5_uuid', 'uuid'),
        ('label', 'label'),
        ('properties','properties'),
        ('attributes', 'attributes'),
        ('start_time', 'start_time'),
        ('bath_solution', 'bathSolution'),
        ('preparation_type', 'preparationType'),
        ('region', 'region'),
        ('array_pitch', 'arrayPitch')
    ],
    'cell': [
        ('h5_uuid', 'uuid'),
        ('label', 'label'),
        ('properties','properties'),
        ('attributes', 'attributes'),
        ('start_time', 'start_time'),
        ('type', 'type'),
    ],
    'epoch_group': [
        ('h5_uuid', 'uuid'),
        ('label', 'label'),
        ('properties','properties'),
        ('attributes', 'attributes'),
        ('start_time', 'start_time'),
        ('end_time', 'end_time'),
    ],
    'epoch_block': [
        ('h5_uuid', 'uuid'),
        ('label', 'label'),
        ('properties','properties'),
        ('attributes', 'attributes'),
        ('start_time', 'start_time'),
        ('end_time', 'end_time'),
        ('parameters', 'parameters'),
        ('array_pitch', 'arrayPitch')
    ],
    'epoch': [
        ('h5_uuid', 'uuid'),
        ('label', 'label'),
        ('properties','properties'),
        ('attributes', 'attributes'),
        ('start_time', 'start_time'),
        ('end_time', 'end_time'),
        ('parameters', 'parameters'),
    ],
    'response': [
        ('h5_uuid', 'uuid'),
        ('label', 'label'),
        ('sample_rate', 'sampleRate'),
        ('sample_rate_units', 'sampleRateUnits'),
        ('offset_hours', 'inputTimeDotNetDateTimeOffsetOffsetHours'),
        ('offset_ticks', 'inputTimeDotNetDateTimeOffsetTicks'),
    ],
}

def table_dict(Experiment: dj.Manual, Animal: dj.Manual, Preparation: dj.Manual,
               Cell: dj.Manual, EpochGroup: dj.Manual,
               EpochBlock: dj.Manual, Epoch: dj.Manual, 
               Response: dj.Manual, Stimulus: dj.Manual,
               Tags: dj.Manual) -> dict:
    return {
        'experiment': Experiment,
        'animal': Animal,
        'preparation': Preparation,
        'cell': Cell,
        'epoch_group': EpochGroup,
        'epoch_block': EpochBlock,
        'epoch': Epoch,
        'response': Response,
        'stimulus': Stimulus,
        'tags': Tags
    }

table_arr = ['experiment', 'animal', 'preparation', 'cell', 'epoch_group', 'epoch_block', 'epoch', 'response', 'stimulus']

def child_table(table_name: str) -> str:
    return None if table_name == 'response' else table_arr[table_arr.index(table_name) + 1]

def parent_table(table_name: str) -> str:
    return None if table_name == 'experiment' else table_arr[table_arr.index(table_name) - 1]