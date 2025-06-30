import datajoint as dj
import json
import os
import datetime
from tqdm import tqdm
from retinanalysis.settings import mea_config

Experiment: dj.Manual = None
Animal: dj.Manual = None
Preparation: dj.Manual = None
Cell: dj.Manual = None
EpochGroup: dj.Manual = None
EpochBlock: dj.Manual = None
Epoch: dj.Manual = None
Response: dj.Manual = None
Stimulus: dj.Manual = None
Protocol: dj.Manual = None
Tags: dj.Manual = None

SortingChunk: dj.Manual = None
SortedCell: dj.Manual = None
CellTypeFile: dj.Manual = None
SortedCellType: dj.Manual = None

NAS_DATA_DIR = mea_config['data'] 
NAS_ANALYSIS_DIR = mea_config['analysis']

db: dj.VirtualModule = None
user = mea_config['user']

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

def make_table_dict(Experiment: dj.Manual, Animal: dj.Manual, Preparation: dj.Manual,
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

def fill_tables():
    if not db:
        print("ERROR")
        return
    global Experiment, Animal, Preparation, Cell, EpochGroup, EpochBlock, Epoch, Response, Stimulus
    global Protocol, Tags, SortingChunk, SortedCell, CellTypeFile, SortedCellType
    global table_dict
    Experiment = db.Experiment
    Animal = db.Animal
    Preparation = db.Preparation
    Cell = db.Cell
    EpochGroup = db.EpochGroup
    EpochBlock = db.EpochBlock
    Epoch = db.Epoch
    Response = db.Response
    Stimulus = db.Stimulus

    Protocol = db.Protocol
    Tags = db.Tags

    SortingChunk = db.SortingChunk
    SortedCell = db.SortedCell
    CellTypeFile = db.CellTypeFile
    SortedCellType = db.SortedCellType

    table_dict = make_table_dict(Experiment, Animal, Preparation, Cell, EpochGroup, 
                                  EpochBlock, Epoch, Response, Stimulus, Tags)

def max_id(table: dj.Manual) -> int:
    return dj.U().aggr(table, max=f'max(id)').fetch1('max')

def build_tuple(base_tuple: dict, level: str, meta: dict) -> dict:
    for dj_name, meta_name in fields[level]:
        if meta_name in meta.keys() and meta[meta_name] is not None:
            field_obj = table_dict[level].heading.attributes[dj_name]
            if field_obj.type == 'timestamp':
                # currently in string form, example "01/22/2021 09:33:51:729159"
                base_tuple[dj_name] = datetime.datetime.strptime(
                    meta[meta_name], '%m/%d/%Y %H:%M:%S:%f')
            elif field_obj.numeric:
                if type(meta[meta_name]) == str:
                    if '.' in meta[meta_name]:
                        base_tuple[dj_name] = float(meta[meta_name])
                    else:
                        base_tuple[dj_name] = int(meta[meta_name])
                else:
                    base_tuple[dj_name] = meta[meta_name]
            else:
                # must be a string or json object, just assign directly
                base_tuple[dj_name] = meta[meta_name]
    return base_tuple

# database populator methods: from analysis
def append_sorting_files(chunk_id: int, algorithm: str, sorting_dir: str):
    p1 = os.path.split(sorting_dir)
    p2 = os.path.split(p1[0])
    p3 = os.path.split(p2[0])
    analysis_dir = os.path.join(NAS_ANALYSIS_DIR, p3[1], p2[1], p1[1])
    # check if real path
    if not os.path.exists(analysis_dir):
        return
    for file in os.listdir(analysis_dir):
        if file.endswith('.txt'):
            CellTypeFile.insert1({"chunk_id": chunk_id, "algorithm": algorithm, "file_name": file})
            file_id = max_id(CellTypeFile)
            cell_types = []
            try: 
                with open(os.path.join(analysis_dir, file)) as f:
                    for line in f:
                        # each line is cluster_id (two spaces) cell_type
                        cluster_id, cell_type = line.split()
                        sorted_cell_id = (SortedCell & f"chunk_id={chunk_id}" & f"algorithm='{algorithm}' " & f"cluster_id={cluster_id}").fetch1()['id']
                        cell_types.append({"sorted_cell_id": sorted_cell_id, "file_id": file_id, "cell_type": cell_type})
            except Exception as e:
                print(f"Error reading cell typing file {file}: {e}")
                continue
            SortedCellType.insert(cell_types)

def append_sorting_chunk(experiment_id: int, chunk_name: str, chunk_path: str):
    SortingChunk.insert1({'experiment_id': experiment_id, 'chunk_name': chunk_name})
    chunk_id = max_id(SortingChunk)
    for algorithm in os.listdir(chunk_path):
        if 'kilosort' not in algorithm:
            print(f'Populator not implemented for {algorithm}')
            continue
        
        algorithm_dir = os.path.join(chunk_path, algorithm)
        if 'cluster_KSLabel.tsv' not in os.listdir(algorithm_dir):
            print(f"Could not find cluster_KSLabel.tsv in {algorithm_dir}")
            continue

        cluster_list = []
        with open(os.path.join(algorithm_dir, 'cluster_KSLabel.tsv')) as f:
            # tsv where first column is "cluster_id", add each one to the database
            for line in f:
                if line.startswith('cluster_id'):
                    continue
                cluster_id = int(line.split('\t')[0])
                ### THIS NEXT LINE IS VERY IMPORTANT: CLUSTER_ID IS ZERO-INDEXED IN THIS ONE LOCATION.
                ### BUT EVERYWHERE ELSE IT IS ONE-INDEXED BECAUSE MATLAB IS ONE-INDEXED.
                ### SO HERE WE WILL ADD ONE TO THE CLUSTER_IDS AND USE THAT AS THE SOURCE-OF-TRUTH.
                cluster_id += 1
                cluster_list.append({"chunk_id": chunk_id, "algorithm": algorithm, "cluster_id": cluster_id})

        SortedCell.insert(cluster_list)
        append_sorting_files(chunk_id, algorithm, algorithm_dir)

def append_experiment_analysis(experiment_id: int, exp_name: str):
    print(f"Adding analysis for experiment {experiment_id}, {exp_name}")
    # exp_name = (Experiment & f"id={experiment_id}").fetch1()['data_file']
    # exp_name = os.path.basename(exp_name)[:-3]
    if exp_name not in os.listdir(NAS_DATA_DIR):
        print(f"Could not find data directory for experiment {exp_name}")
        return
    
    experiment_dir = os.path.join(NAS_DATA_DIR, exp_name)
    print(f"Looking in {experiment_dir}")
    for file in os.listdir(experiment_dir):
        if os.path.isdir(os.path.join(experiment_dir, file)) and not file.startswith('data'):
            append_sorting_chunk(experiment_id, file, os.path.join(experiment_dir, file))

# given a data directory (ending in dataXXX) and the experiment id, find the correct chunk ID.
def get_block_chunk(experiment_id: int, data_dir: str) -> int:
    # data_index = data_dir.split("/")[1]
    data_index = os.path.basename(data_dir)
    possible_chunks = (SortingChunk & f"experiment_id={experiment_id}").fetch()['chunk_name']
    exp_name = (Experiment & f"id={experiment_id}").fetch1('exp_name')
    # exp_name = os.path.basename(exp_name)[:-3]
    experiment_dir = os.path.join(NAS_DATA_DIR, exp_name)
    for chunk_name in possible_chunks:
        f = os.path.join(experiment_dir, f"{exp_name}_{chunk_name}.txt")
        if not os.path.exists(f):
            print(f"ERROR: could not find chunk file: {f}")
            continue
        with open(f) as file:
            if data_index in file.read():
                return (SortingChunk & f"experiment_id={experiment_id}" & f"chunk_name='{chunk_name}'").fetch1()['id']
    print(f"ERROR: could not find a chunk for this data directory: {data_dir}")
    return None

# database populator methods
def append_protocol(protocol_name: str) -> int:
    if not (Protocol & f"name='{protocol_name}'"):
        Protocol.insert1({
            'name': protocol_name
        })
    return (Protocol & f"name='{protocol_name}'").fetch1()['protocol_id']

# def append_tags(h5_uuid: str, experiment_id: int, table_name: str, table_id: int, user: str, tags_dict: dict):
#     if tags_dict and h5_uuid in tags_dict.keys():
#         if 'tags' in tags_dict[h5_uuid].keys() and user in tags_dict[h5_uuid]['tags'].keys():
#             Tags.insert1({
#                 'h5_uuid': h5_uuid,
#                 'experiment_id': experiment_id,
#                 'table_name': table_name,
#                 'table_id': table_id,
#                 'user': user,
#                 'tag': tags_dict[h5_uuid]['tags'][user]
#             })
#         return tags_dict[h5_uuid]
#     return None

# expects: tags_dict = {h5_uuid: {tags: [(user, tag), ...]}}
# if user specified, only append tags from other users. if null, append all tags

def append_tags(h5_uuid: str, experiment_id: int, table_name: str, table_id: int, user_skip: str, tags_dict: dict):
    if tags_dict and h5_uuid in tags_dict.keys() and 'tags' in tags_dict[h5_uuid].keys():
        for user, tag in tags_dict[h5_uuid]['tags']:
            if user_skip and user == user_skip:
                continue
            Tags.insert1({
                'h5_uuid': h5_uuid,
                'experiment_id': experiment_id,
                'table_name': table_name,
                'table_id': table_id,
                'user': user,
                'tag': tag
            })
        return tags_dict[h5_uuid]
    return None

def append_response(epoch_id: int, device_name: str, response: dict, is_mea: bool):
    # Response.insert1({
    #     'h5_uuid': response['uuid'],
    #     'parent_id': epoch_id,
    #     'device_name': device_name,
    #     'h5path': response['h5path'] if not is_mea else ''
    # })
    base_tuple = {
        'parent_id': epoch_id,
        'device_name': device_name,
        'h5path': response['h5path']
    }
    Response.insert1(build_tuple(base_tuple, 'response', response))

def append_stimulus(epoch_id: int, device_name: str, stimulus: dict, is_mea: bool):
    Stimulus.insert1({
        'h5_uuid': stimulus['uuid'],
        'parent_id': epoch_id,
        'device_name': device_name,
        'h5path': stimulus['h5path']
    })

def append_epoch(experiment_id: int, parent_id: int, epoch: dict, user: str, tags: dict, is_mea: bool):
    # Epoch.insert1({
    #     'h5_uuid': epoch['attributes']['uuid'],
    #     'experiment_id': experiment_id,
    #     'parent_id': parent_id,
    #     'properties': epoch['properties'],
    #     'parameters': epoch['parameters']
    # })
    base_tuple = {
        'experiment_id': experiment_id,
        'parent_id': parent_id
    }
    Epoch.insert1(build_tuple(base_tuple, 'epoch', epoch))
    epoch_id = max_id(Epoch)
    append_tags(epoch['attributes']['uuid'], experiment_id, 'epoch', epoch_id, None, tags)
    for device_name in epoch['responses'].keys():
        append_response(epoch_id, device_name, epoch['responses'][device_name], is_mea)
    for device_name in epoch['stimuli'].keys():
        append_stimulus(epoch_id, device_name, epoch['stimuli'][device_name], is_mea)

def append_epoch_block(experiment_id: int, parent_id: int, epoch_block: dict, user: str, tags: dict, is_mea: bool):
    # EpochBlock.insert1({
    #     'h5_uuid': epoch_block['attributes']['uuid'],
    #     'data_dir': epoch_block['dataFile'] if is_mea else '',
    #     'experiment_id': experiment_id,
    #     'parent_id': parent_id,
    #     'protocol_id': append_protocol(epoch_block['protocolID']),
    #     'chunk_id': get_block_chunk(experiment_id, epoch_block['dataFile']) if is_mea else ''
    # })
    # Get the chunk_id from the data directory.
    if is_mea:
        data_xxx = epoch_block['dataFile'].split('/')[1]
        exp_name = (Experiment & f"id={experiment_id}").fetch1('exp_name')
        # exp_name = os.path.basename(exp_name)[:-3]
        data_dir = os.path.join(exp_name, data_xxx)
    else:
        data_dir = ''
    
    try:
        chunk_id = ''
        if is_mea:
            # Check that spike sorted outputs exist for this Experiment
            if os.path.exists(os.path.join(NAS_DATA_DIR, exp_name)):
                chunk_id = get_block_chunk(experiment_id, data_dir)
    except Exception as e:
        print(f"Error getting chunk_id for {experiment_id}, {data_dir}: {e}")
        chunk_id = ''

    base_tuple = {
        'experiment_id': experiment_id,
        'parent_id': parent_id,
        'data_dir': data_dir, #epoch_block['dataFile'] if is_mea else '',
        'protocol_id': append_protocol(epoch_block['protocolID']),
        'chunk_id': chunk_id #get_block_chunk(experiment_id, epoch_block['dataFile']) if is_mea else ''
    }
    EpochBlock.insert1(build_tuple(base_tuple, 'epoch_block', epoch_block))
    epoch_block_id = max_id(EpochBlock)
    tags = append_tags(epoch_block['attributes']['uuid'], experiment_id, 'epoch_block', epoch_block_id, None, tags)
    for epoch in epoch_block['epochs']:
        append_epoch(experiment_id, epoch_block_id, epoch, user, tags, is_mea)

def append_epoch_group(experiment_id: int, parent_id: int, epoch_group: dict, user: str, tags: dict, is_mea: bool):
    # first, check if every block has the same protocol_id
    single_protocol = True
    prev_protocol = None
    for epoch_block in epoch_group['epoch_blocks']:
        if prev_protocol == None:
            prev_protocol = epoch_block['protocolID']
        elif prev_protocol != epoch_block['protocolID']:
            single_protocol = False
            break
        else:
            prev_protocol = epoch_block['protocolID']
    
    base_tuple = {
        'experiment_id': experiment_id,
        'parent_id': parent_id
    }

    if single_protocol and epoch_group['epoch_blocks']:
        protocol_id = append_protocol(epoch_group['epoch_blocks'][0]['protocolID'])
    else:
        protocol_id = append_protocol("no_group_protocol")
    base_tuple['protocol_id'] = protocol_id

    EpochGroup.insert1(build_tuple(base_tuple, 'epoch_group', epoch_group))

    epoch_group_id = max_id(EpochGroup)
    tags = append_tags(epoch_group['attributes']['uuid'], experiment_id, 'epoch_group', epoch_group_id, None, tags)
    for epoch_block in epoch_group['epoch_blocks']:
        append_epoch_block(experiment_id, epoch_group_id, epoch_block, user, tags, is_mea)

def append_cell(experiment_id: int, parent_id: int, cell: dict, user: str, tags: dict, is_mea: bool):
    # Cell.insert1({
    #     'h5_uuid': cell['uuid'],
    #     'experiment_id': experiment_id,
    #     'parent_id': parent_id,
    #     'label': cell['label'],
    #     'properties': cell['properties']
    # })
    base_tuple = {
        'experiment_id': experiment_id,
        'parent_id': parent_id,
    }
    Cell.insert1(build_tuple(base_tuple, 'cell', cell))
    cell_id = max_id(Cell)
    tags = append_tags(cell['uuid'], experiment_id, 'cell', cell_id, None, tags)
    for epoch_group in cell['epoch_groups']:
        append_epoch_group(experiment_id, cell_id, epoch_group, user, tags, is_mea)

def append_preparation(experiment_id: int, parent_id: int, preparation: dict, user:str, tags: dict, is_mea: bool):
    # Preparation.insert1({
    #     'h5_uuid': preparation['uuid'],
    #     'experiment_id': experiment_id,
    #     'parent_id': parent_id,
    #     'label': preparation['label'],
    #     'properties': preparation['properties']
    # })
    base_tuple = {
        'experiment_id': experiment_id,
        'parent_id': parent_id,
    }
    Preparation.insert1(build_tuple(base_tuple, 'preparation', preparation))
    preparation_id = max_id(Preparation)
    tags = append_tags(preparation['uuid'], experiment_id, 'preparation', preparation_id, None, tags)
    for cell in preparation['cells']:
        append_cell(experiment_id, preparation_id, cell, user, tags, is_mea)

def append_animal(experiment_id: int, parent_id: int, animal: dict, user: str, tags: dict, is_mea: bool):
    # Animal.insert1({
    #     'h5_uuid': animal['uuid'],
    #     'experiment_id': experiment_id,
    #     'parent_id': parent_id,
    #     'label': animal['label'],
    #     'properties': animal['properties']
    # })
    base_tuple = {
        'experiment_id': experiment_id,
        'parent_id': parent_id,
    }
    Animal.insert1(build_tuple(base_tuple, 'animal', animal))
    animal_id = max_id(Animal)
    tags = append_tags(animal['uuid'], experiment_id, 'animal', animal_id, None, tags)
    for preparation in animal['preparations']:
        append_preparation(experiment_id, animal_id, preparation, user, tags, is_mea)

def append_experiment(meta: str, data: str, tags: str, experiment: dict, user: str, tags_dict: dict):
    exp_name = os.path.basename(data)[:-3]
    base_tuple = {
        'exp_name': exp_name,
        'meta_file': meta,
        'data_file': data,
        'tags_file': tags,
        'is_mea': 1 if experiment['rig_type'] == 'MEA' else 0,
        'date_added': datetime.datetime.now(),
    }
    Experiment.insert1(build_tuple(base_tuple, 'experiment', experiment))
    # Experiment.insert1({
    #     'h5_uuid': experiment['uuid'],
    #     'label': experiment['label'],
    #     'properties': experiment['properties']
    # })
    experiment_id = max_id(Experiment)
    if experiment['rig_type'] == 'MEA':
        try:
            append_experiment_analysis(experiment_id, exp_name)
        except Exception as e:
            print(f"Error adding analysis for experiment {experiment_id}: {e}")
    tags_dict = append_tags(experiment['uuid'], experiment_id, 'experiment', experiment_id, None, tags_dict)
    for animal in experiment['animals']:
        append_animal(experiment_id, experiment_id, animal, user, tags_dict,
                           experiment['rig_type'] == 'MEA')

# dummy method for now, will implement later.
# If there are files to parse, throws error for now.
def parse_data(source: str, dest: str):
    if source.endswith('.h5'):
        print(f'Need to convert {source} to json')
        print("going to implement this eventually")

def gen_tags(file_to_create: str, dir: str):
    # file_to_create is the name of the file to create, with the .json extension.
    # dir is the directory to create the file in.
    # create an empty '{}' json file in the directory with the given name.
    with open(os.path.join(dir, file_to_create), 'w') as f:
        f.write('{}')

# returns a list of [meta_file, data_file, tag_file] tuples in the directory
def gen_meta_list(data_dir: str, meta_dir: str, tags_dir: str) -> list:
    stack = [data_dir]
    meta_list = []

    while stack:
        current_dir = stack.pop()
        for item in os.listdir(current_dir):
            full_path = os.path.join(current_dir, item)
            if os.path.isdir(full_path):
                stack.append(full_path)
            else:
                if item.endswith('.h5'):
                    # check for meta
                    meta_file = os.path.join(meta_dir, item[:-3] + '.json')
                    if not os.path.exists(meta_file):
                        parse_data(full_path, meta_dir)
                        # As parse_data is not implemented, we will skip this file for now.
                        continue
                    # check for tags
                    tags_file = os.path.join(tags_dir, item[:-3] + '.json')
                    if not os.path.exists(tags_file):
                        gen_tags(item[:-3] + '.json', tags_dir)
                    meta_list.append([meta_file, full_path, tags_file])
    
    # that should be all of the single cell. Now for MEA, we want to find dir in NAS_DATA_DIR
    for item in os.listdir(meta_dir):
        if item.endswith('.json') and item[:-5] + '.h5' not in os.listdir(data_dir):
            # check for tags
            tags_file = os.path.join(tags_dir, item[:-5] + '.json')
            if not os.path.exists(tags_file):
                gen_tags(item[:-5] + '.json', tags_dir)
            # Check that NAS directory exists
            if not os.path.exists(NAS_DATA_DIR):
                print(f"Could not find NAS_DATA_DIR: {NAS_DATA_DIR}")
                print('Make sure you are connected and that api/helpers/utils.py has the correct path.')
                continue
            
            # find the right directory in NAS_DATA_DIR
            if item[:-5] not in os.listdir(NAS_DATA_DIR):
                print(f"Could not find data directory for {item}")
                continue
            meta_list.append([os.path.join(meta_dir, item), item[:-5], tags_file])
    return meta_list

# entrance method to generate database from a directory
def append_data(data_dir: str, meta_dir: str, tags_dir: str, username: str, db_param: dj.VirtualModule):
    global db
    global user
    db = db_param
    user = username
    fill_tables()

    meta_list = gen_meta_list(data_dir, meta_dir, tags_dir)
    records_added = 0
    for meta, data, tags in tqdm(meta_list):
        # check if meta already in database
        if len(Experiment & {'meta_file' : meta}) == 1:
            print(f"Already in database: {meta}")
            continue
        
        print("Adding", meta, flush=True)
        # not in database, add to database
        with open(meta, 'r') as f:
            meta_dict = json.load(f)
        with open(tags, 'r') as f:
            tags_dict = json.load(f)
        append_experiment(meta, data, tags, meta_dict, user, tags_dict)
        records_added += 1
    return records_added

def reload_celltypefiles():
    # TODO: currently takes 8 min for me.
    # Could be optimized as collecting all text files takes <1min - VR. 
    global db
    db = dj.VirtualModule('schema.py', 'schema')
    fill_tables()
    
    CellTypeFile.delete(safemode=False)
    
    df_sc = SortingChunk().fetch(format='frame')
    ls_text_files = []
    print(f'Found {len(df_sc)} sorting chunks in database.')
    print('Adding CellTypeFile entries for each chunk...')
    for chunk_id in tqdm(df_sc.index):
        experiment_id = df_sc.at[chunk_id, 'experiment_id']
        exp_name = (Experiment()& f"id={experiment_id}").fetch1('exp_name')
        chunk_name = df_sc.at[chunk_id, 'chunk_name']

        chunk_path = os.path.join(NAS_DATA_DIR, exp_name, chunk_name)
        for file in os.listdir(chunk_path):
            algorithm = os.listdir(chunk_path)[0]
            algorithm_dir = os.path.join(chunk_path, algorithm)
            append_sorting_files(chunk_id, algorithm, algorithm_dir)
            p1 = os.path.split(algorithm_dir)
            p2 = os.path.split(p1[0])
            p3 = os.path.split(p2[0])
            analysis_dir = os.path.join(NAS_ANALYSIS_DIR, p3[1], p2[1], p1[1])
            if not os.path.exists(analysis_dir):
                continue
            for file in os.listdir(analysis_dir):
                if file.endswith('.txt'):
                    ls_text_files.append(os.path.join(analysis_dir, file))
    print(f'Found {len(ls_text_files)} text files in analysis directories.')
    print(f'Added {len(CellTypeFile())} entries in CellTypeFile.')
