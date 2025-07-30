import datajoint as dj

dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'simple'

try:
    dj.conn().connect()
    B_CONNECTED = True
    schema = dj.schema('schema')
except Exception as e:
    print(f"Could not connect to DataJoint database: {e}")
    B_CONNECTED = False
    schema = lambda x: x  # Dummy schema for testing purposes


@schema
class Protocol(dj.Manual):
    definition = """
    # protocol information
    protocol_id: int auto_increment
    ---
    name: varchar(255)
    """

# central schema: directly from h5 files
# for now going to be mostly empty! just connectors to each other

@schema
class Experiment(dj.Manual):
    definition = """
    # experiment metadata, including pointers to files
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    exp_name: varchar(255)
    meta_file: varchar(255)
    data_file: varchar(255) # empty if MEA for now, maybe should store "/Volumes/data/data/sorted" here?
    tags_file: varchar(255)
    is_mea: tinyint unsigned # 1 if MEA, 0 if not
    date_added: timestamp
    label: varchar(255)
    properties: json
    attributes: json
    start_time = NULL : timestamp
    experimenter = NULL : varchar(255)
    institution = NULL : varchar(255)
    lab = NULL : varchar(255)
    project = NULL : varchar(255)
    rig = NULL : varchar(255)
    rig_type = NULL : varchar(255)
    """

@schema
class Animal(dj.Manual):
    definition = """
    # animal information
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    -> Experiment.proj(experiment_id='id')
    -> Experiment.proj(parent_id='id')
    label: varchar(255)
    properties: json
    attributes: json
    start_time = NULL : timestamp
    props_id = NULL : varchar(255)
    description = NULL : varchar(255)
    sex = NULL : varchar(255)
    age = NULL : varchar(255)
    weight = NULL : varchar(255)
    dark_adaptation = NULL : varchar(255)
    species = NULL : varchar(255)
    """

@schema
class Preparation(dj.Manual):
    definition = """
    # preparation information
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    -> Experiment.proj(experiment_id='id')
    -> Animal.proj(parent_id='id')
    label: varchar(255)
    properties: json
    attributes: json
    start_time = NULL : timestamp
    bath_solution = NULL : varchar(255)
    preparation_type = NULL : varchar(255)
    region = NULL : varchar(255)
    array_pitch = NULL : varchar(255)
    """

@schema
class Cell(dj.Manual):
    definition = """
    # cell information
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    -> Experiment.proj(experiment_id='id')
    -> Preparation.proj(parent_id='id')
    label: varchar(255)
    properties: json
    attributes: json
    start_time = NULL : timestamp
    type = NULL : varchar(255)
    """

@schema
class EpochGroup(dj.Manual):
    definition = """
    # epoch group information
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    -> Experiment.proj(experiment_id='id')
    -> Cell.proj(parent_id='id')
    -> Protocol
    label = NULL : varchar(255)
    properties: json
    attributes: json
    start_time = NULL : timestamp
    end_time = NULL : timestamp
    """

# analysis table
@schema
class SortingChunk(dj.Manual):
    definition = """
    # sorting chunk information: algorithm generated
    id: int auto_increment
    ---
    -> Experiment.proj(experiment_id='id')
    chunk_name: varchar(255)
    """

# analysis table
@schema
class SortedCell(dj.Manual):
    definition = """
    # sorted cell information: algorithm generated
    id: int auto_increment
    ---
    -> SortingChunk.proj(chunk_id='id')
    algorithm: varchar(200) # should be directory name
    cluster_id: int
    """

# extra fields for sorted cell:
# - STAfit: lots of fields here, could include time course as well?
# - Spike count
# - more ideas/things they use?


# NEW TABLE: BlockSortedCell (inherit from SortedCell and EpochBlock)
# - Spike count
# - % ISI violations, figure out what to keep: full binned spike data, or just preset cutoff thing
# - 

# analysis table
@schema
class CellTypeFile(dj.Manual):
    definition = """
    # cell typing file: human generated
    id: int auto_increment
    ---
    -> SortingChunk.proj(chunk_id='id')
    algorithm: varchar(200) # should be directory name
    file_name: varchar(255) # name of sorting file
    """

# analysis table
@schema
class SortedCellType(dj.Manual):
    definition = """
    # sorted cell type information: human generated
    id: int auto_increment
    ---
    -> SortedCell.proj(sorted_cell_id='id')
    -> CellTypeFile.proj(file_id='id')
    cell_type: varchar(255)
    """

@schema
class EpochBlock(dj.Manual):
    definition = """
    # epoch block information
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    data_dir: varchar(255) # only for MEA
    -> Experiment.proj(experiment_id='id')
    -> EpochGroup.proj(parent_id='id')
    -> Protocol
    -> [nullable] SortingChunk.proj(chunk_id='id')
    label = NULL : varchar(255)
    properties: json
    attributes: json
    start_time = NULL : timestamp
    end_time = NULL : timestamp
    parameters = NULL : json
    array_pitch = NULL : varchar(255)
    """

@schema
class Epoch(dj.Manual):
    definition = """
    # epoch information
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    -> Experiment.proj(experiment_id='id')
    -> EpochBlock.proj(parent_id='id')
    label = NULL : varchar(255)
    properties: json
    attributes: json
    start_time = NULL : timestamp
    end_time = NULL : timestamp
    parameters = NULL : json
    """

@schema
class Response(dj.Manual):
    definition = """
    # response information
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    -> Epoch.proj(parent_id='id')
    device_name: varchar(255)
    h5path: varchar(511)
    label = NULL : varchar(255)
    sample_rate = NULL : varchar(255)
    sample_rate_units = NULL : varchar(255)
    offset_hours = NULL : varchar(255)
    offset_ticks = NULL : varchar(255)
    """

@schema
class Stimulus(dj.Manual):
    definition = """
    # stimulus information
    id: int auto_increment
    ---
    h5_uuid: varchar(255)
    -> Epoch.proj(parent_id='id')
    device_name: varchar(255)
    h5path: varchar(511)
    """

# misc. peripheral schema
@schema
class Tags(dj.Manual):
    definition = """
    # tagging information
    tag_id: int auto_increment
    ---
    h5_uuid: varchar(255) # id of object in h5 file
    -> Experiment.proj(experiment_id='id')
    table_name: varchar(255) # name of table in database
    table_id: int # id of object in database table
    user: varchar(63) # name of profile who made this tag: could be a name or anything else
    tag: varchar(255) # tag: THIS SHOULD CHANGE. For now, comma separated list.
    """