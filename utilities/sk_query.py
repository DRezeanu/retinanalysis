import datajoint as dj
import json
import os
import numpy as np
import datetime
from utilities import sk_utils
import base64
from io import BytesIO
from tqdm import tqdm
import h5py
from matplotlib.figure import Figure

NAS_DATA_DIR = sk_utils.NAS_DATA_DIR
NAS_ANALYSIS_DIR = sk_utils.NAS_ANALYSIS_DIR

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

db: dj.VirtualModule = None
table_arr: list = sk_utils.table_arr
table_dict: dict = None
user: str = None
query: dj.expression.QueryExpression = None

def fill_tables(username: str, db_param: dj.VirtualModule):
    global db, user
    if not db or not user:
        user = username
        db = db_param
    global Experiment, Animal, Preparation, Cell, EpochGroup, EpochBlock, Epoch, Response, Stimulus, Protocol, Tags
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
    table_dict = sk_utils.table_dict(Experiment, Animal, Preparation, Cell, EpochGroup, 
                                  EpochBlock, Epoch, Response, Stimulus, Tags)

# get table names ordered by hierarchy
def query_levels():
    return table_arr

# get table fields and their types (date/json/string/numeric), list of tuples (field, type)
def table_fields(table_name: str, username: str, db_param: dj.VirtualModule) -> list:
    if not table_dict:
        fill_tables(username, db_param)
    table: dj.Manual = table_dict[table_name] if table_name in table_dict.keys() else None
    if not table:
        return None
    tuples = []
    for field in table.heading.attributes.keys():
        if table.heading.attributes[field].type == 'timestamp':
            tuples.append((field, 'date'))
        elif table.heading.attributes[field].json:
            tuples.append((field, 'json'))
        elif table.heading.attributes[field].string:
            tuples.append((field, 'string'))
        elif table.heading.attributes[field].numeric:
            tuples.append((field, 'numeric'))
        else:
            print(f"Unknown type for field {field}")
            return None
    if table_name in ['epoch_block', 'epoch_group']:
        tuples.append(('protocol_name', 'string'))
    return tuples

def saved_queries(download_dir: str) -> dict:
    if not os.path.exists(os.path.join(download_dir, 'query.json')):
        return {}
    # if it does, open the json file and read in the keys
    with open(os.path.join(download_dir, 'query.json'), 'r') as f:
        queries = json.load(f)
        return queries
    
def add_query(query_name: str, query_obj: dict, download_dir: str):
    queries = saved_queries(download_dir)
    queries[query_name] = query_obj
    with open(os.path.join(download_dir, 'query.json'), 'w') as f:
        json.dump(queries, f)

def delete_query(query_name: str, download_dir: str):
    queries = saved_queries(download_dir)
    if query_name in queries.keys():
        queries.pop(query_name)
    with open(os.path.join(download_dir, 'query.json'), 'w') as f:
        json.dump(queries, f)

# given cond = {type, value}
def process_condition(table_name: str, cond: dict):
    if cond['type'] == 'TAG':
        return (Tags & f'table_name="{table_name}"' & cond['value']).proj(id='table_id')
    else:
        return cond['value']

def apply_conditions(conds: dict, table_name: str) -> list:
    cur_cond = []
    if not conds:
        return cur_cond
    type = list(conds.keys())[0]
    if type == 'COND':
        cur_cond.append(process_condition(table_name, conds['COND']))
        return cur_cond
    for entry in conds[type]:
        cur_cond.extend(apply_conditions(entry, table_name))
    if type == 'AND' or type == 'NOT':
        cur_cond = dj.AndList(cur_cond)
        if type == 'NOT':
            cur_cond = [dj.Not(cur_cond)]
    return cur_cond

# helper for exec query that actually processes the query object
def process_query(query_obj: dict) -> dj.expression.QueryExpression:
    query = Experiment
    for table in table_arr[:-2]:
        # apply conditions
        if table in query_obj.keys():
            if table in ['epoch_group', 'epoch_block']:
                # add protocol if necessary, rename primary key afterwards
                query = query * table_dict[table] * Protocol.proj(protocol_name = 'name')
                if query_obj[table]:
                    query = query & apply_conditions(query_obj[table], table)
                query = query.proj(**{f'{table}_protocol_id':'protocol_id'})
            else:
                if table != 'experiment':
                    query = query * table_dict[table]
                if query_obj[table]:
                    query = query & apply_conditions(query_obj[table], table)
        # merge down
        query = query.proj(**{f'{table}_id':'id'}) * table_dict[sk_utils.child_table(table)].proj(**{f'{table}_id':'parent_id'})
    return query.proj(response_id='id')

# entry method, initializes the database values and then parses query object
def create_query(query_obj: dict, username: str, db_param: dj.VirtualModule) -> dj.expression.QueryExpression:
    global query
    fill_tables(username, db_param)
    if not db:
        return False
    query = process_query(query_obj)
    return query

# once a query has been run, we want to generate a tree to display the results.
# the actual fields can be narrowed down later (in terms of what needs to be displayed), doesn't matter right now.
# format: {object:{...}, children:[{object:{...}, children:[...]}, {object:{...}, children:[...]}, ...]}
def generate_tree(query: dj.expression.QueryExpression, 
                  exclude_levels: list, # set to [] to include everything
                  include_meta: bool = False, # includes metadata + responses + stimuli
                  cur_level: int = 0) -> list:
    if cur_level == 7:
        return []
    children = []
    if cur_level == 0:
        iter_obj = tqdm(np.unique(query.fetch(f'{table_arr[cur_level]}_id')))
    else:
        iter_obj = np.unique(query.fetch(f'{table_arr[cur_level]}_id'))
    for entry in iter_obj:
        if table_arr[cur_level] in exclude_levels:
            children.extend(generate_tree(query & f"{table_arr[cur_level]}_id={entry}", 
                                          exclude_levels, include_meta, cur_level + 1))
        else:
            child = {}
            obj: dict = ((table_dict[table_arr[cur_level]] & f"id={entry}"
                            ).fetch(as_dict=True) if table_arr[cur_level] != 'epoch_group' and table_arr[cur_level] != 'epoch_block' else (
                                (table_dict[table_arr[cur_level]] & f"id={entry}") * Protocol.proj(protocol_name = 'name')
                                ).fetch(as_dict=True))[0]
            child['level'] = table_arr[cur_level]
            child['id'] = obj['id']
            if child['level'] == 'experiment':
                child['is_mea'] = obj['is_mea']
            else:
                child['experiment_id'] = obj['experiment_id']
            if 'label' in obj.keys():
                child['label'] = obj['label']
            if 'protocol_name' in obj.keys():
                child['protocol'] = obj['protocol_name']
            if include_meta:
                child['object'] = (table_dict[table_arr[cur_level]] & f"id={entry}"
                                ).fetch(as_dict=True) if table_arr[cur_level] != 'epoch_group' and table_arr[cur_level] != 'epoch_block' else (
                                    (table_dict[table_arr[cur_level]] & f"id={entry}") * Protocol.proj(protocol_name = 'name')).fetch(as_dict=True)
            child['tags'] = (Tags & f'table_name="{table_arr[cur_level]}"' & f'table_id={entry}').proj('user', 'tag').fetch(as_dict=True)
            if table_arr[cur_level] == 'epoch':
                child['children'] = []
                if include_meta:
                    child['responses'] = (Response & f'parent_id={entry}').fetch(as_dict=True)
                    child['stimuli'] = (Stimulus & f'parent_id={entry}').fetch(as_dict=True)
            else:
                child['children'] = generate_tree(
                    query & f"{table_arr[cur_level]}_id={entry}",
                    exclude_levels, include_meta, cur_level + 1)
            children.append(child)
    return children

# Generate the full obejct tree, including responses and stimuli
def generate_object_tree(query: dj.expression.QueryExpression, 
                  exclude_levels: list, 
                  cur_level: int = 0) -> list:
    """ Generate the full object tree, including responses and stimuli. 
    Args:
        query (dj.expression.QueryExpression): The query to execute
        exclude_levels (list): List of levels to exclude
        cur_level (int): The current level of the tree
    Returns:
        list: The object tree
    """
    if cur_level == 7:
        return []
    children = []
    for entry in np.unique(query.fetch(f'{table_arr[cur_level]}_id')):
        if table_arr[cur_level] in exclude_levels:
            children.extend(generate_object_tree(query & f"{table_arr[cur_level]}_id={entry}", exclude_levels, cur_level + 1))
        else:
            child = {}
            obj: dict = ((table_dict[table_arr[cur_level]] & f"id={entry}"
                            ).fetch(as_dict=True) if table_arr[cur_level] != 'epoch_group' and table_arr[cur_level] != 'epoch_block' else (
                                (table_dict[table_arr[cur_level]] & f"id={entry}") * Protocol.proj(protocol_name = 'name')
                                ).fetch(as_dict=True))[0]
            child['level'] = table_arr[cur_level]
            child['id'] = obj['id']
            if child['level'] == 'experiment':
                child['is_mea'] = obj['is_mea']
            else:
                child['experiment_id'] = obj['experiment_id']
            if 'label' in obj.keys():
                child['label'] = obj['label']
            if 'protocol_name' in obj.keys():
                child['protocol'] = obj['protocol_name']
            child['object'] = (table_dict[table_arr[cur_level]] & f"id={entry}"
                            ).fetch(as_dict=True) if table_arr[cur_level] != 'epoch_group' and table_arr[cur_level] != 'epoch_block' else (
                                (table_dict[table_arr[cur_level]] & f"id={entry}") * Protocol.proj(protocol_name = 'name')).fetch(as_dict=True)
            child['tags'] = (Tags & f'table_name="{table_arr[cur_level]}"' & f'table_id={entry}').proj('user', 'tag').fetch(as_dict=True)
            if table_arr[cur_level] == 'epoch':
                child['children'] = []
                child['responses'] = (Response & f'parent_id={entry}').fetch(as_dict=True)
                child['stimuli'] = (Stimulus & f'parent_id={entry}').fetch(as_dict=True)
            else:
                child['children'] = generate_object_tree(query & f"{table_arr[cur_level]}_id={entry}", exclude_levels, cur_level + 1)
            children.append(child)
    return children

# results methods: going to keep them here for now for simplicity

def get_metadata_helper(level: str, id: int) -> dict:
    return (table_dict[level] & f"id={id}").fetch1()

def get_options(level:str, id: int, experiment_id: int) -> dict:
    if level == 'epoch':
        h5_file, is_mea = (Experiment & f'id={experiment_id}').fetch1('data_file', 'is_mea')
        if is_mea:
            return None
        responses = []
        for item in (Response & f'parent_id={id}').fetch(as_dict=True):
            responses.append({'label': item['device_name'], 
                              'h5_path': item['h5path'],
                              'h5_file': h5_file,
                              'vis_type': 'epoch-singlecell'})
        stimuli = []
        for item in (Stimulus & f'parent_id={id}').fetch(as_dict=True):
            stimuli.append({'label': item['device_name'], 
                              'h5_path': item['h5path'],
                              'h5_file': h5_file,
                              'vis_type': 'epoch-singlecell'})
        return {'responses': responses, 'stimuli': stimuli}
    elif level == 'epoch_block':
        is_mea = (Experiment & f'id={experiment_id}').fetch1('is_mea')
        if not is_mea:
            return None
        data_dir = (EpochBlock & f"id={id}").fetch1('data_dir')
        full_path = os.path.join(NAS_DATA_DIR, data_dir)
        algorithms = []
        for algo in os.listdir(full_path):
            algorithms.append({'label': algo,
                            'data_path': os.path.join(full_path, algo),
                            'vis_type': 'epoch_block-mea'})
        return {'algorithms': algorithms}
    return None

# direct data grabber for Mike's code. Table name expected to be 'Stimulus'.
# for patch data only, since Experiment.data_file is empty for MEA data.
# returns the list object stored in data['quantity'] in the original h5file, 
# assuming h5path stored is well-formed
def get_data_generic(table_name: str, id: int):
    if table_name == 'Stimulus' or table_name == 'Response':
        # use parent_id to get epoch -> use experiment_id to get experiment -> get data_file
        h5_file = (Experiment & (
                        Epoch & (
                            table_dict[table_name] & f'id={id}').fetch1('parent_id')
                        ).fetch1('experiment_id')
                    ).fetch1('data_file')
    else:
        h5_file = (Experiment & (
                        table_dict[table_name] & f'id={id}').fetch1('experiment_id')
                    ).fetch1('data_file')
    h5_path = (table_dict[table_name] & f'id={id}').fetch1('h5path')
    with h5py.File(h5_file, 'r') as f:
        return f[h5_path]['data']['quantity']

def get_trace_binary(h5_file: str, h5_path: str) -> bytes:
    with h5py.File(h5_file, 'r') as f:
        fig = Figure()
        ax = fig.subplots()
        if 'data' not in f[h5_path].keys():
            return None
        ax.plot(f[h5_path]['data']['quantity'])
        buf = BytesIO()
        fig.savefig(buf, format='png')
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data

def get_spikehist_binary(base_path: str) -> bytes:
    clusters = np.load(os.path.join(base_path, 'spike_clusters.npy')).flatten()
    times = np.load(os.path.join(base_path, 'spike_times.npy')).flatten()
    sample_rate = 20_000
    cluster_counts = np.divide(np.bincount(clusters),
                                ((np.max(times) - np.min(times)) / sample_rate))

    # generate log-spaced bins, from 1 to the maximum number of spikes in a cluster, in 50 steps, and make a histogram
    bins = np.logspace(0, np.log10(cluster_counts.max()), 30)
    hist, bin_edges = np.histogram(cluster_counts, bins=bins)

    # plot the histogram: with bars not a line, and well-labeled axes (not just the powers of ten)
    fig = Figure()
    ax = fig.subplots()
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black')
    ax.set_xscale('log')
    ticks = [(10 ** i) for i in range(int(np.log10(cluster_counts.max())) + 2)]
    ax.set_xticks(ticks, [str(i) for i in ticks])
    ax.set_xlabel('Avg. spikes per second in cluster')
    ax.set_ylabel('Number of clusters')

    # send the plot to the browser
    buf = BytesIO()
    fig.savefig(buf, format='png')
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data

def add_tags(ids: list, tag: str):
    rows = []
    for id in ids:
        experiment_id, table_name, table_id = id.split('-')
        rows.append({'h5_uuid': (table_dict[table_name] & f"id={table_id}").fetch1()['h5_uuid'],
                     'experiment_id': experiment_id,
                     'table_name': table_name,
                     'table_id': table_id,
                     'user': user,
                     'tag': tag})
    print("Tags table:")
    Tags.insert(rows)
    print(Tags.fetch(), flush=True)

def delete_tags(ids: list, tag: str):
    for id in ids:
        experiment_id, table_name, table_id = id.split('-')
        (Tags & f"experiment_id='{experiment_id}'" & f"table_name='{table_name}'" & f"table_id={table_id}" 
         & f"tag='{tag}'").delete(safemode=False)
    print("Tags table:")
    print(Tags.fetch(), flush=True)

# recursive helper method to traverse the database and add tags to dict for user
def build_tags_dict(cur_id: int, cur_level: int, user: str, old_dict: dict) -> dict:
    cur_dict = {}
    tags = []
    # get old tags for other users
    if old_dict and 'tags' in old_dict.keys():
        for index in range(len(old_dict['tags'])):
            u, t = old_dict['tags'][index]
            if u != user:
                tags.append((u, t))
    # get tags for user
    table_name = table_arr[cur_level]
    if len(Tags & f"table_name='{table_name}'" & f"table_id={cur_id}" & f"user='{user}'") > 0: # tags for user
        for tag in (Tags & f"table_name='{table_name}'" & f"table_id={cur_id}" & f"user='{user}'").fetch('tag'):
            tags.append((user, tag))
    if len(tags) > 0:
        cur_dict['tags'] = tags
    # get children of current object
    if table_name != 'epoch':
        children = (table_dict[table_arr[cur_level+1]] & f"parent_id={cur_id}").fetch()
        for child in children:
            cur_dict[child['h5_uuid']] = build_tags_dict(child['id'], cur_level+1, user,
                                                         old_dict[child['h5_uuid']] if old_dict else None)
    return cur_dict

# add all tags made by the user to the tags_file, overwriting their previous tags
def push_tags(experiment_ids: list):
    for experiment_id in experiment_ids:
        print(f"tags for experiment {experiment_id}", flush=True)
        tags_file = (Experiment & f"id={experiment_id}").fetch1('tags_file')
        h5_uuid = (Experiment & f"id={experiment_id}").fetch1('h5_uuid')
        print(f"tags file: {tags_file}, h5: {h5_uuid}", flush=True)
        with open(tags_file, 'r') as f:
            old_dict = json.load(f)
        cur_dict = {}
        cur_dict[h5_uuid] = build_tags_dict(experiment_id, 0, user, old_dict[h5_uuid] if old_dict else None)
        with open(tags_file, 'w') as f:
            json.dump(cur_dict, f)

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

def traverse_and_append_tags(experiment_id: int, parent_id: int, cur_level: int, user_skip: str, tags_dict: dict):
    table_name = table_arr[cur_level]
    if table_name == 'response' or table_name == 'stimulus':
        return
    if table_name == 'experiment':
        ids = (table_dict[table_name] & f"id={experiment_id}").fetch('id')
    else:
        ids = (table_dict[table_name] & f"parent_id={parent_id}").fetch('id')
    for id in ids:
        h5_uuid = (table_dict[table_name] & f"id={id}").fetch1('h5_uuid')
        traverse_and_append_tags(experiment_id, id, cur_level + 1, user_skip,
                                 append_tags(h5_uuid, experiment_id, table_name, 
                                             id, user_skip, tags_dict))

# refresh tags from tags_file made by other users, delete old tags from other users
def pull_tags(experiment_ids: list):
    for experiment_id in experiment_ids:
        (Tags & f"experiment_id='{experiment_id}'" & f"user!='{user}'").delete(safemode=False)
        tags_file = (Experiment & f"id={experiment_id}").fetch1('tags_file')
        with open(tags_file, 'r') as f:
            tags = json.load(f)
        if tags == {}:
            continue
        traverse_and_append_tags(experiment_id, experiment_id, 0, user, tags)

# refresh tags from tags_file made by all users, delete old tags from all users
def reset_tags(experiment_ids: list):
    for experiment_id in experiment_ids:
        (Tags & f"experiment_id='{experiment_id}'").delete(safemode=False)
        tags_file = (Experiment & f"id={experiment_id}").fetch1('tags_file')
        with open(tags_file, 'r') as f:
            tags = json.load(f)
        if tags == {}:
            continue
        traverse_and_append_tags(experiment_id, experiment_id, 0, None, tags)