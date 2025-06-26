import schema
import numpy as np
import datajoint as dj
import os
import pandas as pd
from utils.settings import mea_config
NAS_ANALYSIS_DIR = mea_config['analysis']

def djconnect(host_address: str = '127.0.0.1', user: str = 'root', password: str = 'simple'):
    dj.config["database.host"] = f"{host_address}"
    dj.config["database.user"] = f"{user}"
    dj.config["database.password"] = f"{password}"
    dj.conn()


def populate_ndf_column(df):
    # For each block_id, get first epoch id and its NDF parameter
    df['NDF'] = np.nan
    for bid in df['block_id'].values:
        ep_q = schema.Epoch() & f'parent_id={bid}'
        if len(ep_q) == 0:
            print(f'No epochs found for block {bid}')
            continue
        ep_id = ep_q.fetch('id')[0]
        params = (schema.Epoch() & f'id={ep_id}').fetch('parameters')[0]
        if 'NDF' in params.keys():
            df.loc[df['block_id']==bid, 'NDF'] = params['NDF']
        # else:
            # print(f'NDF parameter not found for block_id {bid}')
    return df


def get_mea_exp_summary(exp_name: str):
    exp_ids = (schema.Experiment() & f'exp_name="{exp_name}"').fetch('id')
    if len(exp_ids) == 0:
        print(f'Experiment "{exp_name}" not found!')
        return None
    exp_id = exp_ids[0]
    eg_q = schema.EpochGroup() & f'experiment_id={exp_id}'
    eg_q = eg_q.proj(group_label='label', group_id='id')
    is_mea = (schema.Experiment() & f'id={exp_id}').fetch1('is_mea')
    
    if not is_mea:
        print(f'Experiment "{exp_name}" is not an MEA experiment!')
        print('Please use INSERT_METHOD for single cell experiments.')
        return None
    sc_q = schema.SortingChunk() & f'experiment_id={exp_id}'
    sc_q = sc_q.proj('chunk_name', 'experiment_id',chunk_id='id')
    eg_sc_q = eg_q * sc_q
    # eb_q = eg_q.proj(group_label='label',group_id='id') * schema.EpochBlock.proj(group_id='parent_id', data_dir='data_dir', chunk_id='chunk_id')
    eb_q = eg_sc_q * schema.EpochBlock.proj('chunk_id', 'protocol_id','data_dir', 
                                            'start_time', 'end_time',
                                            group_id='parent_id', block_id='id')  
    p_q = eb_q * schema.Protocol.proj(..., protocol_name='name')

    df = p_q.fetch(format='frame').reset_index()
    df = df.sort_values('start_time').reset_index()
    
    # Add column of minutes_since_start
    df['minutes_since_start'] = (df['end_time'] - df['start_time'].min()).dt.total_seconds() / 60
    df['minutes_since_start'] = df['minutes_since_start'].round(2)
    # Add delta_minutes which gives derivative along rows
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    df['duration_minutes'] = df['duration_minutes'].round(2)

    df = populate_ndf_column(df)
    df['exp_name'] = df['data_dir'].apply(lambda x: os.path.split(x)[-2])
    df['datafile_name'] = df['data_dir'].apply(lambda x: os.path.split(x)[-1])

    # Order columns
    ls_order = ['exp_name', 'datafile_name', 'group_label', 'NDF','chunk_name', 'protocol_name',
    'duration_minutes', 'minutes_since_start', 'start_time', 'end_time', 'data_dir', 
    'experiment_id', 'group_id', 'block_id', 'chunk_id', 'protocol_id']
    df = df[ls_order]

    return df


def search_protocol(str_search: str):
    str_search = str_search.lower()
    protocols = schema.Protocol().fetch('name')
    protocols = np.unique(protocols)
    matches = []
    for p in protocols:
        if str_search in p.lower():
            matches.append(p)
    matches = np.array(matches)
    print(f'Found {len(matches)} protocols matching "{str_search}":')
    print(matches)
    return matches

def get_datasets_from_protocol_names(ls_protocol_names):
    # Query protocol table
    p_q = schema.Protocol() & [f'name="{protocol}"' for protocol in ls_protocol_names]
    p_ids = p_q.fetch('protocol_id')
    p_q = p_q.proj('protocol_id', protocol_name='name')

    # Query EpochBlock with these protocol IDs, get associated experiment IDs
    # WARNING: do not use EpochGroup ever for protocol_id queries bc of the "no_group_protocol" situation.
    # Thank you to @DRezeanu for pointing this out.
    eb_q = schema.EpochBlock() & [f'protocol_id={p_id}' for p_id in p_ids]
    ex_ids = eb_q.fetch('experiment_id')
    ex_ids = np.unique(ex_ids)

    # Join Experiment, EpochGroup, Protocol
    ex_q = schema.Experiment() & [f'id={ex_id}' for ex_id in ex_ids]
    ex_q = ex_q.proj('exp_name', 'is_mea', experiment_id='id')
    eg_q = schema.EpochGroup() & [f'experiment_id={ex_id}' for ex_id in ex_ids]
    eg_q = eg_q.proj('experiment_id', group_label='label', group_id='id')
    eg_q = p_q * eg_q
    eg_q = eg_q * ex_q

    # Join with EpochBlock
    eb_q = eg_q * schema.EpochBlock.proj('chunk_id', 'data_dir', 'protocol_id',
                                         group_id='parent_id', block_id='id')
    
    # Join with SortingChunk and fetch
    sc_q = schema.SortingChunk.proj('chunk_name', chunk_id='id')
    eb_q = eb_q * sc_q
    df = eb_q.fetch(format='frame').reset_index()

    df = populate_ndf_column(df)

    df['datafile_name'] = df['data_dir'].apply(lambda x: os.path.split(x)[-1])

    # order columns
    ls_order = ['exp_name', 'datafile_name',
                'NDF', 'chunk_name',
                'protocol_name',  'is_mea', 'data_dir', 'group_label',
                'experiment_id', 'protocol_id', 'group_id', 'block_id', 'chunk_id']
    df = df[ls_order]

    n_experiments = df['exp_name'].nunique()
    n_blocks = len(df)
    print(f'Found {n_experiments} experiments, {n_blocks} epoch blocks.')

    return df


def get_noise_chunks_sorted_by_distance(df_exp, datafile_name, noise_protocol_name, verbose=False):
    prot_row = df_exp[df_exp['datafile_name']==datafile_name]
    prot_start_time = prot_row['start_time'].values[0]
    prot_end_time = prot_row['end_time'].values[0]

    noise_chunk_names = df_exp[df_exp['protocol_name']==noise_protocol_name]['chunk_name'].unique()
    noise_chunk_distances = []
    for chunk_name in noise_chunk_names:
        # if chunk_name == prot_chunk_name:
            # noise_chunk_distances.append((chunk_name, 0.0))
            # continue
        noise_rows = df_exp[(df_exp['chunk_name']==chunk_name) & (df_exp['protocol_name']==noise_protocol_name)]
        noise_start_time = noise_rows['start_time'].values[0]
        noise_end_time = noise_rows['end_time'].values[-1]
        if noise_start_time < prot_start_time:
            distance = prot_start_time - noise_end_time
        else:
            distance = noise_start_time - prot_end_time
        # Convert to minutes
        distance = distance / np.timedelta64(1, 'm')
        noise_chunk_distances.append(distance)
    noise_chunk_names = noise_chunk_names[np.argsort(noise_chunk_distances)]
    if verbose:
        print(f'Found {len(noise_chunk_names)} noise chunks for protocol "{noise_protocol_name}"')
        print('Noise chunks sorted by distance:')
        for chunk_name, distance in zip(noise_chunk_names, noise_chunk_distances):
            print(f'  {chunk_name}: {distance:.2f} minutes')
    
    return noise_chunk_names, noise_chunk_distances


def get_typing_files_for_protocol(exp_name, datafile_name, noise_protocol_name, verbose=False):
    # Return df with columns:
    # exp_name, datafile_names, chunk_name, protocol_name, ss_version, typing_file_name, typing_file_path
    # Each row corresponds to a typing file for the nearest noise chunk.

    df_exp = get_mea_exp_summary(exp_name)
    noise_chunk_names, noise_chunk_distances = get_noise_chunks_sorted_by_distance(df_exp, datafile_name, noise_protocol_name, verbose)
    ls_no_cell_typing = []
    d_typing_files = {'datafile_name': [], 'datafile_names': [], 'ss_version': [], 
                      'typing_file_name': [], 'typing_file_path': [], 'typing_file_id': []}
    b_found = False
    selected_noise_chunk = None
    ideal_noise_chunk = noise_chunk_names[0]
    for noise_chunk in noise_chunk_names:
        noise_chunk = noise_chunk_names[0]
        df_chunk = df_exp[(df_exp['chunk_name']==noise_chunk) & (df_exp['protocol_name']==noise_protocol_name)]
        noise_chunk_id = df_chunk['chunk_id'].values[0]
        noise_datafile_names = list(df_chunk['datafile_name'].values)
        df_ct = (schema.CellTypeFile() & {'chunk_id': noise_chunk_id}).fetch(format='frame')
        n_typing_files = df_ct.shape[0]
        print(f'Found {n_typing_files} cell typing file(s) for {noise_chunk}')
        if n_typing_files == 0:
            ls_no_cell_typing.append(noise_chunk)

        if not b_found:
            for idx in df_ct.index:
                ss_version = df_ct.at[idx, 'algorithm']
                typing_file_name = df_ct.at[idx, 'file_name']
                typing_file_path = os.path.join(NAS_ANALYSIS_DIR, exp_name, noise_chunk, ss_version, typing_file_name)
                
                d_typing_files['datafile_name'].append(noise_datafile_names[0])
                d_typing_files['datafile_names'].append(noise_datafile_names)
                d_typing_files['ss_version'].append(ss_version)
                d_typing_files['typing_file_name'].append(typing_file_name)
                d_typing_files['typing_file_path'].append(typing_file_path)
                d_typing_files['typing_file_id'].append(idx)
            selected_noise_chunk = noise_chunk
            b_found = True

    # If no cell typing files found, return noise chunk names.
    if not b_found:
        print(f'No cell typing files found for any noise chunk for protocol "{noise_protocol_name}" in experiment "{exp_name}"')
        print(f'You should type the closest noise chunk:')
        for chunk_name, distance in zip(noise_chunk_names, noise_chunk_distances):
            print(f'  {chunk_name}: {distance:.2f} minutes')
        return noise_chunk_names
    
    # If selected noise chunk is not ideal, print warning.
    if selected_noise_chunk != ideal_noise_chunk:
        print(f'WARNING: Selected noise chunk "{selected_noise_chunk}" is not the ideal noise chunk "{ideal_noise_chunk}".')
        print('Consider typing the ideal noise chunk:')
        for chunk_name, distance in zip(noise_chunk_names, noise_chunk_distances):
            print(f'  {chunk_name}: {distance:.2f} minutes')

    df_typing_files = pd.DataFrame(d_typing_files)
    df_typing_files['exp_name'] = exp_name
    df_typing_files['chunk_name'] = selected_noise_chunk
    df_typing_files['protocol_name'] = noise_protocol_name
    ls_order = ['exp_name', 'datafile_name', 'datafile_names', 'chunk_name', 'protocol_name',
                'ss_version', 'typing_file_name', 'typing_file_path', 'typing_file_id']
    df_typing_files = df_typing_files[ls_order]
    return df_typing_files

def find_varying_epoch_parameters(df):
    # df should have epoch data for a SINGLE EpochBlock.
    # Thus assuming that all epoch parameter dictionaries have the same keys.
    # The epoch_parameters column has dictionaries of {parameter_name: value}
    # We want to find which parameters vary across epochs in the dataframe
    varying_params = set()
    idx = df.index[0]
    param_names = df.at[idx, 'epoch_parameters'].keys()
    for key in param_names:
        # Get all values across all epochs
        values = df['epoch_parameters'].apply(lambda x: x.get(key)).values
        # Check if all None
        if all(value is None for value in values):
            continue
        if len(np.unique(values)) > 1:
            varying_params.add(key)
    return list(varying_params)

def add_parameters_col(df, ls_params, src_col: str='epoch_parameters'):
    # Add a column to the dataframe that contains the values of the specified parameters
    # src_col should have dictionaries of {parameter_name: value}
    for param in ls_params:
        df[param] = df[src_col].apply(lambda x: x.get(param))
    return df


def get_mea_epoch_data_from_exp(exp_name: str, datafile_name: str, ls_params: list=None):
    # Filter Experiment by exp_name, EpochBlock by datafile_name, then join down to Epoch
    ex_q = schema.Experiment() & f'exp_name="{exp_name}"'
    eg_q = schema.EpochGroup() * ex_q.proj('exp_name', experiment_id='id')
    eg_q = eg_q.proj('exp_name', group_label='label', group_id='id')
    eb_q = eg_q * schema.EpochBlock.proj('protocol_id', 'data_dir', group_id='parent_id', block_id='id')
    data_dir = os.path.join(exp_name, datafile_name)
    eb_q = eb_q & f'data_dir="{data_dir}"'
    p_q = eb_q * schema.Protocol.proj(protocol_name='name')
    e_q = p_q * schema.Epoch.proj(epoch_parameters='parameters', block_id='parent_id', epoch_id='id')
    df = e_q.fetch(format='frame')
    df = df.reset_index()

    df['datafile_name'] = df['data_dir'].apply(lambda x: os.path.split(x)[-1])

    varying_params = find_varying_epoch_parameters(df)
    # Add ls_params to varying_params if provided
    if ls_params is not None:
        varying_params = list(set(varying_params).union(set(ls_params)))
    df = add_parameters_col(df, varying_params, 'epoch_parameters')
    ls_order =  varying_params + \
        ['exp_name', 'datafile_name', 'group_label', 'protocol_name',
        'epoch_parameters', 'data_dir', 'experiment_id', 'group_id', 'block_id', 'protocol_id', 'epoch_id']
    df = df[ls_order]
    
    # Name index 'epoch_index'
    df.index = df.index.rename('epoch_index')

    return df

def get_mea_epochblock_timing(exp_name: str, datafile_name: str):
    ex_q = schema.Experiment() & f'exp_name="{exp_name}"'
    eg_q = schema.EpochGroup() * ex_q.proj('exp_name', experiment_id='id')
    eg_q = eg_q.proj('exp_name', group_label='label', group_id='id')
    eb_q = schema.EpochBlock.proj(
        'protocol_id', 'data_dir', group_properties='properties',
        group_id='parent_id', block_id='id'
        )
    eb_q = eg_q * eb_q
    data_dir = os.path.join(exp_name, datafile_name)
    eb_q = eb_q & f'data_dir="{data_dir}"'
    df = eb_q.fetch(format='frame').reset_index()
    if len(df) > 1:
        raise ValueError(f'Expected only one EpochBlock for {exp_name} {datafile_name}, but found {len(df)}')
    if len(df) == 0:
        raise ValueError(f'No EpochBlock found for {exp_name} {datafile_name}')
    d_data = df.loc[0].to_dict()
    epoch_starts = d_data['group_properties']['epochStarts']
    epoch_ends = d_data['group_properties']['epochEnds']
    n_samples = d_data['group_properties']['n_samples']
    frame_times_ms = d_data['group_properties']['frameTimesMs']

    d_timing = {
        'exp_name': exp_name,
        'datafile_name': datafile_name,
        'epoch_starts': epoch_starts,
        'epoch_ends': epoch_ends,
        'n_samples': n_samples,
        'frame_times_ms': frame_times_ms
    }

    return d_timing