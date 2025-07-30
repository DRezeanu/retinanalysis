from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retinanalysis.classes.analysis_chunk import AnalysisChunk

from retinanalysis.utils import (NAS_ANALYSIS_DIR,
                                 H5_DIR,
                                 schema)

import numpy as np
import datajoint as dj
import os
import pandas as pd
import json
from tqdm.auto import tqdm
from IPython.display import display
import h5py


def djconnect(host_address: str = '127.0.0.1', user: str = 'root', password: str = 'simple'):
    """
    Connect to local datajoint database container active inside docker.
    Note: The docker database container MUST be running.
    
    Parameters:
    host_address (str): IP address of mysql/datajoint server. Default '127.0.0.1'
    user (str): username to log onto server. Default 'root'
    password (str): password to log onto server. Default 'simple'

    Default parameters should not be changed unless you have created a custom
    config of the mysql/datajoint docker image and database container.
    """

    try:
        dj.config["database.host"] = f"{host_address}"
        dj.config["database.user"] = f"{user}"
        dj.config["database.password"] = f"{password}"
        dj.conn()
    except Exception as e:
        print(f"Could not connect to DataJoint database: {e}")


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


def get_exp_summary(exp_name: str):
    exp_ids = (schema.Experiment() & f'exp_name="{exp_name}"').fetch('id')
    exp_id = exp_ids[0]
    if len(exp_ids) == 0:
        print(f'Experiment "{exp_name}" not found!')
        return None
    is_mea = (schema.Experiment() & f'id={exp_id}').fetch1('is_mea')

    eg_q = schema.EpochGroup() & f'experiment_id={exp_id}'
    eg_q = eg_q.proj('experiment_id', group_label='label', group_id='id', cell_id='parent_id')
    c_q = schema.Cell.proj(
        prep_id='parent_id', cell_id='id',
        cell_label='label', cell_properties='properties'
        )
    pr_q = schema.Preparation.proj(prep_label='label', prep_id='id')
    eg_q = eg_q * c_q * pr_q
    eb_q = schema.EpochBlock.proj(
        'chunk_id', 'protocol_id','data_dir', 
        'start_time', 'end_time',
        group_id='parent_id', block_id='id'
        )
    eb_q = eg_q * eb_q
    # If MEA experiment, get sorting chunk information
    if is_mea:
        sc_q = schema.SortingChunk() & f'experiment_id={exp_id}'
        sc_q = sc_q.proj('chunk_name', chunk_id='id')
        eb_q = eb_q * sc_q
    p_q = eb_q * schema.Protocol.proj(..., protocol_name='name')

    df = p_q.fetch(format='frame').reset_index()
    df = df.sort_values('start_time').reset_index()
    if len(df)==0:
        raise ValueError(f'No data found for experiment {exp_name}')
    
    # Check that end_time and start_time are in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
        print("Converting 'start_time' to datetime format.")
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df['end_time']):
        print("Converting 'end_time' to datetime format.")
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    # Add column of minutes_since_start
    df['minutes_since_start'] = (df['end_time'] - df['start_time'].min()).dt.total_seconds() / 60
    df['minutes_since_start'] = df['minutes_since_start'].round(2)
    # Add delta_minutes which gives derivative along rows
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    df['duration_minutes'] = df['duration_minutes'].round(2)

    df['exp_name'] = exp_name

    df = populate_ndf_column(df)

    
    if is_mea:
        df['datafile_name'] = df['data_dir'].apply(lambda x: os.path.split(x)[-1])

        # Order columns.
        ls_order = ['exp_name', 'prep_label', 'datafile_name', 'group_label', 'NDF','chunk_name', 'protocol_name',
        'duration_minutes', 'minutes_since_start', 'start_time', 'end_time', 'data_dir', 
        'experiment_id', 'prep_id', 'group_id', 'block_id', 'chunk_id', 'protocol_id']
    else:
        # Add cell type from cell_properties dict w/ key 'type'
        df['cell_type'] = df['cell_properties'].apply(lambda x: x.get('type', 'Unknown'))
        ls_order = ['exp_name', 'prep_label', 'cell_id', 'cell_label', 'cell_type', 
        'NDF', 'protocol_name', 'duration_minutes', 'minutes_since_start', 
        'start_time', 'end_time', 'cell_properties','group_label',
        'experiment_id', 'prep_id', 'group_id', 'block_id', 'protocol_id']
    df = df[ls_order]

    return df

def get_block_id_from_datafile(exp_name: str, datafile_name: str):
    exp_id = (schema.Experiment() & f'exp_name="{exp_name}"').fetch1('id')
    eb_q = schema.EpochBlock() & f'experiment_id={exp_id}'
    df = eb_q.fetch(format='frame').reset_index()
    df['datafile_name'] = df['data_dir'].apply(lambda x: os.path.split(x)[-1])
    block_id = df.query('datafile_name == @datafile_name')['id'].values[0]
    return block_id


def search_protocol(str_search: str):
    """Search for symphony protocols by name.
    
    Parmeters:
    str_search (str): a string contained somewhere inside the protocol name (e.g., PresentImages)
    
    Returns:
    matches (List[str]): a list of full protocol names as strings (e.g., manookinlab.protocols.PresentImages)"""
    
    str_search = str_search.lower()
    protocols = schema.Protocol().fetch('name')
    protocols = np.unique(protocols)
    matches = []
    for p in protocols:
        if str_search in p.lower():
            matches.append(p)
    matches = np.array(matches)
    print(f'\nFound {len(matches)} protocols matching "{str_search}":')
    print(matches)
    return matches

def get_datasets_from_protocol_names(ls_protocol_names):
    # TODO pull available algorithm names from CellTypeFile
    if type(ls_protocol_names) is str:
        ls_protocol_names = [ls_protocol_names]

    found_protocols = []
    for protocol in ls_protocol_names:
        found_protocols  += list(search_protocol(protocol))

    # Query protocol table
    p_q = schema.Protocol() & [f'name="{protocol}"' for protocol in found_protocols]
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
    print(f'\nFound {n_experiments} experiments, {n_blocks} epoch blocks.\n')

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
    noise_chunk_distances = np.sort(noise_chunk_distances)
    if verbose:
        print(f'Found {len(noise_chunk_names)} noise chunks for protocol "{noise_protocol_name}"')
        print('Noise chunks sorted by distance:')
        for chunk_name, distance in zip(noise_chunk_names, noise_chunk_distances):
            print(f'  {chunk_name}: {distance:.2f} minutes')
    
    return noise_chunk_names, noise_chunk_distances


def get_n_cells_of_interest(str_file, ls_cell_types: list = ['OffP', 'OffM', 'OnP', 'OnM']):
    # str_file is cell typing file path
    # Returns the number of cells of interest in the typing file
    if not os.path.exists(str_file):
        print(f'Cell typing file {str_file} does not exist!')
        return 0
    try:
        arr_types = np.genfromtxt(str_file, dtype=str, delimiter='  ')
        if len(arr_types.shape) != 2:
            try_shape = arr_types.shape
            # Try 1 space delimiter
            arr_types = np.genfromtxt(str_file, dtype=str, delimiter=' ')
            if len(arr_types.shape) != 2:
                print(f'Could not parse {str_file}')
                print(f'Error: 2 space delimiter resulted in {try_shape} shape')
                print(f'Error: 1 space delimiter resulted in {arr_types.shape} shape')
                return 0
    except:
        print(f'Error reading cell typing file {str_file}. Please check the file format.')
        return 0
    n_cells = 0
    # 2nd column of arr_types contains cell types, where we want to match substring after .lower()
    for str_match in ls_cell_types:
        ls_match = []
        for i in range(len(arr_types)):
            str_cell_type = arr_types[i, 1].lower()
            if str_match.lower() in str_cell_type:
                ls_match.append(arr_types[i, 0])
        n_cells += len(ls_match)
    return n_cells

def get_noise_name_by_exp(exp_name):
    # Pull appropriate noise protocol for cell typing
    if int(exp_name[:8]) < 20230926:
            noise_protocol_name = 'manookinlab.protocols.FastNoise'
    else:
        noise_protocol_name = 'manookinlab.protocols.SpatialNoise'
    return noise_protocol_name

def get_typing_files_for_datasets(df, ls_cell_types: list = ['OffP', 'OffM', 'OnP', 'OnM'],
                                  verbose: bool = False):
    # Return df_typed with columns:
    # exp_name, datafile_names, chunk_name, protocol_name, ss_version, typing_file_name, typing_file_path
    # Each row corresponds to a typing file for the nearest noise chunk.
    # And df_not_typed with columns:
    # exp_name, datafile_names, nearest_noise_chunk, nearest_noise_distance
    # Each row corresponds to a dataset without any typing files.
    d_not_typed = {'exp_name': [], 'datafile_name': [], 'nearest_noise_chunk': [],
                   'nearest_noise_distance': []}
    d_typed = {'exp_name': [], 'datafile_name': [], 'nearest_noise_chunk': [],
                'nearest_noise_distance': [], 'typed_noise_chunk': [], 
                'nearest_noise_distance': [], 'typed_noise_distance': [], 'is_nearest': [],
                'ss_version': [], 'noise_datafile_names': [],
                'typing_file_name': [], 'typing_file_path': [], 'typing_file_id': [],
                'n_cells_of_interest': []}
    for exp_name in tqdm(df['exp_name'].unique(), desc="Finding typing files for unique experiments"):
        df_q = df.query('exp_name==@exp_name')
        df_exp = get_exp_summary(exp_name)
        noise_protocol_name = get_noise_name_by_exp(exp_name)
        
        for datafile_name in df_q['datafile_name'].values:
            noise_chunk_names, noise_chunk_distances = get_noise_chunks_sorted_by_distance(df_exp, datafile_name, noise_protocol_name, verbose)
    
            # Find nearest noise chunk with typing.
            b_found = False
            nearest_noise_chunk = noise_chunk_names[0]
            for i_c, noise_chunk in enumerate(noise_chunk_names):
                df_chunk = df_exp[(df_exp['chunk_name']==noise_chunk) & (df_exp['protocol_name']==noise_protocol_name)]
                noise_chunk_id = df_chunk['chunk_id'].values[0]
                noise_datafile_names = list(df_chunk['datafile_name'].values)
                df_ct = (schema.CellTypeFile() & {'chunk_id': noise_chunk_id}).fetch(format='frame')

                if not b_found and len(df_ct) > 0:
                    for i_ct in df_ct.index:
                        ss_version = df_ct.at[i_ct, 'algorithm']
                        typing_file_name = df_ct.at[i_ct, 'file_name']
                        typing_file_path = os.path.join(NAS_ANALYSIS_DIR, exp_name, noise_chunk, ss_version, typing_file_name)
                        n_cells_of_interest = get_n_cells_of_interest(typing_file_path, ls_cell_types)
                        d_typed['exp_name'].append(exp_name)
                        d_typed['datafile_name'].append(datafile_name)
                        d_typed['nearest_noise_chunk'].append(nearest_noise_chunk)
                        d_typed['nearest_noise_distance'].append(noise_chunk_distances[0])
                        d_typed['typed_noise_chunk'].append(noise_chunk)
                        d_typed['typed_noise_distance'].append(noise_chunk_distances[i_c])
                        d_typed['noise_datafile_names'].append(noise_datafile_names)
                        d_typed['ss_version'].append(ss_version)
                        d_typed['typing_file_name'].append(typing_file_name)
                        d_typed['typing_file_path'].append(typing_file_path)
                        d_typed['typing_file_id'].append(i_ct)
                        d_typed['n_cells_of_interest'].append(n_cells_of_interest)
                        if noise_chunk == nearest_noise_chunk:
                            d_typed['is_nearest'].append(True)
                        else:
                            d_typed['is_nearest'].append(False)
                    b_found = True

            # If no cell typing files found, append to d_not_typed
            if not b_found:
                d_not_typed['exp_name'].append(exp_name)
                d_not_typed['datafile_name'].append(datafile_name)
                d_not_typed['nearest_noise_chunk'].append(nearest_noise_chunk)
                d_not_typed['nearest_noise_distance'].append(noise_chunk_distances[0])


    df_typed = pd.DataFrame(d_typed)
    df_not_typed = pd.DataFrame(d_not_typed)
    return df_typed, df_not_typed

def plot_mosaics_for_all_datasets(df: pd.DataFrame, ls_cell_types: list=['OffP', 'OffM', 'OnP', 'OnM'],
                                  n_top: int=None):
    # df should be output of get_datasets_from_protocol_names
    df_typed, df_not_typed = get_typing_files_for_datasets(df, ls_cell_types)
    print(f'Found {df_not_typed.shape[0]} datasets without any typing files:')
    display(df_not_typed)
    print(f'Found {df_typed.shape[0]} datasets with typing files.')

    df_u = df_typed[['exp_name', 'typed_noise_chunk', 'n_cells_of_interest']].drop_duplicates()
    # Keep only those with n_cells_of_interest > 0
    df_u = df_u.query('n_cells_of_interest > 0')
    # Sort by n_cells_of_interest
    df_u = df_u.sort_values('n_cells_of_interest', ascending=False)
    if n_top is None:
        n_top = 20
    print(f'Found {df_u.shape[0]} unique datasets with typing files and > 0  cells of interest.')
    for u_idx in df_u.index[:n_top]:
        exp_name = df_u.at[u_idx, 'exp_name']
        
        chunk_name = df_u.at[u_idx, 'typed_noise_chunk']
        df_q = df_typed.query(f'exp_name == "{exp_name}" and typed_noise_chunk == "{chunk_name}"')
        df_q = df_q.reset_index(drop=True)
        datafile_names = df_q['datafile_name'].unique()
        # for i, row in df_q.iterrows():
        # Find row with max n_cells_of_interest
        row = df_q.loc[df_q['n_cells_of_interest'].idxmax()]
        ss_version = row['ss_version']
        typing_file = row['typing_file_name']
        try:    
            ac1 = AnalysisChunk(exp_name, chunk_name, ss_version, b_load_spatial_maps=False,
                                include_ei=False, include_neurons=False, verbose=False)

            axs = ac1.plot_rfs(typing_file=typing_file,
                cell_types=ls_cell_types, units = 'microns', 
                std_scaling = 1.0, b_zoom=True, n_pad = 6
            )
            f = axs[0].get_figure()
            nearest_chunk = row['nearest_noise_chunk']
            str_title = f"{exp_name} {chunk_name} {ss_version} {typing_file}\nfor {datafile_names}"
            if nearest_chunk != chunk_name:
                str_title += f"\n(nearest chunk: {nearest_chunk})"
            f.suptitle(str_title, y=1.05)
            for ax in axs:
                ax.set_aspect('equal', adjustable='box')
        except Exception as e:
            print(f'Error processing {exp_name}, {datafile_names}, {chunk_name}: {e}')
            continue
    return df_typed, df_not_typed


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


def get_epoch_data_from_exp(exp_name: str, block_id: int, ls_params: list=None,
                                stim_time_name: str='stimTime'):
    # Filter Experiment by exp_name, EpochBlock by block_id, then join down to Epoch
    ex_q = schema.Experiment() & f'exp_name="{exp_name}"'
    is_mea = (ex_q.fetch1('is_mea') == 1)
    eg_q = schema.EpochGroup() * ex_q.proj('exp_name', experiment_id='id')
    eg_q = eg_q.proj('exp_name', group_label='label', group_id='id')
    
    ls_eb_cols = ['protocol_id']
    if is_mea:
        ls_eb_cols += ['data_dir']
    eb_q = schema.EpochBlock.proj(
        *ls_eb_cols, group_id='parent_id', block_id='id'
        )
        
    eb_q = eg_q * eb_q
    
    eb_q = eb_q & f'block_id={block_id}'
    
    p_q = eb_q * schema.Protocol.proj(protocol_name='name')
    
    e_q = p_q * schema.Epoch.proj(
        epoch_parameters='parameters', block_id='parent_id', epoch_id='id',
        frame_times_ms="properties->>'$.frameTimesMs'"
        )
    df = e_q.fetch(format='frame')
    df = df.reset_index()
    # Make frame_times_ms list using json.loads
    df['frame_times_ms'] = df['frame_times_ms'].apply(lambda x: json.loads(x))

    if is_mea:
        df['datafile_name'] = df['data_dir'].apply(lambda x: os.path.split(x)[-1])

    varying_params = find_varying_epoch_parameters(df)
    # Add ls_params to varying_params if provided
    if ls_params is not None:
        varying_params = list(set(varying_params).union(set(ls_params)))
    df = add_parameters_col(df, varying_params, 'epoch_parameters')

    # Add preTime, stimTime, tailTime
    ls_time_cols = ['preTime', stim_time_name, 'tailTime']
    df = add_parameters_col(df, ls_time_cols, 'epoch_parameters')
    if is_mea:
        ls_order =  varying_params + \
            ['exp_name', 'datafile_name', 'group_label', 'protocol_name', 'frame_times_ms',
            'epoch_parameters', 'data_dir'] + ls_time_cols + \
            ['experiment_id', 'group_id', 'block_id', 'protocol_id', 'epoch_id']
    else:
        ls_order =  varying_params + \
            ['exp_name', 'group_label', 'protocol_name', 'frame_times_ms',
            'epoch_parameters'] + ls_time_cols + \
            ['experiment_id', 'group_id', 'block_id', 'protocol_id', 'epoch_id']
    df = df[ls_order]
    
    # Add column for 'epoch_index'
    df.index = df.index.rename('epoch_index')
    df = df.reset_index(drop=False)

    return df


def get_epochblock_query(exp_name: str, block_id: int):
    ex_q = schema.Experiment() & f'exp_name="{exp_name}"'
    eg_q = schema.EpochGroup() * ex_q.proj('exp_name', experiment_id='id')
    eg_q = eg_q.proj('exp_name', group_label='label', group_id='id')
    eb_q = schema.EpochBlock.proj(
        'protocol_id', 'data_dir', group_properties='properties',
        group_id='parent_id', block_id='id'
        )
    eb_q = eg_q * eb_q
    eb_q = eb_q & f'block_id={block_id}'
    return eb_q


def get_epochblock_timing(exp_name: str, block_id: int):
    eb_q = get_epochblock_query(exp_name, block_id)
    df = eb_q.fetch(format='frame').reset_index()
    if len(df) > 1:
        raise ValueError(f'Expected only one EpochBlock for {exp_name} {block_id}, but found {len(df)}')
    if len(df) == 0:
        raise ValueError(f'No EpochBlock found for {exp_name} {block_id}')
    d_data = df.loc[0].to_dict()
    # epoch_starts = d_data['group_properties']['epochStarts']
    # epoch_ends = d_data['group_properties']['epochEnds']
    # n_samples = d_data['group_properties']['n_samples']
    # frame_times_ms = d_data['group_properties']['frameTimesMs']

    d_timing = {
        'exp_name': exp_name,
        'block_id': block_id,
        # 'epoch_starts': epoch_starts,
        # 'epoch_ends': epoch_ends,
        # 'n_samples': n_samples,
        # 'frame_times_ms': frame_times_ms
    }
    
    # For MEA data, this has epoch_starts, epoch_ends, n_samples, frame_times_ms
    # For SC data, this has just frame_times_ms
    d_group = d_data['group_properties']
    for key in d_group.keys():
        d_timing[key] = d_group[key]
    

    return d_timing

def get_epochblock_response_query(exp_name: str, block_id: int):
    eb_q = get_epochblock_query(exp_name, block_id)
    p_q = eb_q * schema.Protocol.proj(protocol_name='name')
    e_q = p_q * schema.Epoch.proj(epoch_parameters='parameters', block_id='parent_id', epoch_id='id')
    r_q = e_q * schema.Response.proj(..., epoch_id='parent_id', response_id='id') 
    return r_q


def get_h5_file(exp_name):
    # First try h5 in config h5 dir
    str_h5_in_config = os.path.join(H5_DIR, f'{exp_name}.h5')
    if os.path.exists(str_h5_in_config):
        return str_h5_in_config

    # Otherwise try path from database
    ex_q = schema.Experiment() & f'exp_name="{exp_name}"'
    str_h5_from_db = ex_q.fetch1('data_file')
    if os.path.exists(str_h5_from_db):
        return str_h5_from_db
    else:
        raise FileNotFoundError(f'No h5 file found for experiment {exp_name}.\n'+\
                                f'Tried {str_h5_in_config} and {str_h5_from_db}')



def get_epochblock_frame_data(exp_name: str, block_id: int, str_h5: str=None):
    if str_h5 is None:
        str_h5 = get_h5_file(exp_name)
    print(f'Loading frame monitor data from {str_h5} ...')
    r_q = get_epochblock_response_query(exp_name, block_id)
    df = r_q.fetch(format='frame').reset_index()
    
    df_frame = df[df['device_name']=='Frame Monitor']
    df_frame = df_frame.reset_index(drop=True)

    frame_h5paths = df_frame['h5path'].values

    # Collect data
    frame_data = []
    with h5py.File(str_h5, 'r') as f:
        for h5path in frame_h5paths:
            trace = f[h5path]['data']['quantity']
            frame_data.append(trace)
    
    frame_data = np.array(frame_data)
    print(f'Loaded {frame_data.shape} frame_data.\n')

    sample_rates = df_frame['sample_rate'].unique().astype(float)
    if len(sample_rates) != 1:
        raise ValueError(f'Expected single sample rate for Frame Monitor data, but found {len(sample_rate)}: {sample_rate}')
    sample_rate = sample_rates[0]

    return frame_data, sample_rate

def get_epochblock_amp_data(exp_name: str, block_id: int, str_h5: str=None):
    if str_h5 is None:
        str_h5 = get_h5_file(exp_name)
    print(f'Loading Amp1 data from {str_h5} ...')
    r_q = get_epochblock_response_query(exp_name, block_id)
    df = r_q.fetch(format='frame').reset_index()
    
    df_amp = df[df['device_name']=='Amp1']
    df_amp = df_amp.reset_index(drop=True)

    amp_h5paths = df_amp['h5path'].values

    # Collect data
    amp_data = []
    with h5py.File(str_h5, 'r') as f:
        for h5path in amp_h5paths:
            trace = f[h5path]['data']['quantity']
            amp_data.append(trace)
    
    amp_data = np.array(amp_data)
    print(f'Loaded {amp_data.shape} amp_data.\n')

    sample_rates = df_amp['sample_rate'].unique().astype(float)
    if len(sample_rates) != 1:
        raise ValueError(f'Expected single sample rate for Amp1 data, but found {len(sample_rate)}: {sample_rate}')
    sample_rate = sample_rates[0]

    return amp_data, sample_rate