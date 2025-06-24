import os
import numpy as np
import pandas as pd
import sys
import utilities.vr_spikeoutputs as so
import utilities.vr_spikeplots as sp
import schema
from utilities.sk_utils import NAS_ANALYSIS_DIR


def add_param_column(df: pd.DataFrame, param: str, col: str='epoch_parameters'):
    # Add a column to a dataframe that contains the value of a parameter in a dictionary
    # stored in a column of the dataframe.
    df[param] = np.nan
    for idx in df.index:
        params = df.loc[idx, col]
        if param in params:
            df.at[idx, param] = df.loc[idx, col][param]
        else:
            print(f'Parameter {param} not found in {col} for index {idx}')
    return df


def mea_exp_summary(exp_name: str):
    exp_id = (schema.Experiment() & f'exp_name="{exp_name}"').fetch('id')[0]
    eg_q = schema.EpochGroup() & f'experiment_id={exp_id}'
    sc_q = schema.SortingChunk() & f'experiment_id={exp_id}'
    sc_q = sc_q.proj('chunk_name', 'experiment_id',chunk_id='id')
    eg_sc_q = eg_q.proj(group_label='label', group_id='id') * sc_q
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


    # For each block_id, get first epoch id and its NDF parameter
    df['NDF'] = np.nan
    for bid in df['block_id'].values:
        ep_q = schema.Epoch() & f'parent_id={bid}'
        ep_id = ep_q.fetch('id')[0]
        params = (schema.Epoch() & f'id={ep_id}').fetch('parameters')[0]
        if 'NDF' in params.keys():
            df.loc[df['block_id']==bid, 'NDF'] = params['NDF']
        else:
            print(f'NDF parameter not found for block_id {bid}')


    

    # Order columns
    ls_order = ['data_dir', 'group_label', 'NDF','chunk_name', 'protocol_name',
    'duration_minutes', 'minutes_since_start', 'start_time', 'end_time',
       'experiment_id', 'group_id', 'block_id', 'chunk_id', 'protocol_id']
    df = df[ls_order]

    return df

def get_epoch_data_from_exp(exp_name: str, ls_params: list=None):
    # # Given experiment id, get epoch data with epoch group labels
    # Filter epochgroup by experiment_id, then join on EpochBlock, then join on Epoch
    exp_id = (schema.Experiment() & f'exp_name="{exp_name}"').fetch('id')[0]
    eg_q = schema.EpochGroup() & f'experiment_id={exp_id}'
    eg_q = eg_q.proj(cell_id='parent_id',group_label='label', group_id='id')
    c_q = eg_q * schema.Cell.proj(cell_id='id', cell_label='cell_label', cell_properties='properties')
    eb_q = c_q * schema.EpochBlock.proj('protocol_id', group_id='parent_id', block_id='id')
    p_q = eb_q * schema.Protocol.proj(protocol_name='name')
    e_q = p_q * schema.Epoch.proj(epoch_parameters='parameters', block_id='parent_id', epoch_id='id')
    r_q = e_q * schema.Response.proj(..., epoch_id='parent_id', response_id='id') 
    df = r_q.fetch(format='frame').reset_index()

    # Add column for cell type if exists in cell_properties
    df['cell_type'] = ''
    for idx in df.index:
        cell_props = df.loc[idx, 'cell_properties']
        if cell_props:
            if 'type' in cell_props:
                df.at[idx, 'cell_type'] = cell_props['type']

    # Add columns for any ls_params
    if ls_params:
        for param in ls_params:
            df[param] = np.nan
            for idx in df.index:
                params = df.loc[idx, 'epoch_parameters']
                if param in params:
                    df.at[idx, param] = df.loc[idx, 'epoch_parameters'][param]
                else:
                    print(f'Parameter {param} not found in epoch_parameters for epoch_id {df.loc[idx, "epoch_id"]}')

    # Move id columns to the end
    ls_id_cols = ['protocol_id', 'cell_id', 'group_id', 'block_id', 'epoch_id', 'response_id']
    ls_order = [col for col in df.columns if col not in ls_id_cols] + ls_id_cols
    df = df[ls_order]

    # Move cell type column to the front
    ls_order = ['cell_type'] + [col for col in df.columns if col != 'cell_type']
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
    return matches

def mea_meta_from_protocols(ls_protocol_names):
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

    # For each block_id, get first epoch id and its NDF parameter
    df['NDF'] = np.nan
    for bid in df['block_id'].values:
        ep_q = schema.Epoch() & f'parent_id={bid}'
        ep_id = ep_q.fetch('id')[0]
        params = (schema.Epoch() & f'id={ep_id}').fetch('parameters')[0]
        if 'NDF' in params.keys():
            df.loc[df['block_id']==bid, 'NDF'] = params['NDF']
        # else:
            # print(f'NDF parameter not found for {df.loc[df["block_id"]==bid]}')

    df['data_xxx'] = df['data_dir'].apply(lambda x: x.split('/')[-1])
    
    # order columns
    ls_order = ['data_dir', 'group_label', 'NDF', 'chunk_name', 
                'protocol_name', 'exp_name', 'data_xxx', 'is_mea',
                'experiment_id', 'protocol_id', 'group_id', 'block_id', 'chunk_id']
    df = df[ls_order]
    
    return df


def typing_summary_from_file(typing_file_id: int, ls_cell_types: list=None):
    if ls_cell_types is None:
        ls_cell_types = ['OffP', 'OffM', 'OnP', 'OnM', 'SBC']
    # Query SortedCellType
    sct_q = schema.SortedCellType() & f'file_id={typing_file_id}'
    df = sct_q.fetch(format='frame').reset_index()

    d_summary = {}
    d_summary['total_clusters'] = len(df)
    for str_type in ls_cell_types:
        # Count the number of clusters that contain the cell type.
        # Adding '/' as vision delimits with '/', to avoid matches like OnM with OnMystery
        d_summary[str_type] = df['cell_type'].str.contains(str_type+'/', case=False).sum()
    return d_summary


def cell_typing_from_chunks(ls_chunk_ids, ls_cell_types=None, b_remove_zeros=True):
    if ls_cell_types is None:
        ls_cell_types = ['OffP', 'OffM', 'OnP', 'OnM', 'SBC']
    # Query CellTypeFile
    ctf_q = schema.CellTypeFile() & [f'chunk_id={chunk_id}' for chunk_id in ls_chunk_ids]
    ctf_q = ctf_q.proj('chunk_id', 'algorithm', typing_file_name='file_name',
                       typing_file_id='id')
    # Get associated chunk_name, exp_name
    sc_q = schema.SortingChunk.proj('chunk_name', 'experiment_id', chunk_id='id')
    ctf_q = ctf_q * sc_q
    ex_q = schema.Experiment.proj('exp_name', experiment_id='id')
    ctf_q = ctf_q * ex_q
    df = ctf_q.fetch(format='frame').reset_index()

    # Add cell type summary columns
    df['total_clusters'] = 0
    for str_type in ls_cell_types:
        df[str_type] = 0

    # For each typing file, get summary of cell types
    ls_remove = []
    for idx in df.index:
        file_id = df.loc[idx, 'typing_file_id']
        summary = typing_summary_from_file(file_id, ls_cell_types)
        # Check if all cell types are zero
        b_all_zeros = all([summary[str_type]==0 for str_type in ls_cell_types])
        if b_all_zeros:
            str_print = df.loc[idx, 'exp_name'] + ', ' + df.loc[idx, 'chunk_name'] + ', ' + df.loc[idx, 'typing_file_name']
            print(f'No cell type matches found for {str_print}')
            ls_remove.append(idx)
        
        for key, val in summary.items():
            df.at[idx, key] = val

    if b_remove_zeros:
        df = df.drop(ls_remove)
        df = df.reset_index(drop=True)


    # Order columns
    ls_order = ['exp_name','chunk_name', 'algorithm', 
                'typing_file_name', 'total_clusters'] + \
                ls_cell_types + \
                ['experiment_id',  'chunk_id', 'typing_file_id']
    df = df[ls_order]
    df.attrs['ls_cell_types'] = ls_cell_types
    return df

def mea_data_from_meta(rows: pd.DataFrame, ls_cell_types=None):
    # Given a row from cell_typing_from_chunks
    # Return a SpikeOutputs object
    if ls_cell_types is None:
        ls_cell_types = ['OffP', 'OffM', 'OnP', 'OnM', 'SBC']
    row = rows.iloc[0]
    d_init = {'str_experiment': row['exp_name'],
        'dataset_name': row['algorithm'],
        'str_algo': row['algorithm'],
        'ls_RGC_labels': ls_cell_types}
    d_init['paramsfile'] = os.path.join(NAS_ANALYSIS_DIR, row['exp_name'],
                                        row['chunk_name'], row['algorithm'],
                                        f"{row['algorithm']}.params")
    d_init['str_classification'] = os.path.join(NAS_ANALYSIS_DIR, row['exp_name'],
                                        row['chunk_name'], row['algorithm'],
                                        row['typing_file_name'])
    d_init['ls_filenames'] = rows['data_xxx'].values
    data = so.SpikeOutputs(**d_init)
    return data


def mosaics_from_typing(df_ct: pd.DataFrame, df_meta: pd.DataFrame,
                        analysis_dir: str=None, ls_cell_types: list=None):
    # Input df_ct is output of cell_typing_from_chunks.
    # df_meta is output of mea_meta_from_protocols

    if analysis_dir is None:
        analysis_dir = NAS_ANALYSIS_DIR
    
    if ls_cell_types is None:
        ls_cell_types = df_ct.attrs['ls_cell_types']
        print(ls_cell_types)
    
    df = pd.merge(df_meta, df_ct, on=np.intersect1d(df_meta.columns, df_ct.columns).tolist(), how='inner')
    for tfid in df['typing_file_id'].values:
        rows = df[df['typing_file_id']==tfid]
        data = mea_data_from_meta(rows)
        str_annot = f"{rows['exp_name'].values[0]}, {rows['chunk_name'].values[0]}, {rows['typing_file_name'].values[0]}"
        # Check that cell type IDs were found
        if data.types.no_matches:
            print(f'No cell type matches found for {str_annot}')
            continue

        data.load_sta_from_params()

        # Plot mosaics
        # rf_axs, tc_axs = sp.plot_type_rfs_and_tcs(data)
        rf_axs = sp.plot_type_rfs(data)
        # Add exp and chunk annotation
        
        rf_axs[0].text(0, 1.1, str_annot,
                       transform=rf_axs[0].transAxes, fontsize=12)
        # tc_axs[0].text(0, 1.1, str_annot,
        #                   transform=tc_axs[0].transAxes, fontsize=12)

    return df