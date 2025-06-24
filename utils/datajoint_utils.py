import schema
import numpy as np
import datajoint as dj

def djconnect(host_address: str = '127.0.0.1', user: str = 'root', password: str = 'simple'):
    dj.config["database.host"] = f"{host_address}"
    dj.config["database.user"] = f"{user}"
    dj.config["database.password"] = f"{password}"
    dj.conn()

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