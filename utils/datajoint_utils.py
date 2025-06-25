import schema
import numpy as np
import datajoint as dj

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

    df['datafile_name'] = df['data_dir'].apply(lambda x: x.split('/')[-1])
    
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