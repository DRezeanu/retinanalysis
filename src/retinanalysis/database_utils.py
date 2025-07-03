import datajoint as dj
from retinanalysis.settings import mea_config
import retinanalysis.schema as schema
import retinanalysis.database_pop as database_pop
    

NAS_DATA_DIR = mea_config['data']
NAS_ANALYSIS_DIR = mea_config['analysis']
META_DIR = mea_config['meta']
TAGS_DIR = mea_config['tags']
H5_DIR = mea_config['h5']
USER = mea_config['user']

def populate_database(username = USER, h5_dir = H5_DIR, 
                        meta_dir = META_DIR, tags_dir = TAGS_DIR):
    
    db = dj.VirtualModule('schema.py', 'schema')

    database_pop.append_data(h5_dir, meta_dir, tags_dir, username, db)

def reload_experiment_data(exp_name, username = USER, h5_dir = H5_DIR, 
                    meta_dir = META_DIR, tags_dir = TAGS_DIR):
    
    (schema.Experiment() & {'exp_name' : exp_name}).delete(safemode=False)

    populate_database(username, h5_dir, meta_dir, tags_dir)