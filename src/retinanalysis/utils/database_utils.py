import datajoint as dj
from retinanalysis.utils import (USER,
                                 H5_DIR,
                                 META_DIR,
                                 TAGS_DIR,
                                 database_pop)
import retinanalysis.config.schema as schema
from typing import List

    
def populate_database(username = USER, h5_dir = H5_DIR, 
                        meta_dir = META_DIR, tags_dir = TAGS_DIR):
    
    db = dj.VirtualModule('schema.py', 'schema', create_schema=True)

    database_pop.append_data(h5_dir, meta_dir, tags_dir, username, db)

def reload_experiment_data(exp_name, username = USER, h5_dir = H5_DIR, 
                    meta_dir = META_DIR, tags_dir = TAGS_DIR):
    
    (schema.Experiment() & {'exp_name' : exp_name}).delete(safemode=False)

    populate_database(username, h5_dir, meta_dir, tags_dir)

def delete_experiments(exp_names: List[str]):

    for exp in exp_names:
        (schema.Experiment() & {'exp_name' : exp}).delete(safemode=False)

def purge_database():
    all_experiments = schema.Experiment()
    all_exp_names = all_experiments.fetch('exp_name')

    for exp in all_exp_names:
        (schema.Experiment() & {'exp_name' : exp}).delete(safemode=False)

