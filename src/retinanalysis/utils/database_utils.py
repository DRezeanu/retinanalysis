from retinanalysis.utils import USER, H5_DIR, META_DIR, TAGS_DIR, database_pop
from retinanalysis._database import get_schema_module, schema
from typing import List


def populate_database(
    username=USER, h5_dir=H5_DIR, meta_dir=META_DIR, tags_dir=TAGS_DIR
):

    schema_module = get_schema_module()

    return database_pop.append_data(h5_dir, meta_dir, tags_dir, username, schema_module)


def reload_experiment_data(
    exp_name, username=USER, h5_dir=H5_DIR, meta_dir=META_DIR, tags_dir=TAGS_DIR
):

    (schema.Experiment() & {"exp_name": exp_name}).delete(prompt=False)

    populate_database(username, h5_dir, meta_dir, tags_dir)


def delete_experiments(exp_names: List[str]):

    for exp in exp_names:
        (schema.Experiment() & {"exp_name": exp}).delete(prompt=False)


def purge_database():
    all_experiments = schema.Experiment()
    all_exp_names = all_experiments.to_arrays("exp_name")

    for exp in all_exp_names:
        (schema.Experiment() & {"exp_name": exp}).delete(prompt=False)
