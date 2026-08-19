from retinanalysis.utils import database_pop
from retinanalysis._config import config
from retinanalysis._database import get_schema_module, schema
from typing import List


def populate_database(
    username: str | None = None,
    h5_dir: str | None = None,
    meta_dir: str | None = None,
    tags_dir: str | None = None,
):

    if username is None:
        username=config.USER

    if h5_dir is None:
        h5_dir = config.H5_DIR

    if meta_dir is None:
        meta_dir = config.META_DIR

    if tags_dir is None:
        tags_dir = config.TAGS_DIR

    schema_module = get_schema_module()

    return database_pop.append_data(h5_dir, meta_dir, tags_dir, username, schema_module)


def reload_experiment_data(
    exp_name: str,
    username: str | None = None,
    h5_dir: str | None = None,
    meta_dir: str | None = None,
    tags_dir: str | None = None,
):

    if username is None:
        username=config.USER

    if h5_dir is None:
        h5_dir = config.H5_DIR

    if meta_dir is None:
        meta_dir = config.META_DIR

    if tags_dir is None:
        tags_dir = config.TAGS_DIR

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
