"""Smoke tests for DataJoint-backed retinanalysis workflows.

These tests are intentionally environment-dependent and are excluded from the
normal non-database test target. They validate that key DataJoint query helpers
still work against the local migration test database.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.db

EXP_NAME = "20260401C"
DATAFILE_NAME = "data009"
BLOCK_ID = 4764
ANALYSIS_CHUNK_NAME = "chunk2"
SS_VERSION = "kilosort2.5"
B_LED = False
EXPECTED_N_EPOCHS = 200
PROTOCOL_SEARCH = "presentmatfiles"
EXPECTED_PROTOCOL_NAME = "edu.washington.riekelab.protocols.PresentMatFiles"
EXPECTED_PROTOCOL_DATASET_SHAPE = (160, 13)

SORTED_DATA_DIR = Path(
    f"/Volumes/MEA_SSD/mea_ssd/data/sorted/{EXP_NAME}/{DATAFILE_NAME}/{SS_VERSION}"
)
ANALYSIS_CHUNK_DIR = Path(
    f"/Volumes/MEA_SSD/mea_ssd/analysis/{EXP_NAME}/{ANALYSIS_CHUNK_NAME}/{SS_VERSION}"
)


def test_protocol_search_and_dataset_smoke(test_database_container: str) -> None:
    """Exercise protocol search helpers against the test DB."""
    import retinanalysis as ra

    protocol_matches = ra.search_protocol(PROTOCOL_SEARCH, verbose=False)
    assert protocol_matches.tolist() == [EXPECTED_PROTOCOL_NAME]

    df_datasets = ra.get_datasets_from_protocol_names(PROTOCOL_SEARCH, verbose=False)
    expected_columns = [
        "exp_name",
        "datafile_name",
        "NDF",
        "chunk_name",
        "protocol_name",
        "is_mea",
        "data_dir",
        "group_label",
        "experiment_id",
        "protocol_id",
        "group_id",
        "block_id",
        "chunk_id",
    ]
    assert df_datasets.shape == EXPECTED_PROTOCOL_DATASET_SHAPE
    assert df_datasets.columns.tolist() == expected_columns
    assert df_datasets["protocol_name"].unique().tolist() == [EXPECTED_PROTOCOL_NAME]
    assert {"20250514C", "20260303C"}.issubset(set(df_datasets["exp_name"]))


def test_datajoint_query_smoke(test_database_container: str) -> None:
    """Exercise lightweight DataJoint query helpers against the test DB."""
    import retinanalysis as ra

    df_summary = ra.get_exp_summary(EXP_NAME)
    datafile_rows = df_summary.query("datafile_name == @DATAFILE_NAME")

    assert df_summary.shape == (24, 18)
    assert len(datafile_rows) == 1

    block_id = ra.get_block_id_from_datafile(EXP_NAME, DATAFILE_NAME)
    assert block_id == BLOCK_ID

    df_epochs = ra.get_epoch_data_from_exp(EXP_NAME, block_id, b_LED=B_LED)
    required_epoch_columns = {
        "experiment_id",
        "datafile_name",
        "block_id",
        "epoch_id",
        "frame_times_ms",
    }
    assert required_epoch_columns.issubset(df_epochs.columns)
    assert len(df_epochs) == EXPECTED_N_EPOCHS

    timing = ra.get_epochblock_timing(EXP_NAME, block_id, b_LED=B_LED)
    assert timing["n_epochs"] == EXPECTED_N_EPOCHS
    assert timing["stage_frame_rate"] == 59.0


@pytest.mark.integration
def test_create_mea_pipeline_smoke(test_database_container: str) -> None:
    """Exercise a known MEA pipeline path when local data files are mounted."""
    if not SORTED_DATA_DIR.exists():
        pytest.skip(f"Sorted data directory is not available: {SORTED_DATA_DIR}")
    if not ANALYSIS_CHUNK_DIR.exists():
        pytest.skip(f"Analysis chunk directory is not available: {ANALYSIS_CHUNK_DIR}")

    import retinanalysis as ra

    pipeline = ra.create_mea_pipeline(
        EXP_NAME,
        DATAFILE_NAME,
        analysis_chunk_name=ANALYSIS_CHUNK_NAME,
        ss_version=SS_VERSION,
        b_load_fd=False,
        b_LED=B_LED,
        verbose=False,
    )

    assert type(pipeline).__name__ == "MEAPipeline"
    assert type(pipeline.stim).__name__ == "MEAStimBlock"
    assert type(pipeline.resp).__name__ == "MEAResponseBlock"
    assert type(pipeline.analysis_chunk).__name__ == "AnalysisChunk"
