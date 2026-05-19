# AI Developer Rulebook, Architecture, & Philosophy

Welcome to the retinanalysis repository. You are an AI acting as a core developer on this project. Read this document in its entirety before modifying any code.

---

## 1. Project Overview & Purpose

retinanalysis is a Python analysis pipeline package for multi-electrode array (MEA) and single-cell electrophysiology data recorded from retinal ganglion cells (RGCs). It was created for the Rieke/Manookin lab at the University of Washington.

The package:
- Ingests experiment metadata from Symphony2 H5 files into a local DataJoint/MySQL database (run via Docker)
- Loads spike-sorted data from "vision files" produced by converting Kilosort 2.5 output through the `artificial-retina-software-pipeline` submodule
- Provides classes for organizing stimulus, response, and spatial noise analysis data
- Supports cluster matching (linking cells across different recording files via EI correlation)
- Offers plotting utilities for receptive fields, timecourses, STAs, PSTHs, and more
- Supports stimulus regeneration for several protocols

**Primary users** are neuroscience PhD students and postdocs working in Jupyter notebooks. The API is designed for interactive exploratory analysis, not batch processing or GUI use.

---

## 2. Repository Structure

```
retinanalysis/
├── .pi/                        # AI agent documentation (this directory)
├── docker-compose.yaml         # DataJoint MySQL 8.0 container definition
├── pyproject.toml              # Build config (hatchling), dependencies
├── lib/                        # Git submodules
│   └── artificial-retina-software-pipeline/  # visionloader, visionwriter, bin2py
├── src/retinanalysis/
│   ├── __init__.py             # Package-level imports (order matters!)
│   ├── config/
│   │   ├── config.ini          # User-specific data paths (NOT committed ideally)
│   │   ├── settings.py         # Loads config.ini → exports DATA_DIR, ANALYSIS_DIR, etc.
│   │   ├── schema.py           # DataJoint schema definition (all DB tables)
│   │   └── config_create.py    # PyQt6 helper for creating config via file dialog
│   ├── classes/
│   │   ├── analysis_chunk.py   # AnalysisChunk: spatial noise chunk data (STAs, RFs, EIs)
│   │   ├── stim.py             # StimBlock, MEAStimBlock, MEAStimGroup
│   │   ├── response.py         # ResponseBlock, MEAResponseBlock, SCResponseBlock, MEAResponseGroup
│   │   ├── mea_pipeline.py     # MEAPipeline: aggregates stim + response + analysis chunk
│   │   ├── sc_pipeline.py      # PresentImagesSplitter, ExpandingSpotsPipeline (single-cell)
│   │   ├── qc.py               # MEAQC: quality control (ISI violations, spike counts)
│   │   ├── dedup.py            # DedupBlock: duplicate cluster detection
│   │   └── raw.py              # RawTraces: reads raw .bin electrode data
│   ├── utils/
│   │   ├── __init__.py         # Re-exports settings + schema references
│   │   ├── database_utils.py   # populate_database(), reload_experiment_data(), etc.
│   │   ├── database_pop.py     # Low-level DB population (append_experiment, etc.)
│   │   ├── datajoint_utils.py  # Query helpers (get_exp_summary, get_epoch_data_from_exp, etc.)
│   │   ├── vision_utils.py     # VCD loading, cluster_match(), ei_corr(), get_ells(), etc.
│   │   ├── ei_utils.py         # EI reshaping, electrode map sorting, EI visualization
│   │   ├── regen.py            # Stimulus regeneration (spatial noise, PresentImages, Doves, etc.)
│   │   ├── spike_detector.py   # KMeans-based spike detection for single-cell recordings
│   │   └── parse_data.py       # Symphony2 H5 parser → JSON metadata (large, ~1770 lines)
│   ├── preprocessing/
│   │   ├── ks_to_vision.py     # Convert Kilosort output to vision files
│   │   └── ei_merge.py         # Merge EIs across datafiles within a chunk
│   ├── sorting/
│   │   └── run_vision_ss.sh    # Shell script for running Vision spike sorting
│   └── assets/
│       ├── cell_types.csv      # Canonical cell type list
│       ├── classification_files/  # Auto-classification outputs
│       ├── cone_spectra/       # Mouse, primate, zebrafish cone spectra
│       └── nd_filters/         # Neutral density filter data
└── tests/
    ├── import_test.py          # Basic smoke test
    └── regen_tests/            # Stimulus regeneration tests
```

---

## 3. Critical Import Order & Circular Import Rules

The `__init__.py` has a very specific import order that **must be preserved**:

1. `schema.py` — Must be first (initializes DataJoint connection)
2. `config/settings.py` — Exports path constants (DATA_DIR, ANALYSIS_DIR, etc.)
3. `utils/` — Utility modules. **Utils must NEVER import from `classes/`** except via `TYPE_CHECKING` guards (see `vision_utils.py` for the pattern)
4. `classes/` — Import last, as they depend on utils

**Violating this order causes circular import errors.** When modifying utils that need class type hints, use:
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from retinanalysis.classes.analysis_chunk import AnalysisChunk
```

---

## 4. Data Flow & Architecture

### 4.1 Database Layer
- **Docker MySQL container** runs DataJoint MySQL 8.0 on `127.0.0.1:3306` (user: `root`, password: `simple`)
- **Schema** (`schema.py`): Defines tables — Experiment → Animal → Preparation → Cell → EpochGroup → EpochBlock → Epoch → Response/Stimulus. Also analysis tables: SortingChunk, SortedCell, CellTypeFile, SortedCellType
- **Population** (`database_pop.py`): Reads JSON metadata files (parsed from Symphony H5) and inserts hierarchically
- **Must be running** before importing retinanalysis (connection attempt happens at import time in `schema.py`, but gracefully falls back)

### 4.2 Config Layer
- `config.ini` defines paths per OS (Darwin/Linux/Windows) with DEFAULT and SECONDARY sections
- `settings.py` loads the config at import time, exports: `DATA_DIR`, `RAW_DIR`, `ANALYSIS_DIR`, `H5_DIR`, `META_DIR`, `TAGS_DIR`, `QUERY_DIR`, `USER`
- **DEFAULT** = usually an SSD with a data subset; **SECONDARY** = NAS with full data
- Falls back to secondary if primary path doesn't exist; warns if neither exists

### 4.3 Vision Data Layer
- **Vision Cell Data (VCD)**: Loaded via `visionloader.load_vision_data()` from the `artificial-retina-software-pipeline` submodule
- **Analysis VCD** (from `ANALYSIS_DIR`): Contains `.sta`, `.params`, `.neurons` files for spatial noise chunks — has STAs, timecourses, RF parameters
- **Protocol VCD** (from `DATA_DIR`): Contains spike times and EIs for individual protocol datafiles
- Vision files are the output of converting Kilosort output through the pipeline utilities

### 4.4 Analysis Pipeline Flow

```
Experiment (e.g., "20260318C")
  └── has multiple Sorting Chunks (e.g., chunk1, chunk2)
       └── each chunk = a set of datafiles sorted together by Kilosort
            └── one chunk is typically the "noise" chunk (SpatialNoise protocol)
                 └── contains STAs, RF params, cell typing files

User workflow:
1. ra.populate_database()                    # One-time DB setup
2. ra.get_exp_summary("20260318C")           # View experiment timeline
3. pipeline = ra.create_mea_pipeline(        # Create analysis pipeline
       "20260318C", "data013", "chunk2")
4. pipeline.plot_rfs(cell_types=['OnP'])     # Visualize RFs
5. pipeline.get_psth_arr(bin_rate=1000)      # Get PSTH data
```

### 4.5 Cluster Matching
The core algorithmic contribution: matching cells across different recording files using EI (electrical image) correlation. Three methods:
- **full**: Flatten 512×201 EI, correlate entire vectors
- **space**: Max absolute value over time → 512-dimensional vector
- **power**: Mean of squared EI over time → 512-dimensional vector

Bidirectional verification ensures that if cell A's best match is cell B, then cell B's best match must also be cell A.

---

## 5. Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `AnalysisChunk` | `analysis_chunk.py` | Loads noise chunk data: STAs, RFs, EIs, timecourses, cell typing |
| `MEAStimBlock` | `stim.py` | Stimulus metadata for one MEA datafile |
| `MEAStimGroup` | `stim.py` | Concatenated stimulus from multiple datafiles |
| `MEAResponseBlock` | `response.py` | Spike times + EIs for one MEA datafile |
| `MEAResponseGroup` | `response.py` | Concatenated responses from multiple datafiles |
| `SCResponseBlock` | `response.py` | Single-cell recording response (amp data + spike detection) |
| `MEAPipeline` | `mea_pipeline.py` | Aggregates stim + response + analysis chunk with cluster matching |
| `MEAQC` | `qc.py` | Quality control metrics (ISI violations, spike counts) |
| `DedupBlock` | `dedup.py` | Detects duplicate clusters via EI and spatial map autocorrelation |
| `RawTraces` | `raw.py` | Reads raw .bin electrode recordings |

---

## 6. Environment & Execution Rules

- **Conda Environment**: Use `retinanalysis` environment with Python 3.11. All commands must run in this environment.
- **Docker Required**: The DataJoint MySQL container must be running before importing the package. Start with `docker compose up -d`.
- **Real Data**: Tests and development should use real datasets on the machine. Data lives on mounted NAS (`/Volumes/data-1/` or `/mnt/lab/`) and/or local SSD.
- **Submodule**: The `lib/artificial-retina-software-pipeline` submodule must be cloned (`--recursive`) and installed separately via `pip install .`
- **Config file**: Each user must create their own `config.ini` with paths appropriate to their machine. The checked-in version contains the primary developer's paths.

---

## 7. Key Dependencies

| Package | Purpose |
|---------|---------|
| `datajoint==0.14.4` | ORM for MySQL database schema and queries |
| `visionloader` | Reads Vision analysis files (from submodule) |
| `visionwriter` | Writes Vision files (from submodule) |
| `bin2py` | Reads raw Litke array .bin files (from submodule) |
| `numpy`, `scipy`, `pandas` | Core numerical/data libraries |
| `matplotlib`, `seaborn` | Plotting |
| `xarray`, `dask`, `netCDF4` | Multi-dimensional array support for PSTH data |
| `torch` | Listed as dependency (future ML work, not currently used in core) |
| `h5py`, `hdf5storage` | HDF5 file reading |
| `opencv-python` | Image processing for stimulus regeneration |
| `scikit-learn` | KMeans clustering in spike detector |

---

## 8. Known Issues & Technical Debt

1. **`config.ini` is committed**: Contains hardcoded user-specific paths. Should be `.gitignore`'d with a `config.ini.template` instead.
2. **Bare `except:` blocks**: Many try/except blocks catch all exceptions silently (e.g., EI loading in `AnalysisChunk.__init__`). These hide real errors.
3. **Duplicated code**: `add_cell_types()` method is nearly identical in `MEAResponseBlock` and `MEAResponseGroup`. Cell type parsing logic is also duplicated in `AnalysisChunk.get_df()`.
4. **No formal test suite**: Only a smoke test (`import_test.py`) and one regen test exist. No pytest, no CI.
5. **Import-time side effects**: Importing the package triggers DB connection and config file loading. This makes testing difficult.
6. **Hardcoded constants**: Frame rates (`60.31807657`, `59.941548817817917`), sample rate (`20000`), date thresholds (`20230926`) scattered across files.
7. **Mutable default arguments**: Some constructors modify input dicts in place.
8. **`pickle` for serialization**: Multiple classes use pickle for export/import, which is fragile across versions.
9. **Large monolithic files**: `regen.py` (~1250 lines), `parse_data.py` (~1770 lines), `datajoint_utils.py` (~800 lines) could be split.
10. **Mixed concerns in classes**: `AnalysisChunk` and `MEAPipeline` contain both data loading and plotting logic.
11. **No logging**: Uses `print()` throughout instead of Python's `logging` module.
12. **Type hints are incomplete**: Some functions have type hints, many don't. Several use bare `dict` or `list` without element types.

---

## 9. Git Protocol

- The repo uses git submodules. Always clone with `--recursive`.
- Use descriptive branch names: `feat/`, `fix/`, `refactor/`, `test/`
- Keep commits atomic and well-described.

---

## 10. File Modification Checklist

Before modifying any file, verify:

1. **Import order**: Will your change affect `__init__.py` import chain?
2. **Circular imports**: Are you importing a class in a utility module? Use `TYPE_CHECKING` guard.
3. **Database dependency**: Does your code assume the Docker container is running?
4. **Config dependency**: Does your code assume specific paths exist?
5. **Submodule dependency**: Does your code use `visionloader`, `visionwriter`, or `bin2py`?
6. **Backward compatibility**: Will existing notebooks break? (Users rely on `ra.` namespace heavily)
