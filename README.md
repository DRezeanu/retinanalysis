# RetinAnalysis
MEA and Single Cell Ephys Analysis Package

## Installation
```
cd to root directory
activate your conda environment of choice (if using)
python -m pip install -e . (for dev use)
python -m pip install . (to lock the modules in their current state)
```

## Sample config.ini file (must be in same src folder as settings.py):
```
[DEFAULT]
analysis = /Volumes/Vyom MEA/analysis
data = /Volumes/Vyom MEA/analysis
raw = /Volumes/Vyom MEA/data/raw
h5 = /Volumes/Vyom MEA/data/samarjit_datajoint/data_dirs/data
meta = /Volumes/Vyom MEA/data/samarjit_datajoint/data_dirs/meta
tags = /Volumes/Vyom MEA/data/samarjit_datajoint/data_dirs/tags
query = /Volumes/data-1/analysis
user = vyomr

[SECONDARY]
analysis = /Volumes/data-1/analysis
data = /Volumes/data-1/data/sorted
raw = /Volumes/data-1/data/raw
h5 = /Volumes/data-1/data/h5
meta = /Volumes/data-1/datajoint_testbed/mea/meta
tags = /Volumes/data-1/datajoint_testbed/mea/tags
query = /Volumes/data-1/analysis
user = vyomr

[WINDOWS_DEFAULT]
...

[WINDOWS_SECONDARY]
...
```
`query` dir is used by `datajoint_utils.plot_mosaics_for_all_datasets` and it's useful to have it set to the NAS analysis dir even when all other paths are SSD. This allows loading and plotting mosaics and cell typing from all the data on the NAS instead of just the data on your SSD's `analysis` dir.
