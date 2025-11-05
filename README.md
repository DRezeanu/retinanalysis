# RetinAnalysis
MEA and Single Cell Ephys Analysis Package

## Installation
1. Create a conda environment using python 3.11:
```
conda create --name retinanalysis python=3.11.13
```

2. Activate conda environment, cd to the package directory, and use pip and conda to install all required dependencies:
```
conda activate retinanalysis
cd repositories_dir/retinanalysis
pip install -e . 
```

3. Install additional requirements from Chichilnisky Vision repository:
```
cd repositories_dir/artificial-retina-software-pipeline/utilities/ (external requirement from Chichilnisky lab)
pip install .
```

4. Create a config.ini file using the sample version below as a guide and put this config file in
the src/retinanalysis/config directory.

## Note for Windows Users

The above requirements have been tested to work on both Mac and Linux (Ubuntu 24.04 LTS).

For windows, you may receive a DLL error when the package attempts to import matplotlib
for the first time. To fix this, run:
```
pip uninstall Pillow
pip install -U Pillow.
```

## Sample config.ini file:
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

[LINUX_DEFAULT]
...

[LINUX_SECONDARY]
...

[WINDOWS_DEFAULT]
...

[WINDOWS_SECONDARY]
...
```
Note: The `query` dir is used by `datajoint_utils.plot_mosaics_for_all_datasets` and it's useful to have it set to the NAS analysis dir even when all other paths are SSD. This allows loading and plotting mosaics and cell typing from all the data on the NAS instead of just the data on your SSD's `analysis` dir.

## Docker Installation
Retinanalysis uses a custom datajoint mysql database to store all experiment metadata. This uses the datajoine/mysql docker image found at <a href='https://hub.docker.com/r/datajoint/mysql'>https://hub.docker.com/r/datajoint/mysql</a>.
We've included a docker-compose.yaml file for easy installation using the steps below:

1. Install docker desktop from <a href='https://docs.docker.com/desktop/'>https://docs.docker.com/desktop/</a>

2. Copy the docker-compose.yaml file from the repository's root into an empty directory where you
will store your database. You can create this folder in the repository root if you'd like,
but you must add it to your .gitignore if you do this.

7. cd into the new directory and run:
```
docker compose up -d
```

NOTE: Before importing retinanalysis, you will need to make sure this container is running in Docker 
Desktop (or throught the terminal if you're comfortable with the Docker CLI). If it is running, you will
see a stop icon, otherwise, click the play button.

<img width="1382" height="832" alt="Screenshot 2025-10-24 at 3 00 20 PM" src="https://github.com/user-attachments/assets/45ee0d03-6dd7-48c4-ad38-c75e558259ed" />

8. Populate database. Before you can look up anything in the database you need to fill its entries. This can take a very long time for big databases, and even longer if connecting remotely over a VPN. To populate the database the first time you import retinanalysis, run:
```
import retinanalysis as ra
ra.populate_database()
```
If you have properly set up your config.ini file, there should be no need to give this function any input arguments.

