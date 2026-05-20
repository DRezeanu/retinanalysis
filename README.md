# RetinAnalysis
MEA and Single Cell Ephys Analysis Package

## Installation
1. Pull retinanalysis repo (include --recursive flag to get required submodules contained in 'lib' folder):
```
git clone https://github.com/DRezeanu/retinanalysis.git --recursive 
```
---
### Install with Conda
2. Create a conda environment using python 3.11:
```
conda create --name retinanalysis python=3.11.13
```

3. Activate conda environment, cd to the package directory, and use pip and conda to install all required dependencies:
```
conda activate retinanalysis
cd repositories_dir/retinanalysis
pip install -e . 
```

4. Install additional requirements from artificial-retina-software-pipeline submodule:
```
cd repositories_dir/retinanalysis/lib/artificial-retina-software-pipeline/utilities/ 
pip install .
```
---
### Install with uv
UV is a new, very highly recommended python package and project manager written in Rust that works extremely fast. You can learn more about it here: https://docs.astral.sh/uv/

UV is meant to work with environments at the project level, not system-wide. So you will want to install retinanalysis at the root of every project in which you want to use it (the packages are cached so you aren't using any additional disk space). Virtual environments live in the root of the project in a .venv folder by default, and are named after the root of the project by default. This is best practice, however, you can still select and use any virtual environment on your machine in VSCode, so you could still install it once in one project and use it system wide if you'd like.

2. Create a uv venv in your local project directory using python 3.11.13:
```
uv venv --python 3.11.13
```

3. Activate the uv environment, cd to the package directory, and use pip and conda to install all required dependencies:
```
source .venv/bin/activate
cd ../*your_repositories_directory*/retinanalysis
uv pip install -e . 
```

4. Install additional requirements from artificial-retina-software-pipeline submodule in lib:
```
cd lib/artificial-retina-software-pipeline/utilities/ 
uv pip install .
```
---
### Installation Note for Windows Users

The above requirements have been tested to work on both Mac and Linux (Ubuntu 24.04 LTS).

For Windows, you may receive a DLL error when the package attempts to import matplotlib for the first time. To fix this, run:
```
pip uninstall Pillow *or* uv pip uninstall Pillow
pip install -U Pillow *or* uv pip install -U Pillow
```
---
5. Create a config.ini file using the sample version below as a guide and put this config file inside the retinanalysis/src/retinanalysis/config folder in the repo.

## Sample config.ini file:
In this case, the default is an SSD with a subset of the data in it, the secondary is a NAS with all data.
```
[DEFAULT]
analysis = /Volumes/MEA_SSD/analysis
data = /Volumes/MEA_SSD/data/sorted
raw = /Volumes/MEA_SSD/data/raw
h5 = /Volumes/MEA_SSD/data/datajoint_testbed/mea/data
meta = /Volumes/MEA_SSD/data/datajoint_testbed/mea/meta
tags = /Volumes/MEA_SSD/data/datajoint_testbed/mea/tags
query = /Volumes/data-1/analysis
user = drezeanu

[SECONDARY]
analysis = /Volumes/data-1/analysis
data = /Volumes/data-1/data/sorted
raw = /Volumes/data-1/data/raw
h5 = /Volumes/data-1/data/h5
meta = /Volumes/data-1/datajoint_testbed/mea/meta
tags = /Volumes/data-1/datajoint_testbed/mea/tags
query = /Volumes/data-1/analysis
user = drezeanu

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

We've included a modified docker-compose.yaml file for easy installation using the steps below:

6. Install docker desktop from <a href='https://docs.docker.com/desktop/'>https://docs.docker.com/desktop/</a>

7. Copy the docker-compose.yaml file from the repository's root into an empty directory where you
will store your database. You can create this folder in the repository root if you'd like,
but you must add it to your .gitignore if you do this.

8. cd into the new directory and run:
```
docker-compose up -d
```
In case that doesn't work and you have newer versions of Docker, the command syntax is:
```
docker compose up -d
```

NOTE: Before importing retinanalysis, you will need to make sure this container is running in Docker 
Desktop (or throught the terminal if you're comfortable with the Docker CLI). If it is running, you will
see a stop icon, otherwise, click the play button.

<img width="1382" height="832" alt="Screenshot 2025-10-24 at 3 00 20 PM" src="https://github.com/user-attachments/assets/45ee0d03-6dd7-48c4-ad38-c75e558259ed" />

9. Populate database. Before you can look up anything in the database you need to fill its entries. This can take a very long time for big databases, and even longer if connecting remotely over a VPN. To populate the database the first time you import retinanalysis, run:
```
import retinanalysis as ra
ra.populate_database()
```
If you have properly set up your config.ini file, there should be no need to give this function any input arguments.

## Hyak setup
```
apptainer pull docker://datajoint/mysql:8
export APPTAINERENV_MYSQL_ROOT_PASSWORD=simple

mkdir -p ./mysql/run ./mysql/tmp ./mysql/data

screen -S n1
apptainer run \
  --bind ./mysql/data:/var/lib/mysql \
  --bind ./mysql/run:/var/run/mysqld \
  --bind ./mysql/tmp:/tmp \
  mysql_8.sif \
  --user=$(whoami) \
  --datadir=/var/lib/mysql \
  --socket=/var/run/mysqld/mysqld.sock
```

dj config needs to be:
```
dj.config['database.host'] = 'localhost'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'simple'

# Force TLS off explicitly
dj.config['database.use_tls'] = False 
```