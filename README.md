# RetinAnalysis
MEA and Single Cell Ephys Analysis Package

## Quickstart

These steps get a fresh macOS/Linux setup running with the local DataJoint database workflow.

1. Install Docker Engine. On macOS, install Docker Desktop from <a href='https://docs.docker.com/desktop/'>https://docs.docker.com/desktop/</a>.

2. Clone the repo with submodules and run the installer with either `uv` or `conda`:

```bash
git clone https://github.com/DRezeanu/retinanalysis.git --recursive
cd retinanalysis
./install.sh uv
# or:
./install.sh conda
```

The installer creates/uses the Python environment, installs `retinanalysis`, installs the local `vision-utils` package from the bundled submodule, and creates `src/retinanalysis/config/config.ini` if it is missing.

Useful installer options:

```bash
./install.sh uv --dev
./install.sh conda --dev --env retinanalysis
./install.sh uv --python 3.11.13
```

3. Edit `src/retinanalysis/config/config.ini` and replace the placeholder paths with real data paths for your machine or mounted drive.

4. Start the local DataJoint/MySQL database and populate it:

```bash
mkdir -p ../retinanalysis-database
cp docker-compose.yaml ../retinanalysis-database/
cd ../retinanalysis-database
docker compose up -d
```

Then, from the installed Python environment:

```python
import retinanalysis as ra
ra.populate_database()
```

`import retinanalysis as ra` does not require the database to be running, but database-backed calls such as queries and `ra.populate_database(...)` do.

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

UV is meant to work with environments at the project level, not system-wide. So you will want to install retinanalysis at the root of every project in which you want to use it (the packages are cached so you aren't using any additional disk space). Virtual environments live in the root of the project in a .venv folder by default, and are named after the root of the project by default. 

2. Create a uv venv in your local project directory using python 3.11.13:
```
uv venv --python 3.11.13
```

3. Activate the uv environment, cd to the package directory, and use `uv pip` to install all required dependencies:
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
Use paths that match the machine or mounted drive where your local copies of the MEA data live.
```
[DEFAULT]
analysis = /path/to/analysis
data = /path/to/sorted
raw = /path/to/raw
h5 = /path/to/h5
meta = /path/to/meta
tags = /path/to/tags
query = /path/to/query/analysis
user = your_username

[SECONDARY]
analysis = /path/to/secondary/analysis
data = /path/to/secondary/sorted
raw = /path/to/secondary/raw
h5 = /path/to/secondary/h5
meta = /path/to/secondary/meta
tags = /path/to/secondary/tags
query = /path/to/secondary/query/analysis
user = your_username

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

Retinanalysis uses a custom DataJoint MySQL database to store experiment metadata. DataJoint 2 requires MySQL 8.

We've included a modified docker-compose.yaml file for easy installation using the steps below:

6. Install Docker Desktop from <a href='https://docs.docker.com/desktop/'>https://docs.docker.com/desktop/</a>

7. Copy the docker-compose.yaml file from the repository's root into an empty directory where you
   will store your database. You can create this folder in the repository root if you'd like,
   but you must add it to your .gitignore if you do this.

8. cd into the new directory and run:

```
docker-compose up -d
```

If you have newer versions of Docker, the command syntax is:

```
docker compose up -d
```

NOTE: `import retinanalysis as ra` no longer requires the database to be running; however, database-backed calls, such as queries or `ra.populate_database(...)`, require the local database container to be running.

Before running database-backed calls, make sure the container is running in Docker Desktop (or through the terminal if you're comfortable with the Docker CLI). If it is running, you will see a stop icon; otherwise, click the play button.

<img width="1382" height="832" alt="Screenshot 2025-10-24 at 3 00 20 PM" src="https://github.com/user-attachments/assets/45ee0d03-6dd7-48c4-ad38-c75e558259ed" />

9. Populate the database. Before you can look up anything in the database you need to fill its entries. To populate a fresh database, run:

```
import retinanalysis as ra
ra.populate_database()
```

If you have properly set up your config.ini file, there should be no need to give this function any input arguments.

## UPDATE (Jun3 2026): DataJoint 2 migration note for existing users

Retinanalysis now uses `datajoint==2.2.2`. To use the latest version of retinanalysis, existing users should reinstall or update retinanalysis in their analysis environment, create a fresh local DataJoint/MySQL database (use the docker compose file to initialize a fresh database per steps 7 and 8 above), and repopulate that database (per step 9) AFTER retinanalysis has been updated to datajoint 2.2.2. Do not try to update an old DataJoint 0.14 database in place.

We recommend doing this in a fresh conda or uv environment, and keeping the old database and retinanalysis installation until you have confirmed that the updated version is not causing any issues in your analysis code.   

## DataJoint configuration

Retinanalysis provides fallback local DataJoint settings for the common lab workflow, but DataJoint 2 will warn you if it does not find a `datajoint.json` file in your project root. You can safely ignore this warning, but if you want to make the connection explicit and avoid that warning, create a `datajoint.json` file in the root of your analysis project using the values below:

```json
{
  "database.host": "127.0.0.1",
  "database.port": 3306,
  "database.user": "root",
  "database.password": "simple"
}
```

Project-level `datajoint.json`, environment variables, or explicit `datajoint.config` settings take precedence over retinanalysis' local fallback settings.
