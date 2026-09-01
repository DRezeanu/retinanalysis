/# RetinAnalysis
MEA and Single Cell Ephys Analysis Package

## Choosing an installer

There are two quickstart installers. They accept the same modes and the same options and perform the
same steps, so pick whichever suits your shell:

| Platform | Command | Where to run it |
| --- | --- | --- |
| macOS, Linux | `./install.sh uv` / `./install.sh conda` | Any terminal |
| Windows + uv | `.\install.ps1 uv` | Windows PowerShell (or Git Bash, using `./install.sh uv`) |
| Windows + conda | `.\install.ps1 conda` | **Anaconda PowerShell Prompt** (Start menu) |

If Windows refuses to run the PowerShell script (`running scripts is disabled on this
system`), either launch it as

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 uv
```

or allow local scripts once, for your user account only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Quickstart

These steps get a fresh macOS, Linux, or Windows setup running with the local DataJoint
database workflow.

1. Install Docker Engine. On macOS and Windows, install Docker Desktop from <a href='https://docs.docker.com/desktop/'>https://docs.docker.com/desktop/</a>.

2. Clone the repo with submodules and run the installer with either `uv` or `conda`:

```bash
git clone https://github.com/DRezeanu/retinanalysis.git --recursive
cd retinanalysis
./install.sh uv
# or:
./install.sh conda
```

On Windows, use the PowerShell installer instead (conda mode requires the Anaconda
PowerShell Prompt):

```powershell
git clone https://github.com/DRezeanu/retinanalysis.git --recursive
cd retinanalysis
.\install.ps1 uv
# or:
.\install.ps1 conda
```

The installer creates/uses the Python environment, installs `retinanalysis`, installs the local `vision-utils` package from the bundled submodule, and has the option of launching the config creation GUI. 

Useful installer options:

```bash
./install.sh uv --dev --config
./install.sh conda --dev --env custom_name
./install.sh uv --python 3.12
```

```powershell
.\install.ps1 uv -Dev -Config
.\install.ps1 conda -Dev -Env custom_name
.\install.ps1 uv -Python 3.12
```

- Option 1 above installs using uv, including the developer dependencies (pytest, etc.), and launches the config creation GUI. Note: Do NOT use the --config flag if installing headless on a server/over SSH, as it will try to open a GUI window.
- Option 2 above installs using conda, including the developer dependencies, and sets the conda environment name to 'custom_name' instead of the default 'retinanalysis'
- Option 3 above installs using uv, without developer dependencies, and overwrites the default python version 3.11.13 with 3.12. The version you give using `--python`/`-Python` must be greater than or equal to Python 3.10.

Note that if a `.venv` or conda environment already exists and its Python major/minor
version differs from the one you request, the installer stops and leaves that environment
untouched rather than rebuilding it. Delete the environment first, or ask for the version
it already has.

3. If you did not use the --config flag, import retinanalysis and run config setup:

```python
import retinanalysis as ra
ra.config.setup()
# OR
ra.config.setup_gui()
```

4. Create a database directory outside of the repo root, copy the docker compose file from retinanalysis into this new directory, cd into the database directory, and run docker compose to create the blank database:

```bash
mkdir -p ../retinanalysis-database
cp docker-compose.yaml ../retinanalysis-database/
cd ../retinanalysis-database
docker compose up -d
```

Then, from the installed Python environment, run:

```python
import retinanalysis as ra
ra.populate_database()
```

Note: Populating the database can take a long time the first time you do it. 

`import retinanalysis as ra` does not require the database to be running, but database-backed calls such as queries and `ra.populate_database(...)` do. 

## Installation
1. Pull retinanalysis repo (include --recursive flag to get required submodules contained in 'lib' folder):
```bash
git clone https://github.com/DRezeanu/retinanalysis.git --recursive 
```

The `--recursive` flag is required, not optional — `vision-utils` is a git submodule under
`lib/`, is not published on PyPI, and can only be installed from that directory. If you already
cloned without it:
```bash
git submodule update --init --recursive
```
---
### Install with Conda

#### First time using conda on this machine?

conda 25.x and newer refuse to create any environment until you accept the Terms of
Service for Anaconda's default channels. This surfaces the moment the installer calls
`conda create`:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels.
```

Accept them once, then rerun the installer:

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

#### Manual conda install

2. Create a conda environment using python 3.11.13:
```bash
conda create --name retinanalysis python=3.11.13
```

3. Activate conda environment, cd to the package directory, and install `vision-utils` from the
bundled submodule **before** installing retinanalysis:
```bash
conda activate retinanalysis
cd repositories_dir/retinanalysis
pip install lib/artificial-retina-software-pipeline/utilities
pip install -e . 
```
**The order matters.** `vision-utils` is listed in `[project.dependencies]`, but the path that
points at the submodule lives in `[tool.uv.sources]`, which only uv reads — pip ignores it. So a
plain `pip install -e .` on its own looks for `vision-utils` on PyPI, where it does not exist, and
fails with `No matching distribution found for vision-utils`. Installing the submodule first
leaves the requirement already satisfied, so pip never goes looking.

`./install.sh conda` (or `.\install.ps1 conda` on Windows) does both steps in the right
order for you.

---
### Install with uv
UV is a new, highly recommended python package and project manager written in Rust that works extremely fast. You can learn more about it here: https://docs.astral.sh/uv/

UV is meant to work with environments at the project level, not system-wide. So you will want to install retinanalysis at the root of every project in which you want to use it (the packages are cached so you aren't using any additional disk space). Virtual environments live in the root of the project in a .venv folder by default, and are named after the root of the project by default. 

2. Create a uv venv in your local project directory using python 3.11.13:
```bash
uv venv --python 3.11.13
```

3. Activate the uv environment, cd to the package directory, and use `uv pip` to install all required dependencies:
```bash
# On Mac and Linux:
source .venv/bin/activate
# On Windows (Git Bash)
source .venv/Scripts/activate

# On all systems
cd ../*your_repositories_directory*/retinanalysis
uv pip install -e . 
```
Unlike pip, uv reads `[tool.uv.sources]` in `pyproject.toml`, so it resolves `vision-utils`
straight from `lib/artificial-retina-software-pipeline/utilities` and installs it automatically.
No separate step is needed on this path.

---
### Installation Note for Windows Users

The above requirements have been tested to work on Mac (MacOS Tahoe, Sequoia and Sonoma), Linux (Ubuntu 24.04 LTS), and Windows 11 Pro 25H2 (both `install.ps1` backends and `install.sh uv` under Git Bash).

On older Windows 11 versions, you may receive a DLL error when the package attempts to import matplotlib for the first time. To fix this, run:
```bash
pip uninstall Pillow *or* uv pip uninstall Pillow
pip install -U Pillow *or* uv pip install -U Pillow
```
---
5. Create a config file using the built-in config.setup() or config.setup_gui() methods, or manually create a config.toml file and place it in your platform-specific config directory. You can find a config template at `src/retinanalysis/assets/config_template.toml`:

- Mac and Linux: ~/.config/retinanalysis/
- Windows: C:\Users\YourUsername\AppData\Local\retinanalysis

To use the setup methods:

```python
import retinanalysis as ra
ra.config.setup()
#OR
ra.config.setup_gui()
```

## Docker Installation

Retinanalysis uses a custom DataJoint MySQL database to store experiment metadata. DataJoint 2 requires MySQL 8.

We've included a modified docker-compose.yaml file for easy installation using the steps below:

6. Install Docker Desktop from <a href='https://docs.docker.com/desktop/'>https://docs.docker.com/desktop/</a>

7. Copy the docker-compose.yaml file from the repository's root into an empty directory where you
   will store your database. You can create this folder in the repository root if you'd like,
   but you must add it to your .gitignore if you do this.

8. cd into the new directory and run:

```bash
docker-compose up -d
```

If you have newer versions of Docker, the command syntax is:

```bash
docker compose up -d
```

NOTE: `import retinanalysis as ra` no longer requires the database to be running; however, database-backed calls, such as queries or `ra.populate_database(...)`, require the local database container to be running.

Before running database-backed calls, make sure the container is running in Docker Desktop (or through the terminal if you're comfortable with the Docker CLI). If it is running, you will see a stop icon; otherwise, click the play button.

<img width="1382" height="832" alt="Screenshot 2025-10-24 at 3 00 20 PM" src="https://github.com/user-attachments/assets/45ee0d03-6dd7-48c4-ad38-c75e558259ed" />

9. Populate the database. Before you can look up anything in the database you need to fill its entries. To populate a fresh database, run:

```python
import retinanalysis as ra
ra.populate_database()
```

If you have properly set up your config, there should be no need to give this function any input arguments.

## UPDATE (Jun3 2026): DataJoint 2 migration note for existing users

Retinanalysis now uses `datajoint==2.2.2`. To use the latest version of retinanalysis, existing users should reinstall or update retinanalysis in their analysis environment, create a fresh local DataJoint/MySQL database (use the docker compose file to initialize a fresh database per steps 7 and 8 above), and repopulate that database (per step 9) AFTER retinanalysis has been updated to datajoint 2.2.2. Do not try to update an old DataJoint 0.14 database in place.

We recommend doing this in a fresh conda or uv environment, and keeping the old database and retinanalysis installation until you have confirmed that the updated version is not causing any issues in your analysis code.   

## DataJoint configuration

Retinanalysis provides fallback local DataJoint settings for the common lab workflow, but DataJoint 2 will warn you if it does not find a `datajoint.json` file in your project root. You can safely ignore this warning, but if you want to make the connection explicit and avoid that warning, create a `datajoint.json` file in the root of your analysis project using the values below. The file will be ignored by git:

```json
{
  "database.host": "127.0.0.1",
  "database.port": 3306,
  "database.user": "root",
  "database.password": "simple"
}
```

Project-level `datajoint.json`, environment variables, or explicit `datajoint.config` settings take precedence over retinanalysis' local fallback settings.
