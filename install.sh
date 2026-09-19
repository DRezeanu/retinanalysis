#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON="3.11.13"
DEFAULT_CONDA_ENV="retinanalysis"
DEV=0
CONFIG=0
PYTHON_VERSION="$DEFAULT_PYTHON"
CONDA_ENV="$DEFAULT_CONDA_ENV"
MODE=""
ENV_FLAG_USED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./install.sh uv [--python VERSION] [--dev] [--config]
  ./install.sh conda [--python VERSION] [--dev] [--config] [--env NAME]

Modes:
  uv       Create/use the repo-local .venv and install retinanalysis with uv.
  conda    Create/use a conda environment and install retinanalysis with pip.

Options:
  --python VERSION   Python version to request. Default: 3.11.13.
                     Must satisfy pyproject.toml requires-python.
  --dev              Install development/test dependencies.
  --env NAME         Conda environment name. Default: retinanalysis.
                     Only valid in conda mode. An existing environment is reused if its
                     Python major/minor matches --python; otherwise the script stops.
  --config           Launch config GUI using the setup_gui() method. Do not use
                     if running 'headless' or over SSH. The program attempts to open
                     a window that allows you to browse for all relevant paths.
  -h, --help         Show this help message.

Notes:
  - This script does not run git submodule commands automatically.
  - This script does not create, populate, migrate, or modify DataJoint databases.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '\n==> %s\n' "$*"
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  uv|conda)
    MODE="$1"
    shift
    ;;
  *)
    usage
    fail "first argument must be one of: uv, conda"
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || fail "--python requires a version argument"
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --dev)
      DEV=1
      shift
      ;;
    --env)
      [[ $# -ge 2 ]] || fail "--env requires an environment name"
      CONDA_ENV="$2"
      ENV_FLAG_USED=1
      shift 2
      ;;
    --config)
      CONFIG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      fail "unknown option: $1"
      ;;
  esac
done

if [[ "$MODE" != "conda" && "$ENV_FLAG_USED" -eq 1 ]]; then
  fail "--env is only valid in conda mode"
fi

require_repo_root() {
  [[ -f "pyproject.toml" ]] || fail "pyproject.toml not found. Run this script from the retinanalysis repo root."
  [[ -d "src/retinanalysis" ]] || fail "src/retinanalysis not found. Run this script from the retinanalysis repo root."
}

extract_python_requirement() {
  local requirement
  requirement="$(grep -E '^requires-python[[:space:]]*=' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')"
  [[ -n "$requirement" ]] || fail "could not read requires-python from pyproject.toml"
  printf '%s' "$requirement"
}

parse_major_minor() {
  local version="$1"
  if [[ ! "$version" =~ ^([0-9]+)\.([0-9]+)(\.[0-9]+)?$ ]]; then
    fail "Python version '$version' must look like MAJOR.MINOR or MAJOR.MINOR.PATCH"
  fi
  printf '%s %s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
}

check_python_requirement() {
  local version="$1"
  local requirement="$2"

  if [[ ! "$requirement" =~ ^\>=([0-9]+)\.([0-9]+)$ ]]; then
    fail "unsupported requires-python format '$requirement'; expected a simple '>=MAJOR.MINOR' requirement"
  fi

  local min_major="${BASH_REMATCH[1]}"
  local min_minor="${BASH_REMATCH[2]}"
  local parsed req_major req_minor
  parsed="$(parse_major_minor "$version")"
  req_major="${parsed%% *}"
  req_minor="${parsed##* }"

  if (( req_major < min_major )) || { (( req_major == min_major )) && (( req_minor < min_minor )); }; then
    fail "requested Python $version does not satisfy pyproject.toml requires-python '$requirement'"
  fi
}

check_requested_major_minor_matches() {
  local actual="$1"
  local requested="$2"
  local context="$3"
  local advice="$4"
  local actual_parsed requested_parsed
  actual_parsed="$(parse_major_minor "$actual")"
  requested_parsed="$(parse_major_minor "$requested")"

  if [[ "$actual_parsed" != "$requested_parsed" ]]; then
    fail "$context uses Python $actual, but requested Python $requested. $advice"
  fi
}

check_submodule() {
  local vision_utils_dir="lib/artificial-retina-software-pipeline/utilities"
  if [[ ! -d "$vision_utils_dir" ]] || [[ ! -f "$vision_utils_dir/pyproject.toml" && ! -f "$vision_utils_dir/setup.py" ]]; then
    cat >&2 <<'EOF'
ERROR: required submodule package is missing or incomplete:
  lib/artificial-retina-software-pipeline/utilities

Run this command yourself, then rerun install.sh:
  git submodule update --init --recursive
EOF
    exit 1
  fi
}

# Windows venvs put the interpreter in Scripts/, POSIX in bin/. Probe for the file
# rather than testing $OSTYPE or uname -- Git Bash reports MINGW64_NT/MSYS_NT
# inconsistently across installs, and the file either exists or it does not. Prints
# nothing when there is no .venv yet.
find_venv_python() {
  if [[ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]]; then
    printf '%s' "$SCRIPT_DIR/.venv/Scripts/python.exe"
  elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    printf '%s' "$SCRIPT_DIR/.venv/bin/python"
  fi
}

install_uv() {
  command -v uv >/dev/null 2>&1 || fail "uv not found on PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/"

  # Validate an existing .venv BEFORE uv venv runs. `uv venv --allow-existing` keeps the
  # directory but rebuilds the interpreter at whatever --python asks for, so checking
  # afterwards can never fail: the probe would only ever report the version we just
  # requested, and a working environment would already have been replaced.
  local existing_venv_python existing_python
  existing_venv_python="$(find_venv_python)"
  if [[ -n "$existing_venv_python" ]]; then
    existing_python="$("$existing_venv_python" -c "import platform; print(platform.python_version())")"
    check_python_requirement "$existing_python" "$PYTHON_REQUIREMENT"
    check_requested_major_minor_matches "$existing_python" "$PYTHON_VERSION" "repo-local .venv" "Remove .venv or request its Python major/minor version."
  fi

  info "Creating/updating repo-local .venv with Python $PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" --allow-existing

  local venv_python
  venv_python="$(find_venv_python)"
  [[ -n "$venv_python" ]] || fail "expected interpreter not found under $SCRIPT_DIR/.venv"

  # vision-utils first, for the same reason as the conda path below: it is declared in
  # [project.dependencies] but only resolvable through [tool.uv.sources]. uv reads that
  # table so the order does not strictly matter here, but keeping both backends in the
  # same order means one less thing that is true on only one path.
  info "Installing local vision-utils package from submodule"
  uv pip install --python "$venv_python" "$SCRIPT_DIR/lib/artificial-retina-software-pipeline/utilities"

  info "Installing retinanalysis editable"
  uv pip install --python "$venv_python" -e "$SCRIPT_DIR"

  if [[ "$DEV" -eq 1 ]]; then
    info "Installing development/test dependencies"
    # Resolved from [dependency-groups] dev in pyproject.toml (cwd is the repo root),
    # so this never drifts out of sync with the project definition.
    uv pip install --python "$venv_python" --group dev
  fi

  if [[ "$CONFIG" -eq 1 ]]; then
    info "Launching config setup GUI"
    # Use the interpreter we just installed into, not `uv run`. `uv run` re-syncs the
    # project environment against pyproject.toml/uv.lock before executing, which can
    # undo the --python selection and the editable installs performed just above.
    "$venv_python" -c "import retinanalysis as ra; ra.config.setup_gui()"
  fi

}

# pip only learned to read [dependency-groups] from pyproject.toml (PEP 735) in 25.1.
# Conda environments are frequently created with whatever pip ensurepip bundled, which
# can be much older, so probe before relying on --group. Keep DEV_FALLBACK in sync with
# [dependency-groups] dev in pyproject.toml.
DEV_FALLBACK="pytest>=9.0.3"

conda_pip_supports_dependency_groups() {
  conda run -n "$CONDA_ENV" python -c "import sys; from importlib.metadata import version; sys.exit(0 if tuple(int(n) for n in version('pip').split('.')[:2]) >= (25, 1) else 1)" >/dev/null 2>&1
}

conda_env_exists() {
  conda run -n "$CONDA_ENV" python --version >/dev/null 2>&1
}

conda_env_python_version() {
  # This MUST stay a `python -c` one-liner. The obvious `python - <<'PY'` heredoc form
  # works on macOS/Linux but silently returns an empty string on Windows, because
  # `conda run` does not forward stdin to the child process there. The empty result then
  # reaches parse_major_minor and aborts the install with the baffling message
  # "Python version '' must look like MAJOR.MINOR or MAJOR.MINOR.PATCH".
  conda run -n "$CONDA_ENV" python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
}

install_conda() {
  command -v conda >/dev/null 2>&1 || fail "conda not found on PATH. Install Miniconda/Mambaforge or use './install.sh uv'."

  if conda_env_exists; then
    local existing_python
    existing_python="$(conda_env_python_version | tail -n 1)"
    check_python_requirement "$existing_python" "$PYTHON_REQUIREMENT"
    check_requested_major_minor_matches "$existing_python" "$PYTHON_VERSION" "existing conda env '$CONDA_ENV'" "Use a different --env or update the environment yourself."
    info "Using existing conda environment '$CONDA_ENV' with Python $existing_python"
  else
    info "Creating conda environment '$CONDA_ENV' with Python $PYTHON_VERSION"
    conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
  fi

  # vision-utils MUST be installed before retinanalysis here. It is declared in
  # [project.dependencies] but is resolvable only through [tool.uv.sources], which is a
  # uv-only table -- pip ignores everything under [tool] that is not its own. So pip
  # treats "vision-utils" as an ordinary requirement, looks for it on PyPI, and fails,
  # because it is not published there. Installing it first leaves the requirement already
  # satisfied by an installed distribution, so pip never queries an index for it.
  #
  # --no-capture-output on every long-running or interactive step. Without it `conda run`
  # buffers the child's stdout/stderr until the command exits, so multi-minute pip
  # installs and the config GUI look like the script has hung.
  info "Installing local vision-utils package from submodule into conda env '$CONDA_ENV'"
  conda run --no-capture-output -n "$CONDA_ENV" python -m pip install "$SCRIPT_DIR/lib/artificial-retina-software-pipeline/utilities"

  info "Installing retinanalysis editable into conda env '$CONDA_ENV'"
  conda run --no-capture-output -n "$CONDA_ENV" python -m pip install -e "$SCRIPT_DIR"

  if [[ "$DEV" -eq 1 ]]; then
    info "Installing development/test dependencies into conda env '$CONDA_ENV'"
    if conda_pip_supports_dependency_groups; then
      conda run --no-capture-output -n "$CONDA_ENV" python -m pip install --group dev
    else
      info "pip in '$CONDA_ENV' predates PEP 735; installing dev packages by name instead"
      conda run --no-capture-output -n "$CONDA_ENV" python -m pip install "$DEV_FALLBACK"
    fi
  fi

  if [[ "$CONFIG" -eq 1 ]]; then
    info "Launching config setup GUI"
    conda run --no-capture-output -n "$CONDA_ENV" python -c "import retinanalysis as ra; ra.config.setup_gui()"
  fi

}

require_repo_root
PYTHON_REQUIREMENT="$(extract_python_requirement)"
check_python_requirement "$PYTHON_VERSION" "$PYTHON_REQUIREMENT"
check_submodule

info "Install mode: $MODE"
printf 'Python request: %s (project requires %s)\n' "$PYTHON_VERSION" "$PYTHON_REQUIREMENT"
if [[ "$MODE" == "conda" ]]; then
  printf 'Conda environment: %s\n' "$CONDA_ENV"
fi

case "$MODE" in
  uv)
    install_uv
    ;;
  conda)
    install_conda
    ;;
esac


cat <<'EOF'

Installation finished.

Next steps:
  1. Import the package to make sure everything installed correctly:
       
       import retinanalysis as ra

  2. If you didn't use the --config flag, you must create a config file that
     contains the paths to the relevant directories:
       
       Option 1, use the CLI setup tool:
            ra.config.setup()
        
       Option 2, use the GUI setup tool:
            ra.config.setup_gui()

       Option 3, write your config file manually:
            See src/retinanalysis/assets/config_template.toml
            The directory where you should put this file is printed when you first import retinanalysis

  3. Start the Docker/DataJoint database only when you need database-backed workflows.

  4. After config file and the database are ready, populate a fresh database from Python:

       import retinanalysis as ra
       ra.populate_database()

EOF
