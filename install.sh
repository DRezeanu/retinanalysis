#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON="3.11.13"
DEFAULT_CONDA_ENV="retinanalysis"
DEV=0
PYTHON_VERSION="$DEFAULT_PYTHON"
CONDA_ENV="$DEFAULT_CONDA_ENV"
MODE=""
ENV_FLAG_USED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./install.sh uv [--python VERSION] [--dev]
  ./install.sh conda [--python VERSION] [--dev] [--env NAME]

Modes:
  uv       Create/use the repo-local .venv and install retinanalysis with uv.
  conda    Create/use a conda environment and install retinanalysis with pip.

Options:
  --python VERSION   Python version to request. Default: 3.11.13.
                     Must satisfy pyproject.toml requires-python.
  --dev              Install development/test dependencies.
  --env NAME         Conda environment name. Default: retinanalysis.
                     Only valid in conda mode. Note: Cannot use existing environment!
  -h, --help         Show this help message.

Notes:
  - This script does not run git submodule commands automatically.
  - This script does not create, populate, migrate, or modify DataJoint databases.
  - If config.ini is missing, a placeholder template is created and must be edited.
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

create_config_template() {
  local config_path="src/retinanalysis/config/config.ini"

  if [[ -f "$config_path" ]]; then
    info "Preserving existing $config_path"
    printf 'Please verify that this file contains paths for this machine before using data/database workflows.\n'
    return
  fi

  info "Creating placeholder $config_path"
  mkdir -p "$(dirname "$config_path")"
  cat > "$config_path" <<'EOF'
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
analysis = /path/to/linux/analysis
data = /path/to/linux/sorted
raw = /path/to/linux/raw
h5 = /path/to/linux/h5
meta = /path/to/linux/meta
tags = /path/to/linux/tags
query = /path/to/linux/query/analysis
user = your_username

[LINUX_SECONDARY]
analysis = /path/to/linux/secondary/analysis
data = /path/to/linux/secondary/sorted
raw = /path/to/linux/secondary/raw
h5 = /path/to/linux/secondary/h5
meta = /path/to/linux/secondary/meta
tags = /path/to/linux/secondary/tags
query = /path/to/linux/secondary/query/analysis
user = your_username

[WINDOWS_DEFAULT]
analysis = C:/path/to/analysis
data = C:/path/to/sorted
raw = C:/path/to/raw
h5 = C:/path/to/h5
meta = C:/path/to/meta
tags = C:/path/to/tags
query = C:/path/to/query/analysis
user = your_username

[WINDOWS_SECONDARY]
analysis = C:/path/to/secondary/analysis
data = C:/path/to/secondary/sorted
raw = C:/path/to/secondary/raw
h5 = C:/path/to/secondary/h5
meta = C:/path/to/secondary/meta
tags = C:/path/to/secondary/tags
query = C:/path/to/secondary/query/analysis
user = your_username
EOF

  cat <<EOF
Created $config_path.
IMPORTANT: edit this file and replace placeholder paths with real data paths before using data-loading or database-backed workflows.
EOF
}

install_uv() {
  command -v uv >/dev/null 2>&1 || fail "uv not found on PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/"

  info "Creating/updating repo-local .venv with Python $PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" --allow-existing

  local existing_python
  existing_python="$(.venv/bin/python - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
)"
  check_python_requirement "$existing_python" "$PYTHON_REQUIREMENT"
  check_requested_major_minor_matches "$existing_python" "$PYTHON_VERSION" "repo-local .venv" "Remove .venv or request its Python major/minor version."

  local venv_python="$SCRIPT_DIR/.venv/bin/python"

  info "Installing retinanalysis editable"
  uv pip install --python "$venv_python" -e "$SCRIPT_DIR"

  info "Installing local vision-utils package from submodule"
  uv pip install --python "$venv_python" "$SCRIPT_DIR/lib/artificial-retina-software-pipeline/utilities"

  if [[ "$DEV" -eq 1 ]]; then
    info "Installing development/test dependencies"
    uv pip install --python "$venv_python" "pytest>=9.0.3"
  fi

}

conda_env_exists() {
  conda run -n "$CONDA_ENV" python --version >/dev/null 2>&1
}

conda_env_python_version() {
  conda run -n "$CONDA_ENV" python - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
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

  info "Installing retinanalysis editable into conda env '$CONDA_ENV'"
  conda run -n "$CONDA_ENV" python -m pip install -e "$SCRIPT_DIR"

  info "Installing local vision-utils package from submodule into conda env '$CONDA_ENV'"
  conda run -n "$CONDA_ENV" python -m pip install "$SCRIPT_DIR/lib/artificial-retina-software-pipeline/utilities"

  if [[ "$DEV" -eq 1 ]]; then
    info "Installing development/test dependencies into conda env '$CONDA_ENV'"
    conda run -n "$CONDA_ENV" python -m pip install "pytest>=9.0.3"
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

create_config_template

cat <<'EOF'

Installation finished.

Next steps:
  1. Edit src/retinanalysis/config/config.ini with real data paths for this machine.
  2. Start the Docker/DataJoint database only when you need database-backed workflows.
  3. After editing config.ini, you can test the package import from Python:

       import retinanalysis as ra

  4. After config.ini and the database are ready, populate a fresh database from Python:

       import retinanalysis as ra
       ra.populate_database()

EOF
