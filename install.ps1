#Requires -Version 5.1
<#
.SYNOPSIS
    Install retinanalysis on Windows using uv or conda.

.DESCRIPTION
    PowerShell counterpart to install.sh, for Windows users who do not have a bash
    shell handy. The two scripts accept the same modes and the same options and
    perform the same steps; keep them in sync when either one changes.

    Use conda mode from the "Anaconda PowerShell Prompt" shortcut. A plain PowerShell
    window does not have conda on its PATH unless `conda init powershell` has been run.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Mode,

    [string]$Python = '3.11.13',

    [switch]$Dev,

    [switch]$Config,

    [Alias('Env')]
    [string]$EnvName = 'retinanalysis',

    [switch]$Help
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

# $LASTEXITCODE must be initialised, and read, at GLOBAL scope. Assigning a bare
# `$LASTEXITCODE = 0` here would create a script-scoped variable that permanently shadows
# the automatic one for every function defined in this file: native commands update the
# global, the functions keep reading the script-scoped copy, and every exit-code check
# silently sees 0. That failure is invisible -- the install just reports success after a
# command that failed -- so every read below is written as $global:LASTEXITCODE.
$global:LASTEXITCODE = 0

$ScriptDir = $PSScriptRoot

# Must be captured at script scope. Inside a function, $PSBoundParameters describes that
# function's own arguments, not the script's, so this check cannot be made where it is used.
$EnvNameWasPassed = $PSBoundParameters.ContainsKey('EnvName')

# Keep in sync with [dependency-groups] dev in pyproject.toml. Only used as a fallback
# for pip older than 25.1, which cannot read dependency groups (PEP 735) at all.
$DevFallback = 'pytest>=9.0.3'

function Show-Usage {
    Write-Host @'
Usage:
  .\install.ps1 uv [-Python VERSION] [-Dev] [-Config]
  .\install.ps1 conda [-Python VERSION] [-Dev] [-Config] [-Env NAME]

Modes:
  uv       Create/use the repo-local .venv and install retinanalysis with uv.
  conda    Create/use a conda environment and install retinanalysis with pip.

Options:
  -Python VERSION   Python version to request. Default: 3.11.13.
                    Must satisfy pyproject.toml requires-python.
  -Dev              Install development/test dependencies.
  -Env NAME         Conda environment name. Default: retinanalysis.
                    Only valid in conda mode. An existing environment is reused if its
                    Python major/minor matches -Python; otherwise the script stops.
  -Config           Launch config GUI using the setup_gui() method. Do not use
                    if running 'headless' or over SSH. The program attempts to open
                    a window that allows you to browse for all relevant paths.
  -Help             Show this help message.

Notes:
  - Run conda mode from the "Anaconda PowerShell Prompt" shortcut, which puts conda
    on PATH. In a plain PowerShell window, run `conda init powershell` once first.
  - If Windows refuses to run this script, start it as:
      powershell -ExecutionPolicy Bypass -File .\install.ps1 uv
  - This script does not run git submodule commands automatically.
  - This script does not create, populate, migrate, or modify DataJoint databases.
'@
}

function Fail {
    param([Parameter(Mandatory)][string]$Message)
    [Console]::Error.WriteLine("ERROR: $Message")
    exit 1
}

function Info {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# PowerShell does not stop on a failing native executable the way `set -e` does in bash,
# and $ErrorActionPreference has no effect on native exit codes at all -- a failed uv,
# conda or pip call would otherwise sail straight through to the success message. Every
# external command in this script therefore goes through here.
function Invoke-Checked {
    param(
        [Parameter(Mandatory)][string]$What,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    & $Command
    if ($global:LASTEXITCODE -ne 0) {
        Fail "$What failed with exit code $global:LASTEXITCODE."
    }
}

# Runs a command and returns its last line of stdout, trimmed. stderr is deliberately
# left unredirected: in Windows PowerShell 5.1, redirecting a native command's stderr
# wraps each line in an ErrorRecord and, under $ErrorActionPreference = 'Stop', turns
# harmless progress chatter into a fatal NativeCommandError.
function Invoke-Capture {
    param([Parameter(Mandatory)][scriptblock]$Command)
    $output = & $Command
    if ($null -eq $output) { return '' }
    return ((@($output) | Select-Object -Last 1) -as [string]).Trim()
}

function Assert-RepoRoot {
    if (-not (Test-Path (Join-Path $ScriptDir 'pyproject.toml'))) {
        Fail 'pyproject.toml not found. Run this script from the retinanalysis repo root.'
    }
    if (-not (Test-Path (Join-Path $ScriptDir 'src\retinanalysis'))) {
        Fail 'src\retinanalysis not found. Run this script from the retinanalysis repo root.'
    }
}

function Get-PythonRequirement {
    $match = Select-String -Path (Join-Path $ScriptDir 'pyproject.toml') `
                           -Pattern '^\s*requires-python\s*=' | Select-Object -First 1
    if (-not $match -or $match.Line -notmatch '"([^"]+)"') {
        Fail 'could not read requires-python from pyproject.toml'
    }
    return $Matches[1]
}

# Returns @(major, minor). Mirrors parse_major_minor() in install.sh.
function Get-MajorMinor {
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Version)
    if ($Version -notmatch '^(\d+)\.(\d+)(\.\d+)?$') {
        Fail "Python version '$Version' must look like MAJOR.MINOR or MAJOR.MINOR.PATCH"
    }
    return @([int]$Matches[1], [int]$Matches[2])
}

function Assert-PythonRequirement {
    param(
        [Parameter(Mandatory)][AllowEmptyString()][string]$Version,
        [Parameter(Mandatory)][string]$Requirement
    )
    if ($Requirement -notmatch '^>=(\d+)\.(\d+)$') {
        Fail "unsupported requires-python format '$Requirement'; expected a simple '>=MAJOR.MINOR' requirement"
    }
    # Read both out of $Matches before Get-MajorMinor's own -match overwrites it.
    $minMajor = [int]$Matches[1]
    $minMinor = [int]$Matches[2]

    $parsed = Get-MajorMinor $Version
    if ($parsed[0] -lt $minMajor -or ($parsed[0] -eq $minMajor -and $parsed[1] -lt $minMinor)) {
        Fail "requested Python $Version does not satisfy pyproject.toml requires-python '$Requirement'"
    }
}

function Assert-RequestedMajorMinorMatches {
    param(
        [Parameter(Mandatory)][string]$Actual,
        [Parameter(Mandatory)][string]$Requested,
        [Parameter(Mandatory)][string]$Context,
        [Parameter(Mandatory)][string]$Advice
    )
    $actualParsed = Get-MajorMinor $Actual
    $requestedParsed = Get-MajorMinor $Requested
    if ($actualParsed[0] -ne $requestedParsed[0] -or $actualParsed[1] -ne $requestedParsed[1]) {
        Fail "$Context uses Python $Actual, but requested Python $Requested. $Advice"
    }
}

function Assert-Submodule {
    $dir = Join-Path $ScriptDir 'lib\artificial-retina-software-pipeline\utilities'
    $present = (Test-Path $dir) -and
               ((Test-Path (Join-Path $dir 'pyproject.toml')) -or (Test-Path (Join-Path $dir 'setup.py')))
    if (-not $present) {
        [Console]::Error.WriteLine(@'
ERROR: required submodule package is missing or incomplete:
  lib\artificial-retina-software-pipeline\utilities

Run this command yourself, then rerun install.ps1:
  git submodule update --init --recursive
'@)
        exit 1
    }
}

function Install-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Fail 'uv not found on PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/'
    }

    $venvPython = Join-Path $ScriptDir '.venv\Scripts\python.exe'

    # Validate an existing .venv BEFORE uv venv runs. `uv venv --allow-existing` keeps
    # the directory but rebuilds the interpreter at whatever -Python asks for, so
    # checking afterwards can never fail: the probe would only ever report the version
    # we just requested, and a working environment would already have been replaced.
    if (Test-Path $venvPython) {
        $existingPython = Invoke-Capture { & $venvPython -c "import platform; print(platform.python_version())" }
        Assert-PythonRequirement $existingPython $script:PythonRequirement
        Assert-RequestedMajorMinorMatches $existingPython $Python 'repo-local .venv' `
            'Remove .venv or request its Python major/minor version.'
    }

    Info "Creating/updating repo-local .venv with Python $Python"
    Invoke-Checked 'uv venv' { uv venv --python $Python --allow-existing }

    if (-not (Test-Path $venvPython)) {
        Fail "expected interpreter not found at $venvPython"
    }

    # vision-utils first, matching install.sh. uv reads [tool.uv.sources] so the order is
    # not strictly required here, but keeping both backends in the same order means one
    # less thing that is true on only one path.
    Info 'Installing local vision-utils package from submodule'
    $visionUtils = Join-Path $ScriptDir 'lib\artificial-retina-software-pipeline\utilities'
    Invoke-Checked 'uv pip install (vision-utils)' { uv pip install --python $venvPython $visionUtils }

    Info 'Installing retinanalysis editable'
    Invoke-Checked 'uv pip install -e (retinanalysis)' { uv pip install --python $venvPython -e $ScriptDir }

    if ($Dev) {
        Info 'Installing development/test dependencies'
        # Resolved from [dependency-groups] dev in pyproject.toml, relative to the
        # current directory, which Invoke-Main has already set to the repo root.
        Invoke-Checked 'uv pip install --group dev' { uv pip install --python $venvPython --group dev }
    }

    if ($Config) {
        Info 'Launching config setup GUI'
        Invoke-Checked 'config setup GUI' {
            & $venvPython -c "import retinanalysis as ra; ra.config.setup_gui()"
        }
    }
}

# Asking conda for the environment list is preferable to probing with `conda run`: it
# does not execute anything in the environment, and it does not print a scary
# EnvironmentLocationNotFound to the console on the (normal) first-install path.
function Test-CondaEnvExists {
    param([Parameter(Mandatory)][string]$Name)
    # `conda env list --json` prints a multi-line document, so every line has to be
    # joined back together. Invoke-Capture deliberately returns only the final line,
    # which is right for the single-value version probes but would truncate JSON.
    $output = conda env list --json
    if ($global:LASTEXITCODE -ne 0 -or -not $output) {
        Fail 'could not list conda environments. Is conda working in this shell?'
    }
    $parsed = (@($output) -join "`n") | ConvertFrom-Json
    foreach ($path in $parsed.envs) {
        if ((Split-Path $path -Leaf) -eq $Name) { return $true }
    }
    return $false
}

function Get-CondaEnvPythonVersion {
    param([Parameter(Mandatory)][string]$Name)
    # A `python -c` one-liner, never a heredoc piped into `python -`: on Windows
    # `conda run` does not forward stdin to the child process, so the heredoc form
    # returns an empty string and every downstream version check fails. The Python
    # source also avoids double quotes, because PowerShell strips embedded double
    # quotes when forwarding arguments to a native command.
    return Invoke-Capture {
        conda run -n $Name python -c "import platform; print(platform.python_version())"
    }
}

# pip only learned to read [dependency-groups] (PEP 735) in 25.1, and conda environments
# often carry whatever pip ensurepip bundled, which can be far older.
function Test-CondaPipSupportsDependencyGroups {
    param([Parameter(Mandatory)][string]$Name)
    $pipVersion = Invoke-Capture { conda run -n $Name python -c "import pip; print(pip.__version__)" }
    if ($global:LASTEXITCODE -ne 0 -or $pipVersion -notmatch '^(\d+)\.(\d+)') { return $false }
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    return ($major -gt 25 -or ($major -eq 25 -and $minor -ge 1))
}

function Install-Conda {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        Fail @'
conda not found in this shell.

Open the "Anaconda PowerShell Prompt" shortcut from the Start menu and run this script
from there, or run `conda init powershell` once and reopen PowerShell. Alternatively,
install with uv instead: .\install.ps1 uv
'@
    }

    if (Test-CondaEnvExists $EnvName) {
        $existingPython = Get-CondaEnvPythonVersion $EnvName
        Assert-PythonRequirement $existingPython $script:PythonRequirement
        Assert-RequestedMajorMinorMatches $existingPython $Python "existing conda env '$EnvName'" `
            'Use a different -Env or update the environment yourself.'
        Info "Using existing conda environment '$EnvName' with Python $existingPython"
    }
    else {
        Info "Creating conda environment '$EnvName' with Python $Python"
        Invoke-Checked 'conda create' { conda create -y -n $EnvName "python=$Python" }
    }

    # vision-utils MUST be installed before retinanalysis here. It is declared in
    # [project.dependencies] but is resolvable only through [tool.uv.sources], which is a
    # uv-only table -- pip ignores everything under [tool] that is not its own. So pip
    # treats "vision-utils" as an ordinary requirement, looks for it on PyPI, and fails,
    # because it is not published there. Installing it first leaves the requirement
    # already satisfied by an installed distribution, so pip never queries an index.
    #
    # --no-capture-output on every long step: without it `conda run` buffers the child's
    # output until the command exits, so multi-minute pip installs and the config GUI
    # look like the script has hung.
    Info "Installing local vision-utils package from submodule into conda env '$EnvName'"
    $visionUtils = Join-Path $ScriptDir 'lib\artificial-retina-software-pipeline\utilities'
    Invoke-Checked 'pip install (vision-utils)' {
        conda run --no-capture-output -n $EnvName python -m pip install $visionUtils
    }

    Info "Installing retinanalysis editable into conda env '$EnvName'"
    Invoke-Checked 'pip install -e (retinanalysis)' {
        conda run --no-capture-output -n $EnvName python -m pip install -e $ScriptDir
    }

    if ($Dev) {
        Info "Installing development/test dependencies into conda env '$EnvName'"
        if (Test-CondaPipSupportsDependencyGroups $EnvName) {
            Invoke-Checked 'pip install --group dev' {
                conda run --no-capture-output -n $EnvName python -m pip install --group dev
            }
        }
        else {
            Info "pip in '$EnvName' predates PEP 735; installing dev packages by name instead"
            Invoke-Checked 'pip install (dev fallback)' {
                conda run --no-capture-output -n $EnvName python -m pip install $DevFallback
            }
        }
    }

    if ($Config) {
        Info 'Launching config setup GUI'
        Invoke-Checked 'config setup GUI' {
            conda run --no-capture-output -n $EnvName python -c "import retinanalysis as ra; ra.config.setup_gui()"
        }
    }
}

function Show-NextSteps {
    Write-Host @'

Installation finished.

Next steps:
  1. Import the package to make sure everything installed correctly:

       import retinanalysis as ra

  2. If you did not use the -Config flag, you must create a config file that
     contains the paths to the relevant directories:

       Option 1, use the CLI setup tool:
            ra.config.setup()

       Option 2, use the GUI setup tool:
            ra.config.setup_gui()

       Option 3, write your config file manually:
            See src\retinanalysis\assets\config_template.toml
            The directory where you should put this file is printed when you
            first import retinanalysis

  3. Start the Docker/DataJoint database only when you need database-backed workflows.

  4. After config file and the database are ready, populate a fresh database from Python:

       import retinanalysis as ra
       ra.populate_database()

'@
}

function Invoke-Main {
    if ($Help) {
        Show-Usage
        exit 0
    }

    if (-not $Mode) {
        Show-Usage
        exit 1
    }

    if ($Mode -notin @('uv', 'conda')) {
        Show-Usage
        Fail 'first argument must be one of: uv, conda'
    }

    if ($Mode -ne 'conda' -and $EnvNameWasPassed) {
        Fail '-Env is only valid in conda mode'
    }

    Assert-RepoRoot

    # uv --group dev and pip --group dev both resolve pyproject.toml from the current
    # directory, and the conda pip install steps inherit it too, so pin the location.
    Set-Location $ScriptDir

    $script:PythonRequirement = Get-PythonRequirement
    Assert-PythonRequirement $Python $script:PythonRequirement
    Assert-Submodule

    Info "Install mode: $Mode"
    Write-Host "Python request: $Python (project requires $script:PythonRequirement)"
    if ($Mode -eq 'conda') {
        Write-Host "Conda environment: $EnvName"
    }

    switch ($Mode) {
        'uv'    { Install-Uv }
        'conda' { Install-Conda }
    }

    Show-NextSteps
}

Invoke-Main
