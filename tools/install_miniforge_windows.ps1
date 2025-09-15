# install_miniforge.ps1

# Default Python version
$PY = $env:PY
if ([string]::IsNullOrWhiteSpace($PY)) {
    $PY = "3.12"
}

# Trap (global error handler)
trap {
    Write-Host ""
    Write-Host "#====================================================="
    Write-Host "# Installation failed at line $($MyInvocation.ScriptLineNumber)"
    Write-Host "#====================================================="
    exit 1
}
$ErrorActionPreference = "Stop"

# Find script and parent path
$SCRIPT = $MyInvocation.MyCommand.Definition
$SCRIPTPATH = Split-Path -Parent $SCRIPT
$ANUGA_CORE_PATH = Resolve-Path (Join-Path $SCRIPTPATH "..")

# Check allowed Python version
if ($PY -match '^3\.(1[0-3]|9)$') {
    Write-Host "Requested python version is $PY"
    Write-Host " "
} else {
    Write-Host "Python version must be greater than 3.8 and less than 3.14"
    exit 1
}

Write-Host "#==========================="
Write-Host "# Install miniforge3"
Write-Host "#==========================="

$mf_dir = Join-Path $HOME 'miniforge3'
if (Test-Path $mf_dir) {
    Write-Host "miniforge3 seems to already exist."
} else {
    Write-Host "miniforge3 does not exist."
    $mf_installer = Join-Path $HOME 'Miniforge3.exe'
    if (!(Test-Path $mf_installer)) {
        Write-Host "Miniforge3.exe does not exist. Downloading..."
        Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $mf_installer
    }
    Write-Host "Running Miniforge3.exe installer in silent mode..."
    Start-Process -FilePath $mf_installer -ArgumentList "/InstallationType=JustMe", "/AddToPath=1", "/RegisterPython=0", "/S", "/D=$mf_dir" -Wait
}

Write-Host " "
Write-Host "#==============================================="
Write-Host "# create conda environment anuga_env_${PY}"
Write-Host "#==============================================="
Write-Host "..."

$condaExe = Join-Path $mf_dir "Scripts\conda.exe"
$env_file = Join-Path $ANUGA_CORE_PATH "environments\environment_${PY}.yml"
& $condaExe env create --file $env_file

Write-Host " "
Write-Host "#======================================"
Write-Host "# activate environment anuga_env_${PY}"
Write-Host "#======================================"
Write-Host "..."

& $condaExe run -n anuga_env_${PY} powershell -Command {

    Write-Host "#================================================================"
    Write-Host "# Installing anuga from the $using:ANUGA_CORE_PATH directory"
    Write-Host "#================================================================"
    Write-Host "..."

    Set-Location $using:SCRIPTPATH
    Set-Location ..
    pip install --no-build-isolation --editable .

    Write-Host " "
    Write-Host "#==========================="
    Write-Host "# Run unittests"
    Write-Host "#==========================="
    Write-Host " "

    Set-Location ..
    pytest -q --disable-warnings --pyargs anuga
}

Write-Host " "
Write-Host "#=================================================================="
Write-Host "# Congratulations, Looks like you have successfully installed anuga"
Write-Host "#=================================================================="
Write-Host "#=================================================================="
Write-Host "# To use anuga you must activate the python environment anuga_env_${PY}"
Write-Host "# that has just been created. Run the command"
Write-Host "# "
Write-Host "# $HOME\miniforge3\Scripts\activate.bat anuga_env_${PY}"
Write-Host "# "
Write-Host "# Or use conda activate anuga_env_${PY} if conda is initialized"
Write-Host "#=================================================================="
Write-Host "# NOTE: You can run"
Write-Host "# "
Write-Host "# conda init"
Write-Host "# "
Write-Host "# to enable 'conda activate anuga_env_${PY}' in all new shells"
Write-Host "#=================================================================="
