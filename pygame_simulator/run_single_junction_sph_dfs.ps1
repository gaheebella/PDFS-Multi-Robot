param(
    [ValidateRange(1, 5000)]
    [int]$RobotCount = 680
)

$ErrorActionPreference = "Stop"
$SimulatorDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualEnvironment = Join-Path $SimulatorDirectory ".venv-sph-dfs"
$PythonExecutable = Join-Path $VirtualEnvironment "Scripts\python.exe"
$RequirementsFile = Join-Path $SimulatorDirectory "requirements-sph-dfs.txt"
$SimulatorFile = Join-Path $SimulatorDirectory "single_junction_sph_dfs_environment.py"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    python -m venv $VirtualEnvironment
}

& $PythonExecutable -m pip install --disable-pip-version-check -r $RequirementsFile
$env:SPH_DFS_ROBOT_COUNT = $RobotCount.ToString()

Push-Location $SimulatorDirectory
try {
    & $PythonExecutable $SimulatorFile
}
finally {
    Pop-Location
}
