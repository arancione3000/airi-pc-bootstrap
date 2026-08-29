$ErrorActionPreference='Stop'
$Root = if ($env:AIRIPC_INSTALL_DIR) { $env:AIRIPC_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'AiriPC-Companion' }
New-Item -ItemType Directory -Force -Path "$Root\app" | Out-Null
$Source = Split-Path -Parent $PSScriptRoot
Copy-Item "$Source\companion" "$Root\app" -Recurse -Force
Copy-Item "$Source\requirements.txt" "$Root\requirements.txt" -Force
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python 3 is required' }
python -m venv "$Root\venv"
& "$Root\venv\Scripts\python.exe" -m pip install --upgrade pip | Out-Null
& "$Root\venv\Scripts\pip.exe" install -r "$Root\requirements.txt"
$Run = "`"$Root\venv\Scripts\python.exe`" -m companion.cli"
Set-Content -Path "$Root\run.ps1" -Value "cd `"$Root\app`"; $Run"
Write-Host "Installed to $Root"
Write-Host "Start with: powershell -ExecutionPolicy Bypass -File `"$Root\run.ps1`""
