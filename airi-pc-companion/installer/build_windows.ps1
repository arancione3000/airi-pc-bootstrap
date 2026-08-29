$ErrorActionPreference='Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root 'build'
$Venv = Join-Path $Build 'venv'
New-Item -ItemType Directory -Force -Path $Build | Out-Null
py -3 -m venv $Venv
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Venv\Scripts\pip.exe" install -r "$PSScriptRoot\requirements-build.txt"
Push-Location $Root
& "$Venv\Scripts\pyinstaller.exe" --noconfirm --clean --windowed --name AiriPC-Companion --add-data "companion;companion" --add-data "game_agent;game_agent" app/main.py
Pop-Location
Write-Host "Portable executable: $Root\dist\AiriPC-Companion\AiriPC-Companion.exe"
