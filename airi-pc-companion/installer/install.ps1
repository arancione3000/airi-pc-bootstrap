$ErrorActionPreference='Stop'
$Root = if ($env:AIRIPC_INSTALL_DIR) { $env:AIRIPC_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'AiriPC-Companion' }
$Source = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$Root\app" | Out-Null
Copy-Item "$Source\companion" "$Root\app" -Recurse -Force
Copy-Item "$Source\app" "$Root\app" -Recurse -Force
Copy-Item "$Source\game_agent" "$Root\app" -Recurse -Force
Copy-Item "$Source\requirements.txt" "$Root\requirements.txt" -Force
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Python 3 is required for developer-mode install. Use the packaged EXE for one-click end-user install.' }
py -3 -m venv "$Root\venv"
& "$Root\venv\Scripts\python.exe" -m pip install -r "$Root\requirements.txt"
$Run = "Set-Location `"$Root\app`"; & `"$Root\venv\Scripts\python.exe`" -m app.main"
Set-Content -Path "$Root\run.ps1" -Value $Run
Write-Host "Developer install complete: $Root"
