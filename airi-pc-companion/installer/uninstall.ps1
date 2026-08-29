$ErrorActionPreference='Stop'
$Root = if ($env:AIRIPC_INSTALL_DIR) { $env:AIRIPC_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'AiriPC-Companion' }
Get-Process | Where-Object { $_.ProcessName -eq 'AiriPC-Companion' } | Stop-Process -Force -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $Root -ErrorAction SilentlyContinue
Write-Host 'Airi-PC Companion removed. Pairing data and runtime state were removed with the local app directory.'
