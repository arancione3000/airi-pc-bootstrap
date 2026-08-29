$ErrorActionPreference='SilentlyContinue'
$Root = if ($env:AIRIPC_INSTALL_DIR) { $env:AIRIPC_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'AiriPC-Companion' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*$Root*companion.cli*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Remove-Item -Recurse -Force $Root
Write-Host 'Airi-PC Companion removed.'
