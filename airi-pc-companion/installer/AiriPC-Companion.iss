#define AppName "Airi-PC Companion"
#define AppVersion "0.2.0"
#define ExeName "AiriPC-Companion.exe"
[Setup]
AppId={{B2A91C6B-5B82-45C3-9E6B-8C0A53A6D5D1}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\AiriPC-Companion
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=AiriPC-Companion-Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern
[Files]
Source: "..\dist\AiriPC-Companion\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
[Icons]
Name: "{userdesktop}\Airi-PC Companion"; Filename: "{app}\{#ExeName}"
Name: "{userstartmenu}\Airi-PC Companion"; Filename: "{app}\{#ExeName}"
[Run]
Filename: "{app}\{#ExeName}"; Description: "Launch Airi-PC Companion"; Flags: nowait postinstall skipifsilent
[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\AiriPC-Companion"
