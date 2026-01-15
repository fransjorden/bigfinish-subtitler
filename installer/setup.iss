; Big Finish Subtitler - Inno Setup Script
; Creates a Windows installer that downloads dependencies during setup

#define MyAppName "Big Finish Subtitler"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Big Finish Subtitler"
#define MyAppURL "https://github.com/bigfinish-subtitler"
#define MyAppExeName "BigFinishSubtitler.exe"

[Setup]
AppId={{B1GF1N1SH-SUBT-1TL3-R000-000000000000}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir=output
OutputBaseFilename=BigFinishSubtitler-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
; Require internet connection
; Show custom install progress

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Application files
Source: "..\webapp\*"; DestDir: "{app}\webapp"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\parsed_scripts\*.enc"; DestDir: "{app}\parsed_scripts"; Flags: ignoreversion
Source: "..\*.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Installer scripts
Source: "detect_gpu.py"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "install_deps.py"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "launcher.pyw"; DestDir: "{app}"; Flags: ignoreversion

; Embedded Python (download separately, see BUILD.md)
Source: "python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Run dependency installer after file copy
Filename: "{app}\python\python.exe"; Parameters: """{app}\installer\install_deps.py"" --install-dir ""{app}"" {code:GetGPUFlag}"; StatusMsg: "Installing dependencies (this may take several minutes)..."; Flags: runhidden

; Optionally launch app after install
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
var
  GPUPage: TInputOptionWizardPage;
  GPUDetected: Boolean;
  GPUInfo: String;
  UseGPU: Boolean;

// Detect GPU during install
procedure DetectGPU;
var
  ResultCode: Integer;
  TmpFile: String;
  Output: AnsiString;
begin
  GPUDetected := False;
  GPUInfo := 'No NVIDIA GPU detected';

  // We'll do simple nvidia-smi check
  if Exec('nvidia-smi', '--query-gpu=name --format=csv,noheader', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
    begin
      GPUDetected := True;
      GPUInfo := 'NVIDIA GPU detected - GPU acceleration available';
    end;
  end;
end;

procedure InitializeWizard;
begin
  // Detect GPU
  DetectGPU;

  // Create GPU selection page
  GPUPage := CreateInputOptionPage(wpSelectDir,
    'Processing Mode',
    'Choose how audio will be processed',
    'The installer detected your system configuration:' + #13#10 + #13#10 +
    GPUInfo + #13#10 + #13#10 +
    'Select your preferred processing mode:',
    True, False);

  if GPUDetected then
  begin
    GPUPage.Add('GPU Acceleration (Recommended) - Faster processing using your NVIDIA graphics card');
    GPUPage.Add('CPU Only - Slower but works on all systems');
    GPUPage.SelectedValueIndex := 0;  // Default to GPU
  end
  else
  begin
    GPUPage.Add('CPU Processing - Standard processing mode');
    GPUPage.Add('GPU Acceleration (Not recommended - no compatible GPU detected)');
    GPUPage.SelectedValueIndex := 0;  // Default to CPU
  end;
end;

function GetGPUFlag(Param: String): String;
begin
  if GPUDetected and (GPUPage.SelectedValueIndex = 0) then
    Result := '--gpu'
  else
    Result := '--cpu';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  // Store GPU selection when leaving that page
  if CurPageID = GPUPage.ID then
  begin
    UseGPU := GPUDetected and (GPUPage.SelectedValueIndex = 0);
  end;
end;

// Show download progress (for future use with custom download page)
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Installation complete
  end;
end;
