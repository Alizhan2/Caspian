# Run this file from PowerShell opened with "Run as administrator".
$ErrorActionPreference = "Stop"
$logPath = Join-Path $PSScriptRoot "wsl2-admin-result.log"
Start-Transcript -Path $logPath -Force | Out-Null

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Stop-Transcript | Out-Null
    throw "Open PowerShell as Administrator and run this script again."
}

dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
if ($LASTEXITCODE -notin 0, 3010) { throw "Could not enable Windows Subsystem for Linux." }

dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
if ($LASTEXITCODE -notin 0, 3010) { throw "Could not enable Virtual Machine Platform." }

$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
$vmFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
@($wslFeature, $vmFeature) | Select-Object FeatureName, State | Format-Table -AutoSize

Write-Host "WSL2 prerequisites are enabled. Restart Windows before starting Docker Desktop." -ForegroundColor Green
Stop-Transcript | Out-Null
