$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot ".env"
$modelPath = Join-Path $projectRoot "models\oil_unet.pt"
$docker = Get-Command docker -ErrorAction SilentlyContinue

$credentials = $false
if (Test-Path $envPath) {
    $clientId = Select-String -Path $envPath -Pattern '^CDSE_CLIENT_ID=.+$'
    $clientSecret = Select-String -Path $envPath -Pattern '^CDSE_CLIENT_SECRET=.+$'
    $credentials = [bool]($clientId -and $clientSecret)
}

[ordered]@{
    docker = [bool]$docker
    env_file = Test-Path $envPath
    copernicus_credentials = $credentials
    segmentation_checkpoint = Test-Path $modelPath
    ready_to_start_stack = [bool]($docker -and $credentials -and (Test-Path $modelPath))
} | ConvertTo-Json
