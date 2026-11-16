$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).ProviderPath
foreach ($serviceName in @('frontend', 'backend', 'model_service')) {
    $stopScript = Join-Path $projectRoot "$serviceName\scripts\stop.ps1"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $stopScript
    if ($LASTEXITCODE -ne 0) { throw "Could not stop $serviceName" }
}
