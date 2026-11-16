$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).ProviderPath
foreach ($serviceName in @('model_service', 'backend', 'frontend')) {
    $startScript = Join-Path $projectRoot "$serviceName\scripts\start.ps1"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $startScript
    if ($LASTEXITCODE -ne 0) { throw "Could not start $serviceName" }
}
Write-Output 'Open http://127.0.0.1:3000. The status indicator shows availability when the model is ready.'
