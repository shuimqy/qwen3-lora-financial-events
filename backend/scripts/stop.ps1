$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$statePath = Join-Path $serviceRoot 'run\server.json'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'No managed server recorded.'; exit 0 }
$state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
$serviceProcess = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
if ($serviceProcess) {
    if ($serviceProcess.StartTime.ToUniversalTime().Ticks.ToString() -ne $state.start_ticks) {
        throw 'PID was reused; refusing to stop an unrelated process.'
    }
    if (-not [string]::Equals($serviceProcess.Path, $state.executable, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Executable differs; refusing to stop an unrelated process.'
    }
    # Windows venv Python may start a child interpreter; stop its process tree too.
    & taskkill.exe /PID $serviceProcess.Id /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Failed to stop backend process tree.' }
    Write-Output "Stopped backend (PID $($serviceProcess.Id))."
}
Remove-Item -LiteralPath $statePath -Force
