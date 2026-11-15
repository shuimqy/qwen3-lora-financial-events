$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$pythonPath = Join-Path $serviceRoot '.venv\Scripts\python.exe'
$runDirectory = Join-Path $serviceRoot 'run'
$statePath = Join-Path $runDirectory 'server.json'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Run scripts\setup.ps1 first.'
}
if (Test-Path -LiteralPath $statePath) {
    $oldState = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $oldProcess = Get-Process -Id $oldState.pid -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.StartTime.ToUniversalTime().Ticks.ToString() -eq $oldState.start_ticks) {
        Write-Output "Model service is already running (PID $($oldState.pid))."
        exit 0
    }
}
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$serviceProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'model_service') `
    -WorkingDirectory $serviceRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $runDirectory 'stderr.log')
Start-Sleep -Milliseconds 300
$serviceProcess.Refresh()
if ($serviceProcess.HasExited) { throw 'Service exited; inspect run\stderr.log.' }
$state = [pscustomobject]@{
    pid = $serviceProcess.Id
    start_ticks = $serviceProcess.StartTime.ToUniversalTime().Ticks.ToString()
    executable = $pythonPath
}
[System.IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
Write-Output "Started model service (PID $($serviceProcess.Id)); readiness: http://127.0.0.1:8001/health/ready"
