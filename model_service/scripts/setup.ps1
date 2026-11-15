$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$env:UV_CACHE_DIR = Join-Path $serviceRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $serviceRoot '.python'
Push-Location -LiteralPath $serviceRoot
try {
    uv python install 3.11 --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw 'uv Python installation failed' }
    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { throw 'uv environment synchronization failed' }
} finally {
    Pop-Location
}
