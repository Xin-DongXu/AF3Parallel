# Upload built distributions to PyPI (Windows).
#
#   $env:TWINE_USERNAME = "__token__"
#   $env:TWINE_PASSWORD = "pypi-..."
#   .\tools\publish-pypi.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

python -m pip install --upgrade build twine
python -m build
twine check dist/*

if (-not $env:TWINE_PASSWORD) {
    Write-Error "Set TWINE_USERNAME=__token__ and TWINE_PASSWORD (PyPI API token). See docs/publishing.md"
}

twine upload dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Error "twine upload failed (exit code $LASTEXITCODE). Check TWINE_USERNAME and TWINE_PASSWORD."
}

Write-Host ""
Write-Host "Published. Verify: pip install af3parallel; af3parallel --version"
Get-ChildItem dist\*.tar.gz | ForEach-Object {
    $hash = Get-FileHash $_.FullName -Algorithm SHA256
    Write-Host "sdist sha256 ($($_.Name)): $($hash.Hash.ToLower())"
}
