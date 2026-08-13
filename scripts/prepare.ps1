$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
python -m pgllmkt.cli doctor
python -m pgllmkt.cli prepare @args
