# Build a portable onedir krabobot-voice for Windows (PyInstaller).
#
# From repo root (recommended):
#   .\.venv\Scripts\Activate.ps1
#   .\clients\krabobot-voice\scripts\build_portable.ps1
#
# Output:
#   clients\krabobot-voice\build\krabobot-voice\krabobot-voice.exe
#   (+ _internal\, config.example.yaml, README.md)

[CmdletBinding()]
param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ClientRoot = Resolve-Path (Join-Path $ScriptDir "..")
$Spec = Join-Path $ClientRoot "packaging\krabobot-voice.spec"
$WorkPath = Join-Path $ClientRoot "build\pyi-work"
$DistPath = Join-Path $ClientRoot "build"
$OutDir = Join-Path $DistPath "krabobot-voice"
$ExampleCfg = Join-Path $ClientRoot "config.example.yaml"
$PackagingReadme = Join-Path $ClientRoot "packaging\README.md"

if (-not (Test-Path -LiteralPath $Spec)) {
    throw "Spec not found: $Spec"
}

Write-Host "==> Client root: $ClientRoot"
Write-Host "==> Spec:        $Spec"
Write-Host "==> Dist path:   $DistPath"

if (-not $SkipInstall) {
    Write-Host "==> Ensuring packaging deps (asr + pyinstaller)..."
    python -m pip install -e "$ClientRoot[asr,packaging]"
}

$pyi = Get-Command pyinstaller -ErrorAction SilentlyContinue
if (-not $pyi) {
    throw "pyinstaller not on PATH. Activate venv and install: pip install -e `".\clients\krabobot-voice[packaging]`""
}

New-Item -ItemType Directory -Force -Path $WorkPath | Out-Null
New-Item -ItemType Directory -Force -Path $DistPath | Out-Null

Write-Host "==> Running PyInstaller (onedir)..."
& pyinstaller `
    --noconfirm `
    --clean `
    --workpath $WorkPath `
    --distpath $DistPath `
    $Spec

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath (Join-Path $OutDir "krabobot-voice.exe"))) {
    throw "Expected exe missing: $OutDir\krabobot-voice.exe"
}

# Ensure runtime docs/example sit beside the exe (also collected via datas).
Copy-Item -LiteralPath $ExampleCfg -Destination (Join-Path $OutDir "config.example.yaml") -Force
Copy-Item -LiteralPath $PackagingReadme -Destination (Join-Path $OutDir "README.md") -Force

Write-Host ""
Write-Host "OK: portable build ready:"
Write-Host "  $OutDir\krabobot-voice.exe"
Write-Host ""
Write-Host "Next: edit config.yaml next to the exe (auto-created from config.example.yaml on first run),"
Write-Host "      start ``krabobot serve``, then run the exe."
Write-Host "See: $OutDir\README.md"
