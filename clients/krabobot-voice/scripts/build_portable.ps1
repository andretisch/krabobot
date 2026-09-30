# Build a portable onedir krabobot-voice for Windows (PyInstaller).
#
# From repo root (recommended):
#   .\.venv\Scripts\Activate.ps1
#   pip install -e ".\clients\krabobot-voice[asr,ui,packaging]"
#   .\clients\krabobot-voice\scripts\build_portable.ps1
#
# User-facing output (self-contained — VAD + STT bundled, no first-run download):
#   clients\krabobot-voice\build\krabobot-voice\krabobot-voice.exe
# Docs: clients\krabobot-voice\README.md (Portable build) and packaging\README.md.
# Windowed exe (console=False); tray+UI in-process. Dev debug: python -m … --console.

[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [string]$SileroSource = "",
    [string]$SttSource = ""
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
$PreferredSttName = "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16"

function Resolve-SileroSource {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path -LiteralPath $Explicit)) {
        return (Resolve-Path -LiteralPath $Explicit).Path
    }
    $candidates = @(
        (Join-Path $ClientRoot "models\silero_vad.onnx"),
        (Join-Path $env:LOCALAPPDATA "krabobot-voice\models\silero_vad.onnx")
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path -LiteralPath $c)) {
            return (Resolve-Path -LiteralPath $c).Path
        }
    }
    return $null
}

function Resolve-SttSource {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path -LiteralPath $Explicit)) {
        $p = (Resolve-Path -LiteralPath $Explicit).Path
        if (-not (Test-Path -LiteralPath (Join-Path $p "tokens.txt"))) {
            throw "STT source missing tokens.txt: $p"
        }
        return $p
    }
    $candidates = @(
        (Join-Path $ClientRoot "models\stt\$PreferredSttName"),
        (Join-Path $env:USERPROFILE ".krabobot\models\stt\$PreferredSttName")
    )
    foreach ($c in $candidates) {
        if ((Test-Path -LiteralPath $c) -and (Test-Path -LiteralPath (Join-Path $c "tokens.txt"))) {
            return (Resolve-Path -LiteralPath $c).Path
        }
    }
    $legacyBase = Join-Path $env:USERPROFILE ".krabobot\models\stt"
    if (Test-Path -LiteralPath $legacyBase) {
        $hit = Get-ChildItem -LiteralPath $legacyBase -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "tokens.txt") } |
            Sort-Object Name |
            Select-Object -First 1
        if ($hit) {
            return $hit.FullName
        }
    }
    return $null
}

if (-not (Test-Path -LiteralPath $Spec)) {
    throw "Spec not found: $Spec"
}

Write-Host "==> Client root: $ClientRoot"
Write-Host "==> Spec:        $Spec"
Write-Host "==> Dist path:   $DistPath"

$sileroSrc = Resolve-SileroSource -Explicit $SileroSource
$sttSrc = Resolve-SttSource -Explicit $SttSource
if (-not $sileroSrc) {
    throw @"
Silero VAD ONNX not found.
Place silero_vad.onnx under clients\krabobot-voice\models\ or
%LOCALAPPDATA%\krabobot-voice\models\, or pass -SileroSource <path>.
"@
}
if (-not $sttSrc) {
    throw @"
Sherpa STT model dir not found (needs tokens.txt).
Expected under clients\krabobot-voice\models\stt\$PreferredSttName or
~/.krabobot/models/stt/..., or pass -SttSource <dir>.
"@
}
Write-Host "==> Silero source: $sileroSrc"
Write-Host "==> STT source:    $sttSrc"

if (-not $SkipInstall) {
    Write-Host "==> Ensuring packaging deps (asr + pyinstaller)..."
    python -m pip install -e "$ClientRoot[asr,ui,packaging]"
}

$pyi = Get-Command pyinstaller -ErrorAction SilentlyContinue
if (-not $pyi) {
    throw "pyinstaller not on PATH. Activate venv and install: pip install -e `".\clients\krabobot-voice[packaging]`""
}

New-Item -ItemType Directory -Force -Path $WorkPath | Out-Null
New-Item -ItemType Directory -Force -Path $DistPath | Out-Null

# PyInstaller --noconfirm replaces build\krabobot-voice. Keep config.yaml across that wipe.
$preservedCfg = $null
$existingCfg = Join-Path $OutDir "config.yaml"
if (Test-Path -LiteralPath $existingCfg) {
    $preservedCfg = Join-Path $env:TEMP ("krabobot-voice-config-{0}.yaml" -f [guid]::NewGuid().ToString("N"))
    Copy-Item -LiteralPath $existingCfg -Destination $preservedCfg -Force
    Write-Host "==> Preserving existing config.yaml"
}

try {
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

    # Bundle models beside the exe — fully self-contained, no first-run download.
    $ModelsOut = Join-Path $OutDir "models"
    $SttOutName = Split-Path -Leaf $sttSrc
    $SttOut = Join-Path $ModelsOut "stt\$SttOutName"
    Write-Host "==> Copying Silero → $ModelsOut\silero_vad.onnx"
    New-Item -ItemType Directory -Force -Path $ModelsOut | Out-Null
    Copy-Item -LiteralPath $sileroSrc -Destination (Join-Path $ModelsOut "silero_vad.onnx") -Force
    Write-Host "==> Copying STT → $SttOut"
    if (Test-Path -LiteralPath $SttOut) {
        Remove-Item -LiteralPath $SttOut -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SttOut) | Out-Null
    Copy-Item -LiteralPath $sttSrc -Destination $SttOut -Recurse -Force

    if (-not (Test-Path -LiteralPath (Join-Path $ModelsOut "silero_vad.onnx"))) {
        throw "Failed to copy silero_vad.onnx into $ModelsOut"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SttOut "tokens.txt"))) {
        throw "Failed to copy STT model (tokens.txt missing) into $SttOut"
    }
} finally {
    if ($preservedCfg -and (Test-Path -LiteralPath $preservedCfg)) {
        if (Test-Path -LiteralPath $OutDir) {
            Copy-Item -LiteralPath $preservedCfg -Destination (Join-Path $OutDir "config.yaml") -Force
            Remove-Item -LiteralPath $preservedCfg -Force -ErrorAction SilentlyContinue
            Write-Host "==> Restored existing config.yaml"
        } else {
            Write-Host "==> Output dir missing; kept config.yaml at $preservedCfg"
        }
    }
}

Write-Host ""
Write-Host "OK: portable build ready (self-contained models):"
Write-Host "  $OutDir\krabobot-voice.exe"
Write-Host "  $OutDir\models\silero_vad.onnx"
Write-Host "  $OutDir\models\stt\$SttOutName"
Write-Host ""
Write-Host "Next: edit config.yaml next to the exe (auto-created from config.example.yaml on first run),"
Write-Host "      start ``krabobot serve``, then run the exe."
Write-Host "      VAD/STT work offline from models/ - no download."
Write-Host "See: $OutDir\README.md"
