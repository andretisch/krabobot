#Requires -Version 5.1
<#
.SYNOPSIS
  Install krabobot from a git checkout (Windows): venv + editable install.

.EXAMPLE
  .\scripts\install.ps1
  .\scripts\install.ps1 -SkipOnboard -Extras "dev,api"
#>
param(
    [string] $VenvDir = ".venv",
    [string] $Extras = "dev,api",
    [switch] $SkipVenv,
    [switch] $SkipOnboard,
    [switch] $Yes
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Test-Python311 {
    $null = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) { return $false }
    $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $parts = $ver.Split(".")
    $maj = [int]$parts[0]
    $min = [int]$parts[1]
    return ($maj -gt 3) -or (($maj -eq 3) -and ($min -ge 11))
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python не найден в PATH. Установите Python 3.11+."
}

if (-not (Test-Python311)) {
    Write-Error "Нужен Python 3.11+. Текущая версия: $(python --version)"
}

$pip = @("python", "-m", "pip")
if (-not $SkipVenv) {
    $venvAbs = Join-Path $Root $VenvDir
    Write-Host "[install] venv: $venvAbs"
    if (-not (Test-Path $venvAbs)) {
        python -m venv $venvAbs
    }
    $activate = Join-Path $venvAbs "Scripts\Activate.ps1"
    . $activate
    $pip = @("python", "-m", "pip")
    Write-Host "[install] Активация в новой сессии: . '$activate'"
}

Write-Host "[install] pip upgrade…"
& $pip[0] $pip[1] $pip[2] install --upgrade pip

if ($Extras -and $Extras.Trim().Length -gt 0) {
    $editable = ".[$Extras]"
} else {
    $editable = "."
}

Write-Host "[install] pip install -e $editable"
& $pip[0] $pip[1] $pip[2] install -e $editable

Write-Host "[install] Готово."
if (Get-Command krabobot -ErrorAction SilentlyContinue) {
    krabobot --version
}

if (-not $SkipOnboard -and -not $Yes) {
    $r = Read-Host "Запустить krabobot onboard сейчас? [y/N]"
    if ($r -match '^(y|Y|yes|да|Да)$') {
        krabobot onboard
    }
}
