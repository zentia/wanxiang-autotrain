<#
.SYNOPSIS
    Windows / PowerShell 5.1 verification driver for wanxiang-autotrain M1 + M1.5.

.DESCRIPTION
    CPU-only and idempotent. Default run does:
      1. create/reuse a Python 3.12 venv under .venv
      2. install CPU torch + the editable package (skipped if already present)
      3. verify the frozen eval corpora byte-for-byte
      4. run the full pytest suite
      5. exercise the M1.5 export/load contract with a step-0 baseline bundle

    With -WithTraining it additionally runs the complete M1.5 A/B:
      structured corpus -> short CPU train -> export -> two eval runs -> compare.
    The training config is COPIED into runs/windows-verify/ with a smaller
    max_steps; no tracked repository file is modified. max_steps is excluded
    from config_hash by design, so shortening it is legitimate.

    This script never commits, pushes, or contacts any remote repository.
    Artifacts land in runs/windows-verify/ (ignored by .gitignore).

.PARAMETER Python
    Python 3.12 interpreter used to create the venv. Override if the default
    uv-managed path is not present on this machine.

.PARAMETER WithTraining
    Also run the end-to-end M1.5 A/B. Slower: 13.86M params on CPU, two full
    eval passes plus a 64-step greedy generation each.

.PARAMETER TrainSteps
    max_steps for the short CPU smoke run (default 20; original config is 500).

.PARAMETER SkipInstall
    Reuse the existing .venv and skip all pip operations.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1
    powershell -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1 -WithTraining
    powershell -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1 -WithTraining -TrainSteps 50

.NOTES
    Intentionally 100% ASCII. Windows PowerShell 5.1 decodes BOM-less .ps1
    files using the system ANSI code page, so non-ASCII text may be mangled.
    Uses $LASTEXITCODE explicitly (PS 5.1 has no && / || operators).
    Native stderr is merged into stdout: Python logging goes to stderr, and
    PS 5.1 would otherwise surface every trainer step as NativeCommandError.
#>

[CmdletBinding()]
param(
    [string]$Python = "C:\Users\zentiali\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe",
    [switch]$WithTraining,
    [ValidateRange(1, 100000)]
    [int]$TrainSteps = 20,
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Out = Join-Path $Root "runs\windows-verify"
$Config = Join-Path $Root "autotrain\configs\wanxiang-smoke-4m.yaml"
$Baseline = Join-Path $Out "bundle_step0"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Exe,
        [string[]]$Arguments
    )
    Write-Host "-> $Label"
    # PS 5.1 wraps native stderr into an ErrorRecord on the ERROR stream.
    # Launched via `powershell -File ...`, that stream is re-reported by the
    # parent shell as NativeCommandError (red noise) even when the child is
    # perfectly healthy. Python logging (the trainer) writes to stderr, so
    # this happens on every training step. Merge stderr into stdout and emit
    # plain strings: nothing reaches the error stream, output stays live.
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.ToString()
            } else {
                $_
            }
        }
    } finally {
        $ErrorActionPreference = $old
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed (exit code $LASTEXITCODE)"
    }
}

Write-Host "wanxiang-autotrain M1/M1.5 Windows verification"
Write-Host "repo : $Root"
Write-Host "out  : $Out"

Push-Location -LiteralPath $Root
try {
    # ---- Preflight -------------------------------------------------------
    Write-Step "Preflight"
    if (-not (Test-Path -LiteralPath $Config)) {
        throw "config not found: $Config"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Root "evals\manifest.json"))) {
        throw "evals/manifest.json missing - run from a full checkout"
    }
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    Write-Host "preflight OK"

    # ---- 1. venv ---------------------------------------------------------
    Write-Step "Step 1 - Python 3.12 venv"
    if (Test-Path -LiteralPath $Py) {
        Write-Host "reusing existing venv: $Venv"
    } else {
        if (-not (Test-Path -LiteralPath $Python)) {
            throw "Python 3.12 not found: $Python`nPass -Python <path> to a Python 3.10-3.12 interpreter."
        }
        & $Python -m venv $Venv
        if ($LASTEXITCODE -ne 0) {
            throw "venv creation failed (exit code $LASTEXITCODE)"
        }
        Write-Host "created venv: $Venv"
    }
    Invoke-Checked "python version" $Py @("-c", "import sys; print(sys.version.split()[0])")

    # ---- 2. dependencies -------------------------------------------------
    Write-Step "Step 2 - Dependencies (CPU-only torch)"
    if ($SkipInstall) {
        Write-Host "skipped (-SkipInstall)"
    } else {
        # find_spec only: no heavy imports, and the probe writes solely to
        # stdout so a missing dependency never triggers NativeCommandError.
        $probe = "import importlib.util as u, sys; mods=('torch','numpy','yaml','safetensors','pytest'); missing=[m for m in mods if u.find_spec(m) is None]; print('missing: '+','.join(missing) if missing else 'deps already OK'); sys.exit(1 if missing else 0)"
        $old = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Py -c $probe 2>&1 | ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    $_.ToString()
                } else {
                    $_
                }
            }
        } finally {
            $ErrorActionPreference = $old
        }
        if ($LASTEXITCODE -ne 0) {
            Invoke-Checked "upgrade pip" $Py @("-m", "pip", "install", "--upgrade", "pip")
            Invoke-Checked "install CPU torch" $Py @(
                "-m", "pip", "install", "torch",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            )
            Invoke-Checked "install editable package" $Py @("-m", "pip", "install", "-e", ".[dev]")
            Invoke-Checked "recheck deps" $Py @("-c", "import torch, numpy, yaml, safetensors, pytest; print('deps OK')")
        }
    }

    # ---- 3. frozen corpora ----------------------------------------------
    Write-Step "Step 3 - Verify frozen eval corpora"
    Invoke-Checked "autotrain.eval verify" $Py @("-m", "autotrain.eval", "verify")

    # ---- 4. full test suite ---------------------------------------------
    Write-Step "Step 4 - Full pytest suite"
    Invoke-Checked "pytest tests/ -v" $Py @("-m", "pytest", "tests/", "-v")

    # ---- 5. export contract (no training) -------------------------------
    Write-Step "Step 5 - M1.5 export/load contract (step-0 baseline)"
    Invoke-Checked "export-initial" $Py @(
        "-m", "autotrain.export", "export-initial",
        "--config", $Config, "--out", $Baseline, "--seed", "123", "--force"
    )
    Invoke-Checked "describe baseline" $Py @("-m", "autotrain.export", "describe", "--bundle", $Baseline)
    Invoke-Checked "reload baseline" $Py @(
        "-c", "from autotrain.export.loader import load_bundle; m, i = load_bundle(r'$Baseline', device='cpu'); print('loaded step', i.step, 'vocab', i.vocab_size)"
    )

    # ---- 6. optional end-to-end A/B -------------------------------------
    Write-Step "Step 6 - M1.5 end-to-end A/B"
    if (-not $WithTraining) {
        Write-Host "skipped (pass -WithTraining to run the full CPU training A/B)"
    } else {
        $ShortCfg = Join-Path $Out "smoke_cpu.yaml"
        $raw = Get-Content -Raw -LiteralPath $Config
        if ($raw -notmatch '(?m)^\s*max_steps:\s*\d+') {
            throw "max_steps not found in $Config"
        }
        $replacement = '${1}' + [string]$TrainSteps
        $patched = $raw -replace '(?m)(^\s*max_steps:\s*)\d+', $replacement
        Set-Content -LiteralPath $ShortCfg -Value $patched -Encoding ASCII
        Write-Host "short config: $ShortCfg (max_steps=$TrainSteps, original untouched)"

        $TrainTokens = Join-Path $Out "train_tokens.bin"
        Invoke-Checked "make-corpus" $Py @(
            "-m", "autotrain.eval", "make-corpus", "--out", $TrainTokens, "--tokens", "2000000"
        )

        $RunDir = Join-Path $Out "smoke"
        Invoke-Checked "short CPU train" $Py @(
            "-m", "autotrain.trainer",
            "--config", $ShortCfg, "--data", $TrainTokens,
            "--out", $RunDir, "--db", (Join-Path $Out "experiments.db")
        )

        $CkptDir = Join-Path $RunDir "checkpoints"
        if (-not (Test-Path -LiteralPath $CkptDir)) {
            throw "no checkpoints directory: $CkptDir"
        }
        $Ckpt = Get-ChildItem -LiteralPath $CkptDir -Filter "ckpt_step_*.pt" |
            Sort-Object { [int]($_.BaseName -replace '^ckpt_step_', '') } |
            Select-Object -Last 1
        if ($null -eq $Ckpt) {
            throw "no ckpt_step_*.pt produced in $CkptDir"
        }
        $CkptPath = $Ckpt.FullName
        Write-Host "using checkpoint: $CkptPath"

        $Bundle = Join-Path $Out "bundle_trained"
        Invoke-Checked "export trained" $Py @(
            "-m", "autotrain.export", "export",
            "--checkpoint", $CkptPath, "--config", $ShortCfg,
            "--out", $Bundle, "--force"
        )
        Invoke-Checked "validate bundle vs checkpoint" $Py @(
            "-m", "autotrain.export", "validate",
            "--bundle", $Bundle, "--checkpoint", $CkptPath
        )

        # Both runs MUST use the same device, otherwise compare_reports
        # deliberately returns INCOMPARABLE.
        $Rep0 = Join-Path $Out "report_step0.json"
        $Rep1 = Join-Path $Out "report_trained.json"
        Invoke-Checked "eval run (step-0)" $Py @(
            "-m", "autotrain.eval", "run", "--bundle", $Baseline,
            "--device", "cpu", "--out", $Rep0
        )
        Invoke-Checked "eval run (trained)" $Py @(
            "-m", "autotrain.eval", "run", "--bundle", $Bundle,
            "--device", "cpu", "--out", $Rep1
        )
        Invoke-Checked "compare reports" $Py @("-m", "autotrain.eval", "compare", $Rep0, $Rep1)

        Write-Host ""
        Write-Host "reports: $Rep0 , $Rep1" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host "artifacts: $Out" -ForegroundColor Green
}
finally {
    Pop-Location
}
