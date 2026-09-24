# Launch Phase 7 Stage 2 as parallel shards.
#
#   powershell -ExecutionPolicy Bypass -File tools\launch_stage2.ps1
#
# Same harness as Stage 1: one results file and one log per worker, nothing
# piped, output redirected straight to a file.
#
# Two passes per shard, in this order and NOT in parallel:
#
#   1. logreg, svm_linear, svm_rbf, mlp   -- five seeds, both directions
#   2. transformer                        -- two seeds, reduced arm
#
# The order matters. Stage 2 is long enough that it may be interrupted, and the
# sklearn + MLP arm is what every table in the paper is built on. Finishing it
# first means an interrupted run still leaves a complete, reportable result
# rather than five partial arms.
#
# Resumable and idempotent. A worker that dies can be restarted with the same
# command; it reads its own shard file plus results\runs.jsonl and skips
# exactly what is already committed. Stage 1 cells that survived pruning are
# already in runs.jsonl and are NOT recomputed.
#
# Progress:  Get-ChildItem logs\stage2_shard*.log | ForEach-Object { $_.Name; Get-Content $_ -Tail 2 }
# Merge:     python tools\merge_shards.py --pattern "results/shards/stage2_*.jsonl"

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$nShards = 4
# 16 cores across 4 workers. Without the cap each worker spawns one BLAS thread
# per core and they fight; that is slower than running the shards serially.
$threads = 4

New-Item -ItemType Directory -Force -Path "results\shards" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Output "Stage 2: $nShards shards, $threads threads each"
Write-Output "repo: $repo"
Write-Output ""

for ($i = 0; $i -lt $nShards; $i++) {
    $log    = Join-Path $repo "logs\stage2_shard$i.log"
    $main   = Join-Path $repo "results\shards\stage2_main_shard$i.jsonl"
    $trans  = Join-Path $repo "results\shards\stage2_transformer_shard$i.jsonl"

    $inner = @"
`$env:PYTHONPATH='src'
`$env:OMP_NUM_THREADS='$threads'
`$env:MKL_NUM_THREADS='$threads'
`$env:OPENBLAS_NUM_THREADS='$threads'
`$env:NUMEXPR_NUM_THREADS='$threads'
Set-Location '$repo'
python -u -m ser.cli run-grid --stage 2 --corpora ravdess,cremad ``
    --families logreg,svm_linear,svm_rbf,mlp ``
    --shard $i --n-shards $nShards --results '$main' --heartbeat-every 10
python -u -m ser.cli run-grid --stage 2 --corpora ravdess,cremad ``
    --families transformer ``
    --shard $i --n-shards $nShards --results '$trans' --heartbeat-every 2
"@

    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
    Start-Process -FilePath "powershell" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded `
        -RedirectStandardOutput $log `
        -RedirectStandardError "$log.err" `
        -WindowStyle Hidden
    Write-Output "  shard $i -> $log"
}

Write-Output ""
Write-Output "All $nShards workers started. They run in the background; this window can close."
Write-Output ""
Write-Output "Check progress:"
Write-Output "  Get-ChildItem logs\stage2_shard*.log | ForEach-Object { `$_.Name; Get-Content `$_ -Tail 2 }"
Write-Output ""
Write-Output "When every log ends with the sklearn AND transformer passes done:"
Write-Output "  python tools\merge_shards.py --pattern `"results/shards/stage2_*.jsonl`""
