# Launch Phase 7 Stage 1 as 4 parallel shards.
#
#   powershell -ExecutionPolicy Bypass -File tools\launch_stage1.ps1
#
# Each worker gets its own results file and its own log. Nothing is piped:
# output is redirected straight to a file, because buffered pipes have
# swallowed output repeatedly on this project.
#
# Each worker does screening (sklearn + MLP) for its shard, then the
# transformer probe for its shard. Both are resumable and idempotent -- a
# worker that dies can simply be restarted, and it will skip exactly the runs
# already committed to its shard file.
#
# Progress:  Get-Content logs\stage1_shard0.log -Tail 5
# All logs:  Get-ChildItem logs\stage1_shard*.log | ForEach-Object { $_.Name; Get-Content $_ -Tail 2 }

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$nShards = 4
# 16 cores across 4 workers. Capped explicitly: without this each worker
# spawns as many BLAS threads as there are cores and they fight each other,
# which is slower than running them serially.
$threads = 4

New-Item -ItemType Directory -Force -Path "results\shards" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Output "Stage 1: $nShards shards, $threads threads each"
Write-Output "repo: $repo"
Write-Output ""

for ($i = 0; $i -lt $nShards; $i++) {
    $log     = Join-Path $repo "logs\stage1_shard$i.log"
    $screen  = Join-Path $repo "results\shards\stage1_screen_shard$i.jsonl"
    $probe   = Join-Path $repo "results\shards\stage1_probe_shard$i.jsonl"

    # Both phases in one child process, so the shard finishes screening before
    # starting its probe and only one python starts per shard.
    $inner = @"
`$env:PYTHONPATH='src'
`$env:OMP_NUM_THREADS='$threads'
`$env:MKL_NUM_THREADS='$threads'
`$env:OPENBLAS_NUM_THREADS='$threads'
`$env:NUMEXPR_NUM_THREADS='$threads'
Set-Location '$repo'
python -u -m ser.cli run-grid --stage 1 --corpora ravdess,cremad ``
    --families logreg,svm_linear,svm_rbf,mlp ``
    --shard $i --n-shards $nShards --results '$screen' --heartbeat-every 5
python -u -m ser.cli run-grid --stage 1 --corpora ravdess,cremad --probe ``
    --shard $i --n-shards $nShards --results '$probe' --heartbeat-every 2
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
Write-Output "All 4 workers started. They run in the background; this window can close."
Write-Output ""
Write-Output "Check progress:"
Write-Output "  Get-ChildItem logs\stage1_shard*.log | ForEach-Object { `$_.Name; Get-Content `$_ -Tail 2 }"
Write-Output ""
Write-Output "When every log ends with 'ok,' and no python remains:"
Write-Output "  python tools\merge_shards.py"
