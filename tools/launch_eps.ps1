# Relaunch the CORAL eps=100/1000 probe, both directions.
#
#   powershell -ExecutionPolicy Bypass -File tools\launch_eps.ps1
#
# 120 runs enumerated; 35 are already complete and are skipped. Split into two
# workers, one per transfer direction.
#
# WHY DIRECTION AND NOT BACKBONE. The sweep was split by backbone because each
# worker then held one backbone's feature cache instead of three, which is what
# fixed the memory thrashing. This probe is single-backbone (hubert) by design,
# so there is no backbone axis to split on; direction is the axis that halves
# the work, and both workers touch the same two cache entries either way. The
# memory discipline here comes from the worker count and the thread cap, not
# from the split.
#
# TWO WORKERS, NOT FOUR. The previous run of this probe died because it was one
# of six concurrent python processes on a 15.6 GB machine whose committed memory
# reached 20.6 GB. Everything was swapping and each worker got roughly a quarter
# of a core. Do not raise this number, and do not start it while the layer sweep
# is still running.
#
# Resumable across restarts and across shard layouts: each worker reads every
# results/eps_*.jsonl and results/shards/eps_*.jsonl before deciding what to do,
# so the shard split can change between runs without redoing work.
#
# Progress:  Get-ChildItem logs\eps_*.log | ForEach-Object { $_.Name; Get-Content $_ -Tail 2 }

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Refuse to start on top of the sweep. Two heavy jobs at once is exactly what
# produced 26 hours of thrashing for 1200 rows.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*layer_sweep_v2*" }
if ($running) {
    Write-Output "REFUSING TO START: $($running.Count) layer_sweep_v2 worker(s) are still running."
    Write-Output "Wait for the sweep to finish, or stop it first. Running both"
    Write-Output "oversubscribes memory and both jobs slow to a crawl."
    exit 1
}

New-Item -ItemType Directory -Force -Path "results\shards" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$threads = 4
$directions = @(
    @{ name = "fwd"; spec = "ravdess>cremad" },
    @{ name = "rev"; spec = "cremad>ravdess" }
)

Write-Output "eps probe: 2 workers (one per direction), $threads threads each"
Write-Output "repo: $repo"
Write-Output ""

foreach ($d in $directions) {
    $log = Join-Path $repo "logs\eps_$($d.name).log"
    $out = Join-Path $repo "results\shards\eps_$($d.name).jsonl"

    $inner = @"
`$env:PYTHONPATH='src'
`$env:OMP_NUM_THREADS='$threads'
`$env:MKL_NUM_THREADS='$threads'
`$env:OPENBLAS_NUM_THREADS='$threads'
`$env:NUMEXPR_NUM_THREADS='$threads'
Set-Location '$repo'
python -u tools/eps_asymptote.py --runs ``
    --directions '$($d.spec)' ``
    --families logreg,svm_rbf,mlp --aggs last,layer ``
    --results '$out'
"@

    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
    Start-Process -FilePath "powershell" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded `
        -RedirectStandardOutput $log `
        -RedirectStandardError "$log.err" `
        -WindowStyle Hidden
    Write-Output "  $($d.spec) -> $log"
}

Write-Output ""
Write-Output "Both workers started. Expect roughly 85 runs total."
Write-Output ""
Write-Output "Check progress:"
Write-Output "  Get-ChildItem logs\eps_*.log | ForEach-Object { `$_.Name; Get-Content `$_ -Tail 2 }"
Write-Output ""
Write-Output "When both are done, regenerate the report:"
Write-Output "  python tools\eps_asymptote_report.py"
