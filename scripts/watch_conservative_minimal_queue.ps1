$ErrorActionPreference = "SilentlyContinue"

$project = "D:\la_v12"
$done = "D:\la_v12\result\analysis\journal_validation\proof_conservative_mechanism_swat_e8.done.json"
$parentPid = 26332
$python = "D:\Anaconda3\envs\lagraph5070\python.exe"
$log = "D:\la_v12\logs\journal_validation\conservative_mechanism_minimal_watch.log"

"[$(Get-Date -Format s)] watcher started" | Add-Content -LiteralPath $log

while ((Get-Process -Id $parentPid -ErrorAction SilentlyContinue) -and !(Test-Path -LiteralPath $done)) {
    Start-Sleep -Seconds 30
}

if (Test-Path -LiteralPath $done) {
    "[$(Get-Date -Format s)] swat conservative done; stopping original queue" | Add-Content -LiteralPath $log
    Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.ParentProcessId -eq $parentPid } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $parentPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5

    "[$(Get-Date -Format s)] starting WADI conservative only" | Add-Content -LiteralPath $log
    Start-Process `
        -FilePath $python `
        -ArgumentList @(
            "-u",
            "scripts\run_journal_validation_queue.py",
            "--suite",
            "conservative-mechanism-proof",
            "--only",
            "proof_conservative_mechanism_wadi_e8",
            "--skip-completed"
        ) `
        -WorkingDirectory $project `
        -RedirectStandardOutput "D:\la_v12\logs\journal_validation\conservative_mechanism_wadi_only.out.log" `
        -RedirectStandardError "D:\la_v12\logs\journal_validation\conservative_mechanism_wadi_only.err.log" `
        -WindowStyle Hidden
} else {
    "[$(Get-Date -Format s)] original queue exited before done marker" | Add-Content -LiteralPath $log
}
