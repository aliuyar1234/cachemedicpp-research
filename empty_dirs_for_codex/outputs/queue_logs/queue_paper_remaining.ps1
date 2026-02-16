$ErrorActionPreference = "Stop"
$repo = "E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness"
$waitPid = 6472
while (Get-Process -Id $waitPid -ErrorAction SilentlyContinue) {
  Start-Sleep -Seconds 30
}
Set-Location $repo
$env:HF_HOME='E:\Model\huggingface'
$env:HF_HUB_CACHE='E:\Model\huggingface\hub'
$env:HUGGINGFACE_HUB_CACHE='E:\Model\huggingface\hub'
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$fullLog=Join-Path 'E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness\empty_dirs_for_codex\outputs\queue_logs' ("sweep_full_matrix_" + $stamp + ".log")
$oodLog=Join-Path 'E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness\empty_dirs_for_codex\outputs\queue_logs' ("sweep_ood_loto_all_" + $stamp + ".log")
python -m cachemedicpp.sweep --sweep_config configs/full_gpt2_matrix.yaml --out_root empty_dirs_for_codex/outputs/runs *>&1 | Tee-Object -FilePath $fullLog
if ($LASTEXITCODE -eq 0) {
  python -m cachemedicpp.sweep --sweep_config configs/ood_loto_all.yaml --out_root empty_dirs_for_codex/outputs/runs *>&1 | Tee-Object -FilePath $oodLog
}
