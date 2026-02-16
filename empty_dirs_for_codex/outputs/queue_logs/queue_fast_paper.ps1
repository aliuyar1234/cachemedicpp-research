$ErrorActionPreference = "Stop"
$repo = "E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness"
$outRoot = "E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness\empty_dirs_for_codex\outputs\runs_fast_paper"
$logRoot = "E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness\empty_dirs_for_codex\outputs\queue_logs"
Set-Location $repo
$env:HF_HOME='E:\Model\huggingface'
$env:HF_HUB_CACHE='E:\Model\huggingface\hub'
$env:HUGGINGFACE_HUB_CACHE='E:\Model\huggingface\hub'
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$sweepLog=Join-Path $logRoot ("fast_paper_sweep_" + $stamp + ".log")
$secondLog=Join-Path $logRoot ("fast_second_model_" + $stamp + ".log")
$secondRunDir=Join-Path $outRoot ("second_model_gpt2_large_" + $stamp)
python -m cachemedicpp.sweep --sweep_config configs/fast_paper_sweep.yaml --out_root $outRoot *>&1 | Tee-Object -FilePath $sweepLog
if ($LASTEXITCODE -eq 0) {
  python -m cachemedicpp.train --config configs/fast_second_model_gpt2_large.yaml --run_dir $secondRunDir *>&1 | Tee-Object -FilePath $secondLog
}
if ($LASTEXITCODE -eq 0) {
  python -m cachemedicpp.eval --config configs/fast_second_model_gpt2_large.yaml --run_dir $secondRunDir *>&1 | Tee-Object -FilePath $secondLog -Append
}
if ($LASTEXITCODE -eq 0) {
  python -m cachemedicpp.stability --config configs/fast_second_model_gpt2_large.yaml --run_dir $secondRunDir *>&1 | Tee-Object -FilePath $secondLog -Append
}
if ($LASTEXITCODE -eq 0) {
  python -m cachemedicpp.plots --run_dir $secondRunDir *>&1 | Tee-Object -FilePath $secondLog -Append
}
