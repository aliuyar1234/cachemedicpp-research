$ErrorActionPreference = "Stop"
$repo = "E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness"
$runMve = "E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness\empty_dirs_for_codex\outputs\runs\mve_gpu_20260215_121245"
$outRoot = "E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness\empty_dirs_for_codex\outputs\runs_fast_paper_core"
$logRoot = "E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness\empty_dirs_for_codex\outputs\queue_logs"
Set-Location $repo
$env:HF_HOME='E:\Model\huggingface'
$env:HF_HUB_CACHE='E:\Model\huggingface\hub'
$env:HUGGINGFACE_HUB_CACHE='E:\Model\huggingface\hub'
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$mveLog=Join-Path $logRoot ("mve_eval_stability_plots_" + $stamp + ".log")
$coreLog=Join-Path $logRoot ("fast_paper_core_sweep_" + $stamp + ".log")
$secondLog=Join-Path $logRoot ("fast_second_model_" + $stamp + ".log")
$statusFile=Join-Path $logRoot ("queue_sweetspot_core_" + $stamp + ".status.txt")
$secondRunDir=Join-Path $outRoot ("second_model_gpt2_large_" + $stamp)
"START $(Get-Date -Format o)" | Out-File -FilePath $statusFile -Encoding utf8
python -m cachemedicpp.eval --config configs/mve.yaml --run_dir $runMve *>&1 | Tee-Object -FilePath $mveLog
if ($LASTEXITCODE -eq 0) {
  python -m cachemedicpp.stability --config configs/mve.yaml --run_dir $runMve *>&1 | Tee-Object -FilePath $mveLog -Append
}
if ($LASTEXITCODE -eq 0) {
  python -m cachemedicpp.plots --run_dir $runMve *>&1 | Tee-Object -FilePath $mveLog -Append
}
if ($LASTEXITCODE -eq 0) {
  python -m cachemedicpp.sweep --sweep_config configs/fast_paper_sweep_core.yaml --out_root $outRoot *>&1 | Tee-Object -FilePath $coreLog
}
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
"END $(Get-Date -Format o) EXIT=$LASTEXITCODE" | Out-File -FilePath $statusFile -Append -Encoding utf8
