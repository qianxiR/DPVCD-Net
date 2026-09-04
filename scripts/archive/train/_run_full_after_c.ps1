$ErrorActionPreference = 'Continue'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

# 等待 config_4 的 final_model.pth 生成（表示 config_4 训练完成）
$config4Final = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive\config_4_adaptive\LBFD-CD\final_model.pth'
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] 等待 config_4 (X3D+C) 完成...")

$maxWait = 36000  # 最长等待 10 小时
$waited = 0
while (-not (Test-Path $config4Final) -and $waited -lt $maxWait) {
    Start-Sleep -Seconds 30
    $waited += 30
}

if (-not (Test-Path $config4Final)) {
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_4 等待超时，放弃启动 Full")
    exit 1
}

Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_4 已完成，等待 60 秒释放显存...")
Start-Sleep -Seconds 60

# 启动 Full (config_7)
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] START config_7 (Full = X3D+A+B+C)")

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$expDir = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive'

$args = @(
    '-u', 'scripts\train\train_ablation.py',
    '--file_root', 'E:/rqx/dataes/LBFD-CD',
    '--ablation_config', '7',
    '--experiment_name', 'adaptive',
    '--save_dir', $expDir,
    '--loss_mode', 'adaptive',
    '--loss_weight', '0.1',
    '--batch_size', '8',
    '--max_steps', '80000',
    '--learning_rate', '0.0002',
    '--num_workers', '0'
)

$log = Join-Path $expDir 'config_7_adaptive_train.log'
$err = Join-Path $expDir 'config_7_adaptive_err.log'

$p = Start-Process -FilePath $py -ArgumentList $args `
    -WorkingDirectory 'E:\rqx\DPVCD-Net' `
    -RedirectStandardOutput $log `
    -RedirectStandardError $err `
    -PassThru -NoNewWindow

$p.WaitForExit()
$code = $p.ExitCode
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_7 (Full) EXIT code=$code PID=" + $p.Id)
