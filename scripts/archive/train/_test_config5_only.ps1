$ErrorActionPreference = 'Stop'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$expDir = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive'

# 续训 config_5 从 epoch 140 到完成（268）
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] Resume config_5 from epoch 140 to completion...")

$trainArgs = @(
    '-u', 'scripts\train\train_ablation.py',
    '--file_root', 'E:/rqx/dataes/LBFD-CD',
    '--ablation_config', '5',
    '--experiment_name', 'adaptive',
    '--save_dir', $expDir,
    '--loss_mode', 'adaptive',
    '--loss_weight', '0.1',
    '--batch_size', '8',
    '--max_steps', '80000',
    '--learning_rate', '0.0002',
    '--num_workers', '0',
    '--resume', (Join-Path $expDir 'config_5_adaptive\LBFD-CD\checkpoint.pth.tar')
)

$log = Join-Path $expDir 'config_5_adaptive_train.log'
$err = Join-Path $expDir 'config_5_adaptive_err.log'

$p = Start-Process -FilePath $py -ArgumentList $trainArgs `
    -WorkingDirectory 'E:\rqx\DPVCD-Net' `
    -RedirectStandardOutput $log `
    -RedirectStandardError $err `
    -PassThru -NoNewWindow

$p.WaitForExit()
$code = $p.ExitCode
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_5 EXIT code=" + $code)
Write-Output "config_5 DONE. NOT starting config_6/config_7."
