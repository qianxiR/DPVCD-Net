$ErrorActionPreference = 'Stop'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$expDir = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive'

Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] START config_6 (X3D+B+C) fixed w=0.3")

$trainArgs = @(
    '-u', 'scripts\train\train_ablation.py',
    '--file_root', 'E:/rqx/dataes/LBFD-CD',
    '--ablation_config', '6',
    '--experiment_name', 'adaptive',
    '--save_dir', $expDir,
    '--loss_mode', 'fixed',
    '--loss_weight', '0.3',
    '--batch_size', '8',
    '--max_steps', '80000',
    '--learning_rate', '0.0002',
    '--num_workers', '0'
)

$log = Join-Path $expDir 'config_6_adaptive_train.log'
$err = Join-Path $expDir 'config_6_adaptive_err.log'

$p = Start-Process -FilePath $py -ArgumentList $trainArgs `
    -WorkingDirectory 'E:\rqx\DPVCD-Net' `
    -RedirectStandardOutput $log `
    -RedirectStandardError $err `
    -PassThru -NoNewWindow

Write-Output ("PID=" + $p.Id)
$p.WaitForExit()
$code = $p.ExitCode
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_6 EXIT code=" + $code)
