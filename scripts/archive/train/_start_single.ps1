param(
    [Parameter(Mandatory=$true)][int]$ConfigId,
    [Parameter(Mandatory=$true)][string]$ExpDir
)

$ErrorActionPreference = 'Stop'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$args = @(
    '-u', 'scripts\train\train_ablation.py',
    '--file_root', 'E:/rqx/dataes/LBFD-CD',
    '--ablation_config', "$ConfigId",
    '--experiment_name', 'adaptive',
    '--save_dir', $ExpDir,
    '--loss_mode', 'adaptive',
    '--loss_weight', '0.1',
    '--batch_size', '8',
    '--max_steps', '80000',
    '--learning_rate', '0.0002',
    '--num_workers', '0'
)

$logName = "config_${ConfigId}_adaptive_train.log"
$errName = "config_${ConfigId}_adaptive_err.log"

$p = Start-Process -FilePath $py -ArgumentList $args `
    -WorkingDirectory 'E:\rqx\DPVCD-Net' `
    -RedirectStandardOutput (Join-Path $ExpDir $logName) `
    -RedirectStandardError  (Join-Path $ExpDir $errName) `
    -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 3
Write-Output ("PID=" + $p.Id)
