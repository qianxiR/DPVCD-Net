$ErrorActionPreference = 'Continue'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$expDir = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive'

$configs = @(1, 2, 3, 4)
$configNames = @{ 1='X3D'; 2='X3D+A'; 3='X3D+B'; 4='X3D+C' }

foreach ($cid in $configs) {
    $name = $configNames[$cid]
    Write-Output ""
    Write-Output "========================================"
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] START config_${cid} (${name})")
    Write-Output "========================================"

    $args = @(
        '-u', 'scripts\train\train_ablation.py',
        '--file_root', 'E:/rqx/dataes/LBFD-CD',
        '--ablation_config', "$cid",
        '--experiment_name', 'adaptive',
        '--save_dir', $expDir,
        '--loss_mode', 'adaptive',
        '--loss_weight', '0.1',
        '--batch_size', '8',
        '--max_steps', '80000',
        '--learning_rate', '0.0002',
        '--num_workers', '0'
    )

    $log = Join-Path $expDir "config_${cid}_adaptive_train.log"
    $err = Join-Path $expDir "config_${cid}_adaptive_err.log"

    $p = Start-Process -FilePath $py -ArgumentList $args `
        -WorkingDirectory 'E:\rqx\DPVCD-Net' `
        -RedirectStandardOutput $log `
        -RedirectStandardError $err `
        -PassThru -NoNewWindow

    $p.WaitForExit()
    $code = $p.ExitCode
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_${cid} (${name}) EXIT code=$code PID=" + $p.Id)
}

Write-Output ""
Write-Output "========================================"
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] ALL CONFIGS DONE")
Write-Output "========================================"
