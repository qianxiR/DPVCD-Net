$ErrorActionPreference = 'Continue'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$expDir = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive'

# 清理 config_7 残留
$config7Dir = Join-Path $expDir 'config_7_adaptive'
if (Test-Path $config7Dir) {
    Remove-Item -Recurse -Force $config7Dir
    Write-Output "cleaned config_7_adaptive residual"
}

$configs = @(5, 7)
$configNames = @{ 5='X3D+A+B(V2)'; 7='Full' }

foreach ($cid in $configs) {
    $name = $configNames[$cid]
    Write-Output ""
    Write-Output "========================================"
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] START config_" + $cid + " (" + $name + ")")
    Write-Output "========================================"

    $trainArgs = @(
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

    $log = Join-Path $expDir ("config_" + $cid + "_adaptive_train.log")
    $err = Join-Path $expDir ("config_" + $cid + "_adaptive_err.log")

    $p = Start-Process -FilePath $py -ArgumentList $trainArgs `
        -WorkingDirectory 'E:\rqx\DPVCD-Net' `
        -RedirectStandardOutput $log `
        -RedirectStandardError $err `
        -PassThru -NoNewWindow

    Write-Output ("PID=" + $p.Id)
    $p.WaitForExit()
    $code = $p.ExitCode
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_" + $cid + " (" + $name + ") EXIT code=" + $code)

    Start-Sleep -Seconds 30
}

Write-Output ""
Write-Output "========================================"
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_5 and config_7 ALL DONE")
Write-Output "========================================"
