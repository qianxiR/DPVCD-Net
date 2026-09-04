$ErrorActionPreference = 'Continue'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$expDir = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive'

# Step 1: wait for config_5 final model
$config5Final = Join-Path $expDir 'config_5_adaptive\LBFD-CD\final_model.pth'
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] waiting for config_5 final...")

$maxWait = 43200
$waited = 0
while (-not (Test-Path $config5Final) -and $waited -lt $maxWait) {
    Start-Sleep -Seconds 30
    $waited += 30
}

if (-not (Test-Path $config5Final)) {
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_5 wait timeout, abort")
    exit 1
}

Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_5 done, sleep 60s for GPU release...")
Start-Sleep -Seconds 60

# Step 2: run config_6 (B+C) and config_7 (Full) serially
$configs = @(6, 7)
$configNames = @{ 6='X3D+B+C'; 7='Full' }

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

    $p.WaitForExit()
    $code = $p.ExitCode
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_" + $cid + " (" + $name + ") EXIT code=" + $code + " PID=" + $p.Id)

    Start-Sleep -Seconds 30
}

Write-Output ""
Write-Output "========================================"
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_6 and config_7 ALL DONE")
Write-Output "========================================"
