$ErrorActionPreference = 'Continue'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$expDir = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive'

function Run-Config($cid, $name, $extraArgs) {
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
    if ($extraArgs) {
        $trainArgs += $extraArgs
    }

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

# Step 1: resume config_4 from epoch 220
Run-Config 4 "X3D+C resume" @('--resume', (Join-Path $expDir 'config_4_adaptive\LBFD-CD\checkpoint.pth.tar'))

# Step 2: run all module-B configs from scratch (original module B)
Run-Config 3 "X3D+B" $null
Run-Config 5 "X3D+A+B" $null
Run-Config 6 "X3D+B+C" $null
Run-Config 7 "Full" $null

Write-Output ""
Write-Output "========================================"
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] ALL DONE (4-resume, 3, 5, 6, 7)")
Write-Output "========================================"
