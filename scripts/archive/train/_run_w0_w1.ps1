$ErrorActionPreference = 'Continue'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$expDir = 'E:\rqx\DPVCD-Net\exp_ablation_adaptive'

# Step 1: wait for config_6 final model
$config6Final = Join-Path $expDir 'config_6_adaptive\LBFD-CD\final_model.pth'
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] waiting for config_6 final...")

$maxWait = 43200
$waited = 0
while (-not (Test-Path $config6Final) -and $waited -lt $maxWait) {
    Start-Sleep -Seconds 30
    $waited += 30
}

if (-not (Test-Path $config6Final)) {
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_6 wait timeout, abort")
    exit 1
}

Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] config_6 done, sleep 60s for GPU release...")
Start-Sleep -Seconds 60

# Step 2: run w=0 and w=1 loss experiments (Full model)
$experiments = @(
    @{name='loss_fixed_000'; weight='0.0'},
    @{name='loss_fixed_100'; weight='1.0'}
)

foreach ($exp in $experiments) {
    $expName = $exp.name
    $weight = $exp.weight
    Write-Output ""
    Write-Output "========================================"
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] START " + $expName + " (w=" + $weight + ")")
    Write-Output "========================================"

    $trainArgs = @(
        '-u', 'scripts\train\train_BCD.py',
        '--file_root', 'E:/rqx/dataes/LBFD-CD',
        '--save_dir', ('./exp_loss_ablation/' + $expName),
        '--loss_mode', 'fixed',
        '--loss_weight', $weight,
        '--gpu_id', '0',
        '--batch_size', '8',
        '--max_steps', '80000',
        '--num_workers', '0',
        '--learning_rate', '0.0002'
    )

    $logDir = Join-Path 'E:\rqx\DPVCD-Net\exp_loss_ablation' $expName
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir ($expName + '_train.log')
    $err = Join-Path $logDir ($expName + '_err.log')

    $p = Start-Process -FilePath $py -ArgumentList $trainArgs `
        -WorkingDirectory 'E:\rqx\DPVCD-Net' `
        -RedirectStandardOutput $log `
        -RedirectStandardError $err `
        -PassThru -NoNewWindow

    $p.WaitForExit()
    $code = $p.ExitCode
    Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $expName + " EXIT code=" + $code + " PID=" + $p.Id)

    Start-Sleep -Seconds 30
}

Write-Output ""
Write-Output "========================================"
Write-Output ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] w=0 and w=1 experiments ALL DONE")
Write-Output "========================================"
