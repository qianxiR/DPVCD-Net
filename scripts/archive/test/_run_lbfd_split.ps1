$ErrorActionPreference = 'Stop'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'
$args = @(
    '-u', 'scripts\test\test_lbfd_split.py',
    '--file_root', 'E:\rqx\dataes\LBFD-CD',
    '--model_path', 'exp_loss_ablation\loss_fixed_030\LBFD-CD\best_model.pth',
    '--batch_size', '1'
)

& $py @args
