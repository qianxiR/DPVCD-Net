$ErrorActionPreference = 'Stop'
Set-Location 'E:\rqx\DPVCD-Net'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUTF8 = '1'

$py = 'D:\Anaconda3\envs\Seg_310\python.exe'

$args = @(
    '-u', '-c',
@'
import sys, os, torch, numpy as np
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import data.dataset as RSDataset
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter
from model.trainer_ablation import TrainerAblation, create_ablation_model

class Args:
    in_height = 256
    in_width = 256
    num_perception_frame = 1
    pretrained = r'model\X3D_L.pyth'
    use_attention = True
    use_aspp = True
    use_transformer = False

args = Args()
model = create_ablation_model(args=args, experiment_id=5)
sd = torch.load(r'exp_ablation_adaptive\config_5_adaptive\LBFD-CD\best_model.pth', map_location='cpu', weights_only=False)
if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
missing, unexpected = model.load_state_dict(sd, strict=False)
print('missing keys:', len(missing), 'unexpected keys:', len(unexpected))
model.cuda().eval()

train_tf, val_tf = RSTransforms.BCDTransforms.get_transform_pipelines(args)
test_data = RSDataset.BCDDataset(file_root=r'E:\rqx\dataes\LBFD-CD', split='test', transform=val_tf)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

meter = ConfuseMatrixMeter(n_class=2)
with torch.no_grad():
    for i, (img, target) in enumerate(test_loader):
        pre = img[:, :3, :, :].cuda().float()
        post = img[:, 3:, :, :].cuda().float()
        target = target.cuda().float()
        out = model(pre, post)
        pred = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out)).long()
        meter.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
        if (i+1) % 50 == 0:
            print('processed %d/%d' % (i+1, len(test_loader)))

s = meter.get_scores()
print('')
print('=== config_5 (X3D+A+B) test result (best_model epoch139) ===')
print('Kappa=%.4f  IoU=%.4f  F1=%.4f  OA=%.4f  Recall=%.4f  Precision=%.4f' % (
    s['Kappa'], s['IoU'], s['F1'], s['OA'], s['recall'], s['precision']))
'@
)

& $py @args
