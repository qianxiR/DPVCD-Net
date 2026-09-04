import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

from model.losses import CompositeBCEIoUFocalLoss
from model.utils import load_checkpoint


class CompositeBCEIoUFocalLossTest(unittest.TestCase):
    """
    入参:
    - 无

    方法:
    - 验证复合损失公式、混合权重约束、梯度更新和状态恢复

    出参:
    - unittest测试结果
    """

    def setUp(self):
        """
        入参:
        - 无

        方法:
        - 构造固定概率、二值标签和初始权重为0.3的损失模块

        出参:
        - None
        """
        self.inputs = torch.tensor(
            [[[[0.9, 0.2], [0.6, 0.1]]], [[[0.3, 0.8], [0.4, 0.7]]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        self.targets = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 1.0], [0.0, 1.0]]]],
            dtype=torch.float32,
        )
        self.criterion = CompositeBCEIoUFocalLoss(
            mode='adaptive',
            initial_weight=0.3,
        )

    def test_matches_equation_16(self):
        """
        入参:
        - 固定测试概率与标签

        方法:
        - 独立计算BCE、逐样本Soft IoU和Focal后按公式16组合

        出参:
        - 复合损失与参考公式数值一致
        """
        bce_loss = F.binary_cross_entropy(self.inputs, self.targets)
        intersection = (self.inputs * self.targets).sum(dim=(1, 2, 3))
        union = (
            self.inputs.sum(dim=(1, 2, 3))
            + self.targets.sum(dim=(1, 2, 3))
            - intersection
        )
        iou_loss = 1 - ((intersection + 1e-5) / (union + 1e-5)).mean()
        pt = self.inputs * self.targets + (1 - self.inputs) * (1 - self.targets)
        focal_loss = (-0.8 * (1 - pt) ** 2.0 * torch.log(pt)).mean()
        expected = 0.7 * (bce_loss + iou_loss) + 0.3 * focal_loss

        actual = self.criterion(self.inputs, self.targets)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertAlmostEqual(self.criterion.mixing_weight.item(), 0.3, places=6)

    def test_weight_is_learnable_and_bounded(self):
        """
        入参:
        - 可学习复合损失模块

        方法:
        - 执行一次Adam更新并检查logit梯度及Sigmoid边界

        出参:
        - w发生更新且始终位于(0, 1)
        """
        optimizer = torch.optim.Adam(self.criterion.parameters(), lr=0.1)
        initial_weight = self.criterion.mixing_weight.detach().item()

        loss = self.criterion(self.inputs, self.targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        updated_weight = self.criterion.mixing_weight.detach().item()
        self.assertIsNotNone(self.criterion.weight_logit.grad)
        self.assertNotEqual(updated_weight, initial_weight)
        self.assertGreater(updated_weight, 0.0)
        self.assertLess(updated_weight, 1.0)

    def test_fixed_weights_do_not_update(self):
        """
        入参:
        - 固定权重0.3、0.5和0.7

        方法:
        - 分别执行一次反向传播和优化器更新，检查冻结的logit参数及混合权重

        出参:
        - 固定模式参数无梯度，且三组权重均保持初始值
        """
        for fixed_weight in (0.3, 0.5, 0.7):
            with self.subTest(fixed_weight=fixed_weight):
                inputs = self.inputs.detach().clone().requires_grad_(True)
                criterion = CompositeBCEIoUFocalLoss(
                    mode='fixed',
                    initial_weight=fixed_weight,
                )
                optimizer = torch.optim.SGD(
                    [inputs, criterion.weight_logit],
                    lr=0.1,
                )

                optimizer.zero_grad()
                criterion(inputs, self.targets).backward()
                optimizer.step()

                self.assertFalse(criterion.weight_logit.requires_grad)
                self.assertIsNone(criterion.weight_logit.grad)
                self.assertAlmostEqual(
                    criterion.mixing_weight.item(),
                    fixed_weight,
                    places=6,
                )

    def test_state_dict_restores_weight(self):
        """
        入参:
        - 人工调整后的损失状态字典

        方法:
        - 将状态加载到不同初始权重的新实例

        出参:
        - 恢复实例的w与保存值一致
        """
        with torch.no_grad():
            self.criterion.weight_logit.add_(0.5)
        restored = CompositeBCEIoUFocalLoss(
            mode='adaptive',
            initial_weight=0.7,
        )
        restored.load_state_dict(self.criterion.state_dict())

        self.assertAlmostEqual(
            restored.mixing_weight.item(),
            self.criterion.mixing_weight.item(),
            places=7,
        )

    def test_checkpoint_restores_loss_and_optimizer(self):
        """
        入参:
        - 包含网络、损失和Adam状态的临时检查点

        方法:
        - 先执行一次联合更新，再加载到新建的网络、损失和优化器

        出参:
        - 恢复后的w、训练元数据和优化器状态与保存值一致
        """
        model = torch.nn.Conv2d(1, 1, kernel_size=1)
        optimizer = torch.optim.Adam(
            [
                {'params': model.parameters()},
                {'params': self.criterion.parameters(), 'weight_decay': 0.0},
            ],
            lr=0.01,
            weight_decay=1e-4,
        )
        predictions = torch.sigmoid(model(self.inputs[:, :, :1, :1]))
        targets = self.targets[:, :, :1, :1]
        optimizer.zero_grad()
        self.criterion(predictions, targets).backward()
        optimizer.step()
        saved_weight = self.criterion.mixing_weight.detach().item()

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / 'checkpoint.pth.tar'
            torch.save(
                {
                    'epoch': 4,
                    'state_dict': model.state_dict(),
                    'criterion_state_dict': self.criterion.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_f1': 0.81,
                    'cur_iter': 120,
                },
                checkpoint_path,
            )

            restored_model = torch.nn.Conv2d(1, 1, kernel_size=1)
            restored_criterion = CompositeBCEIoUFocalLoss(
                mode='adaptive',
                initial_weight=0.7,
            )
            restored_optimizer = torch.optim.Adam(
                [
                    {'params': restored_model.parameters()},
                    {'params': restored_criterion.parameters(), 'weight_decay': 0.0},
                ],
                lr=0.01,
                weight_decay=1e-4,
            )
            _, epoch, best_f1, cur_iter = load_checkpoint(
                checkpoint_path,
                restored_model,
                restored_criterion,
                restored_optimizer,
            )

        self.assertAlmostEqual(restored_criterion.mixing_weight.item(), saved_weight, places=7)
        self.assertEqual(epoch, 4)
        self.assertAlmostEqual(best_f1, 0.81)
        self.assertEqual(cur_iter, 120)
        self.assertGreater(len(restored_optimizer.state), 0)


if __name__ == '__main__':
    unittest.main()
