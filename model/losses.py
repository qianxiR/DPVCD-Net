import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    入参:
    - alpha (float): Focal缩放系数
    - gamma (float): 困难样本聚焦指数

    方法:
    - 根据二值目标选择正确类别概率并降低易分类像素贡献

    出参:
    - 标量Focal损失
    """

    def __init__(self, alpha=0.8, gamma=2.0):
        """
        入参:
        - alpha (float): Focal缩放系数，默认0.8
        - gamma (float): 困难样本聚焦指数，默认2.0

        方法:
        - 保存无梯度超参数，不注册额外训练参数

        出参:
        - None
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        入参:
        - inputs (torch.Tensor): Sigmoid后的变化概率，形状[B, 1, H, W]
        - targets (torch.Tensor): 二值变化标签，与inputs同形状

        方法:
        - 计算-alpha*(1-pt)^gamma*log(pt)并对像素取平均

        出参:
        - loss (torch.Tensor): 标量Focal损失
        """
        eps = 1e-6
        probabilities = torch.clamp(inputs, eps, 1.0 - eps)
        target_probabilities = (
            probabilities * targets + (1 - probabilities) * (1 - targets)
        )
        loss = (
            -self.alpha
            * (1 - target_probabilities) ** self.gamma
            * torch.log(target_probabilities)
        )
        return loss.mean()


class CompositeBCEIoUFocalLoss(nn.Module):
    """
    入参:
    - mode (str): adaptive或fixed，控制混合权重是否参与梯度更新
    - initial_weight (float): Focal混合权重w的初始值或固定值
    - alpha (float): Focal缩放系数
    - gamma (float): Focal聚焦指数
    - smooth (float): Soft IoU的数值稳定项

    方法:
    - 通过w=sigmoid(theta)将权重约束在(0, 1)
    - 按(1-w)*(L_bce+L_iou)+w*L_focal计算复合损失

    出参:
    - 支持固定权重和自适应权重的统一损失模块
    """

    MODES = {'adaptive': True, 'fixed': False}

    def __init__(
        self,
        mode='adaptive',
        initial_weight=0.3,
        alpha=0.8,
        gamma=2.0,
        smooth=1e-5,
    ):
        """
        入参:
        - mode (str): adaptive时训练theta，fixed时冻结theta
        - initial_weight (float): w的初始值或固定值，必须位于(0, 1)
        - alpha (float): Focal缩放系数，默认0.8
        - gamma (float): Focal聚焦指数，默认2.0
        - smooth (float): Soft IoU稳定项，默认1e-5

        方法:
        - 通过模式映射确定theta的requires_grad并在logit空间保存w

        出参:
        - None
        """
        super().__init__()
        requires_grad = self.MODES[mode]
        initial_logit = torch.logit(torch.tensor(initial_weight, dtype=torch.float32))
        self.weight_logit = nn.Parameter(initial_logit, requires_grad=requires_grad)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.mode = mode
        self.smooth = smooth

    @property
    def mixing_weight(self):
        """
        入参:
        - 无

        方法:
        - 对logit执行Sigmoid，将w映射到(0, 1)

        出参:
        - w (torch.Tensor): 当前混合权重标量
        """
        return torch.sigmoid(self.weight_logit)

    def forward(self, inputs, targets):
        """
        入参:
        - inputs (torch.Tensor): Sigmoid后的变化概率，形状[B, 1, H, W]
        - targets (torch.Tensor): 二值变化标签，与inputs同形状

        方法:
        - 计算像素级BCE、逐样本Soft IoU和困难样本Focal损失
        - 使用固定或可学习的同一权重表达式进行融合

        出参:
        - total_loss (torch.Tensor): 标量复合损失
        """
        bce_loss = F.binary_cross_entropy(inputs, targets)
        reduction_dims = tuple(range(1, inputs.ndim))
        intersection = (inputs * targets).sum(dim=reduction_dims)
        union = (
            inputs.sum(dim=reduction_dims)
            + targets.sum(dim=reduction_dims)
            - intersection
        )
        iou_loss = 1 - (
            (intersection + self.smooth) / (union + self.smooth)
        ).mean()
        focal_loss = self.focal_loss(inputs, targets)
        weight = self.mixing_weight
        return (1 - weight) * (bce_loss + iou_loss) + weight * focal_loss
