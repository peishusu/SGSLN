import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, batch=True):
        super(DiceLoss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 1e-5
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()

        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(dim=(1, 2, 3))
            j = y_pred.sum(dim=(1, 2, 3))
            intersection = (y_true * y_pred).sum(dim=(1, 2, 3))

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.to(torch.float32)
        return 1.0 - self.soft_dice_coeff(y_pred, y_true)


class BCEWithLogitsLossLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        with torch.no_grad():
            # 防止标签过拟合：将 target 从 0/1 变为更平滑的值
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.loss(input, target)


class DiceBCELossWithSmoothing(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, smoothing=0.1):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = BCEWithLogitsLossLabelSmoothing(smoothing=smoothing)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, scores, labels):
        # sigmoid for dice, raw logits for BCE
        dice_loss = self.dice(torch.sigmoid(scores), labels)
        bce_loss = self.bce(scores, labels)
        return [dice_loss * self.dice_weight, bce_loss * self.bce_weight]


def FCCDN_loss_without_seg(scores, labels):
    """
    适用于二分类变化检测任务的损失计算
    :param scores: (B, 1, H, W)
    :param labels: (B, 1, H, W)
    :return: list[dice_loss, bce_loss]
    """
    if len(scores.shape) > 3:
        scores = scores.squeeze(1)  # 变成 (B, H, W)
    if len(labels.shape) > 3:
        labels = labels.squeeze(1)

    # 初始化损失函数（带 label smoothing）
    criterion_change = DiceBCELossWithSmoothing(dice_weight=1.0, bce_weight=1.0, smoothing=0.1)

    return criterion_change(scores, labels)
