import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        """
        alpha: 控制 FP 的权重
        beta: 控制 FN 的权重（Recall 相关）
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky_index


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        """
        alpha: 控制正类权重
        gamma: 控制难样本聚焦程度
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, focal_alpha=0.75, focal_gamma=2.0):
        super(TverskyFocalLoss, self).__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, scores, labels):
        loss_tversky = self.tversky(scores, labels)
        loss_focal = self.focal(scores, labels)
        return [loss_tversky, loss_focal]


def FCCDN_loss_without_seg(scores, labels):
    """
    用于二值变化检测任务的损失函数，输出 Tversky Loss + Focal Loss
    输入:
        - scores: (B, 1, H, W) 或 (B, H, W)
        - labels: (B, 1, H, W) 或 (B, H, W)
    输出:
        - [TverskyLoss, FocalLoss]
    """
    if len(scores.shape) > 3:
        scores = scores.squeeze(1)
    if len(labels.shape) > 3:
        labels = labels.squeeze(1)

    criterion = TverskyFocalLoss(alpha=0.3, beta=0.7, focal_alpha=0.75, focal_gamma=2.0)
    loss = criterion(scores, labels)
    return loss
