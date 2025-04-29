import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, batch=True):
        super().__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 1e-5
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(dim=(1,2,3))
            j = y_pred.sum(dim=(1,2,3))
            intersection = (y_true * y_pred).sum(dim=(1,2,3))

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def forward(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)



class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, scores, labels):
        dice_loss_val = self.dice_loss(torch.sigmoid(scores), labels)
        focal_loss_val = self.focal_loss(scores, labels)

        # 返回列表形式，与调用方期望一致
        return [
            self.dice_weight * dice_loss_val + self.focal_weight * focal_loss_val,  # 总损失
            dice_loss_val,  # dice损失分量
            focal_loss_val  # bce损失分量
        ]



def FCCDN_loss_without_seg(scores, labels):
    '''
        计算二进制变化检测任务的综合loss
    '''
    if len(scores.shape) > 3:
        scores = scores.squeeze(1)
    if len(labels.shape) > 3:
        labels = labels.squeeze(1)

    criterion = DiceFocalLoss(dice_weight=0.5, focal_weight=0.5)
    loss = criterion(scores, labels)
    return loss

