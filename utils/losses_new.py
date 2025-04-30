import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = sigmoid(pred)*target + (1-sigmoid(pred))*(1-target)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()



class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        # batch equal to True means views all batch images as an entity and calculate loss
        # batch equal to False means calculate loss of every single image in batch and get their mean
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)


# class dice_focal_loss(nn.Module):
#
#     def __init__(self):
#         super(dice_focal_loss, self).__init__()
#         self.focal_loss = nn.BCEWithLogitsLoss()
#         self.binnary_dice = dice_loss()
#
#     def __call__(self, scores, labels):
#         diceloss = self.binnary_dice(torch.sigmoid(scores.clone()), labels)
#         foclaloss = self.focal_loss(scores.clone(), labels)
#         return [diceloss, foclaloss]

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = dice_loss()

    def forward(self, scores, labels):
        scores = scores.clone()
        dice = self.dice(torch.sigmoid(scores), labels)
        focal = self.focal(scores, labels)
        return [dice, focal]


def FCCDN_loss_without_seg(scores, labels):
    '''
        定义了一个计算损失函数的过程，专门用于二进制变化检测任务，具体是 FCCDN_loss_without_seg 函数。它接受模型预测的 scores 和真实标签 labels，并计算每个时间步的损失
    '''

    # scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
    # labels = [label.squeeze(1) if len(label.shape) > 3 else label for label in labels]
    if len(scores.shape) > 3:
        scores = scores.squeeze(1) # 此时scores的格式为(B,H,W)
    if len(labels.shape) > 3:
        labels = labels.squeeze(1) # 此时labels的格式为(B,H,W)
    """ for binary change detection task"""
    criterion_change = DiceFocalLoss(alpha=0.25, gamma=1.0)

    loss_change = criterion_change(scores, labels)
    return loss_change
