import torch
import torch.nn as nn


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


class dice_focal_loss(nn.Module):

    def __init__(self):
        super(dice_focal_loss, self).__init__()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels):
        diceloss = self.binnary_dice(torch.sigmoid(scores.clone()), labels)
        foclaloss = self.focal_loss(scores.clone(), labels)
        return [diceloss, foclaloss]


def FCCDN_loss_without_seg(scores, labels):
    '''
        定义了一个计算损失函数的过程，专门用于二进制变化检测任务，具体是 FCCDN_loss_without_seg 函数。它接受模型预测的 scores 和真实标签 labels，并计算每个时间步的损失
    '''
    # scores = change_pred
    # labels = binary_cd_labels
    # 目的是确保 scores 和 labels 的形状一致
    scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
    labels = [label.squeeze(1) if len(label.shape) > 3 else label for label in labels]
    # if len(scores.shape) > 3:
    #     scores = scores.squeeze(1)
    # if len(labels.shape) > 3:
    #     labels = labels.squeeze(1)
    """ for binary change detection task"""
    # 定义了一个损失函数，dice_focal_loss 是一种结合了 Dice 系数和焦点损失的损失函数。Dice 损失用于衡量预测与真实标签之间的相似度，Focal 损失主要解决类别不平衡的问题，特别适用于变化检测等任务。
    criterion_change = dice_focal_loss()

    # change loss
    # 计算第一类变化检测的损失。scores[0] 和 labels[0] 分别代表预测结果和真实标签的第一个元素
    loss_change = criterion_change(scores[0], labels[0])
    # 计算第二类的损失，scores[1] 和 labels[1] 代表预测结果和真实标签的第二个元素。
    loss_seg1 = criterion_change(scores[1], labels[1])
    # 计算第三类的损失，scores[2] 和 labels[1] 仍然代表预测结果和第一个时间戳的真实标签。这里 labels[1] 被重复使用来进行与 scores[2] 的比较。
    loss_seg2 = criterion_change(scores[2], labels[1])

    for i in range(len(loss_change)):
        loss_change[i] += 0.2 * (loss_seg1[i] + loss_seg2[i])

    return loss_change
