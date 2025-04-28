from pathlib import Path
import time
import numpy as np
import torch.nn.functional as F
from utils.path_hyperparameter import ph
import torch
import logging
from tqdm import tqdm
import wandb
import ipdb
from PIL import Image



def save_model(model, path, epoch, mode, optimizer=None):
    """
        在val的过程中去保存模型
    """
    # 自动创建目标路径（若不存在），parents=True 允许创建多级目录。
    Path(path).mkdir(parents=True,
                     exist_ok=True)  # create a dictionary
    #  获取当前时间
    localtime = time.asctime(time.localtime(time.time()))

    if mode == 'checkpoint': # 表示中间状态保存：模型权重、优化器状态
        state_dict = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict,
                   str(path + f'checkpoint_epoch{epoch}_{localtime}.pth'))

    else: # 表示保存性能最好的模型，只保存：模型权重
        torch.save(model.state_dict(), str(path + f'best_{mode}_epoch{epoch}_{localtime}.pth'))
    logging.info(f'best {mode} model {epoch} saved at {localtime}!')


def train_val(
        mode, dataset_name,
        dataloader, device, log_wandb, net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_f1score_model_path=None, best_loss_model_path=None, non_improved_epoch=None
):
    assert mode in ['train', 'val'], 'mode should be train, val'
    epoch_loss = 0 # 初始化累计损失
    # Begin Training/Evaluating
    if mode == 'train':
        net.train() # .train() 是用来切换成「训练模式」，确保 Dropout 和 BatchNorm 行为正确。
    else:
        net.eval()
    logging.info(f'SET model mode to {mode}!')
    # 这是一个用来记录已经遍历了多少个样本的计数器。虽然这个变量名叫 batch_iter，但它其实在这里记录的是「已经遍历了多少个样本」而不是 batch 数。
    # 如果 batch_size = 8，那每过一个 batch，这个 batch_iter 就会加上 8。
    batch_iter = 0

    # 是你传入的训练或验证集的 DataLoader， 是一个进度条库，用来美化训练输出，让你看到处理进度
    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=n_iter) # 从 0 到 n_iter-1 的整数范围内随机抽取一个整数，并将结果赋值给 sample_batch


    # 在当前epoch中，遍历所有的图片，进行训练
    for i, (batch_img1, batch_img2, labels, name) in enumerate(tbar):
        # batch_img1和batch_img2的格式是:(B,3,H,W) 这里就是（B,3,H,W） H和W随着输入图片的尺寸大小改变而改变
        # labels的格式是(B,2,H,W)
        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + ph.batch_size))
        # 一开始是0，总计的加载图片的数量 = 已有的 + 每批次的图片数量
        batch_iter = batch_iter + ph.batch_size
        total_step += 1

        # 在训练前，清除梯度
        if mode == 'train':
            optimizer.zero_grad()
            # warm up
            if total_step < ph.warm_up_step:
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr[total_step]

        batch_img1 = batch_img1.float().to(device)
        batch_img2 = batch_img2.float().to(device)
        labels = labels.float().to(device)



        if mode == 'train':
            # using amp
            with torch.cuda.amp.autocast():
                # preds格式为(B,1,H,W),不再是tuple元组了
                preds = net(batch_img1, batch_img2) # 前向传播
                loss = criterion(preds, labels) # labels的格式为(B,h,w)
            cd_loss = sum(loss)
            grad_scaler.scale(cd_loss).backward() # 反向传播
            torch.nn.utils.clip_grad_norm_(net.parameters(), 20, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            preds = net(batch_img1, batch_img2)
            loss = criterion(preds, labels)
            cd_loss = sum(loss)

        epoch_loss += cd_loss
        # preds从元组 -> (1B,2,H，W)
        preds = torch.sigmoid(preds)

        # log the t1_img, t2_img, pred and label
        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=batch_img1.shape[0])
            # ipdb.set_trace()
            t1_images_dir = Path(f'../datasets/{dataset_name}/{mode}/t1/')
            t2_images_dir = Path(f'../datasets/{dataset_name}/{mode}/t2/')
            labels_dir = Path(f'../datasets/{dataset_name}/{mode}/label/')
            t1_img_log = Image.open(list(t1_images_dir.glob(name[sample_index] + '.*'))[0])
            t2_img_log = Image.open(list(t2_images_dir.glob(name[sample_index] + '.*'))[0])
            label_log = Image.open(list(labels_dir.glob(name[sample_index] + '.*'))[0])
            pred_log = torch.round(preds[sample_index]).cpu().clone().float()
            # pred_log[pred_log >= 0.5] = 1
            # pred_log[pred_log < 0.5] = 0
            # pred_log = pred_log.float()

        preds = preds.float()  #格式为(1,2,h,w)
        labels = labels.int().unsqueeze(1) # labels从(1,512,512)->变成 (1,1,512,512)
        batch_metrics = metric_collection.forward(preds, labels)  # compute metric

        # log loss and metric
        log_wandb.log({
            f'{mode} loss': cd_loss,
            f'{mode} accuracy': batch_metrics['accuracy'],
            f'{mode} precision': batch_metrics['precision'],
            f'{mode} recall': batch_metrics['recall'],
            f'{mode} f1score': batch_metrics['f1score'],
            'learning rate': optimizer.param_groups[0]['lr'],
            f'{mode} loss_dice': loss[0],
            f'{mode} loss_bce': loss[1],
            'step': total_step,
            'epoch': epoch
        })

        # if torch.isnan(loss[0]):
        #     torch.save(preds, f'pred_{total_step}.pth')
        #     torch.save(labels, f'label_{total_step}.pth')

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    # epoch_metric是一个dict,key是accuray/recall/f1score/precision,value是其对应的值
    epoch_metrics = metric_collection.compute()  # compute epoch metric
    epoch_loss /= n_iter  # n_iter代表图片的总数量，epoch_loss代表平均损失多少


    #  # 将每个指标按mode（如train/val）和epoch分类记录到W&B仪表盘。
    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k],
                       'epoch': epoch})  # log epoch metric
    metric_collection.reset() # 清空所有累积的中间统计量（如TP/FP/TN/FN），为下个epoch做准备。

    log_wandb.log({f'epoch_{mode}_loss': epoch_loss,
                   'epoch': epoch})  # log epoch loss

    # 在wandb-summary.json文件中打印出关于原始图片t1\t2\label以及预测标签pred的相关信息
    log_wandb.log({
        f'{mode} t1_images': wandb.Image(t1_img_log),
        f'{mode} t2_images': wandb.Image(t2_img_log),
        f'{mode} masks': {
            'label': wandb.Image(label_log),
            'pred': wandb.Image(to_pilimg(pred_log)),
        },
        'epoch': epoch
    })  # log the t1_img, t2_img, pred and label

    # save best model and adjust learning rate according to learning rate scheduler
    if mode == 'val':
        if epoch_metrics['f1score'] > best_metrics['best_f1score']:
            non_improved_epoch = 0
            best_metrics['best_f1score'] = epoch_metrics['f1score']
            if ph.save_best_model:
                save_model(net, best_f1score_model_path, epoch, 'f1score')
        elif epoch_loss < best_metrics['lowest loss']:
            best_metrics['lowest loss'] = epoch_loss
            if ph.save_best_model:
                save_model(net, best_loss_model_path, epoch, 'loss')
        else:
            non_improved_epoch += 1
            if non_improved_epoch == ph.patience:
                lr *= ph.factor
                for g in optimizer.param_groups:
                    g['lr'] = lr
                non_improved_epoch = 0

        # save checkpoint every specified interval,推荐每 10 epoch 保存一次，便于中断恢复。(所以：ph.save_intervalv == 10)
        if (epoch + 1) % ph.save_interval == 0 and ph.save_checkpoint:
            save_model(net, checkpoint_path, epoch, 'checkpoint', optimizer=optimizer)

    if mode == 'train':
        return log_wandb, net, optimizer, grad_scaler, total_step, lr
    elif mode == 'val':
        return log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch
    else:
        raise NameError('mode should be train or val')
