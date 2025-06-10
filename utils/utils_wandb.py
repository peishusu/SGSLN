from pathlib import Path
import time
import numpy as np
import torch.nn.functional as F
from utils.path_hyperparameter import ph
import torch
import logging
from tqdm import tqdm
# import wandb
import swanlab
import ipdb
from PIL import Image


def save_model(model, path, epoch, mode, optimizer=None):
    # mode should be checkpoint or loss or f1score
    Path(path).mkdir(parents=True,
                     exist_ok=True)  # create a dictionary
    # ipdb.set_trace()
    localtime = time.asctime(time.localtime(time.time()))
    if mode == 'checkpoint':
        state_dict = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict,
                   str(path + f'checkpoint_epoch{epoch}_{localtime}.pth'))
    else:
        torch.save(model.state_dict(), str(path + f'best_{mode}_epoch{epoch}_{localtime}.pth'))
    logging.info(f'best {mode} model {epoch} saved at {localtime}!')


def train_val(
        mode, dataset_name,
        dataloader, device, log_swanlab, net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_f1score_model_path=None, non_improved_epoch=None
):
    assert mode in ['train', 'val'], 'mode should be train, val'
    epoch_loss = 0
    # Begin Training/Evaluating
    if mode == 'train':
        net.train()
    else:
        net.eval()
    logging.info(f'SET model mode to {mode}!')
    batch_iter = 0

    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=n_iter)

    for i, (batch_img1, batch_img2, labels, name) in enumerate(tbar):

        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + ph.batch_size))
        batch_iter = batch_iter + ph.batch_size
        total_step += 1

        # Zero the gradient if train
        if mode == 'train':
            optimizer.zero_grad()
            '''
                Warmup 是在调度器之前手动设定的。
                通常在 warmup 阶段结束后，再正式使用 lr_scheduler.step() 来控制学习率。
                所以 两者不会冲突，只是先后起作用。
            '''
            if total_step < ph.warm_up_step:
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr[total_step]

        batch_img1 = batch_img1.float().to(device)
        batch_img2 = batch_img2.float().to(device)
        labels = labels.float().to(device)

        if mode == 'train':
            # using amp
            with torch.cuda.amp.autocast():
                preds = net(batch_img1, batch_img2)  # preds的格式为b,1,h,w
                loss = criterion(preds, labels)
            cd_loss = loss
            # grad_scaler与 autocast() 配合，在训练阶段包装反向传播流程：控制反向传播的稳定性；
            grad_scaler.scale(cd_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 20, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            preds = net(batch_img1, batch_img2)
            # loss = criterion(preds, (labels, seg_label))
            loss = criterion(preds, labels)
            cd_loss = loss
        # epoch_loss记录当前所有的batch_loss之和
        epoch_loss += cd_loss

        # log the t1_img, t2_img, pred and label
        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=batch_img1.shape[0])
            # ipdb.set_trace()
            t1_images_dir = Path(f'../autodl-tmp/datasets/{dataset_name}/{mode}/t1/')
            t2_images_dir = Path(f'../autodl-tmp/datasets/{dataset_name}/{mode}/t2/')
            labels_dir = Path(f'../autodl-tmp/datasets/{dataset_name}/{mode}/label/')
            t1_img_log = Image.open(list(t1_images_dir.glob(name[sample_index] + '.*'))[0])
            t2_img_log = Image.open(list(t2_images_dir.glob(name[sample_index] + '.*'))[0])
            label_log = Image.open(list(labels_dir.glob(name[sample_index] + '.*'))[0])
            pred_log = torch.round(preds[sample_index]).cpu().clone().float()

        preds = torch.sigmoid(preds).float()
        labels = labels.int()
        batch_metrics = metric_collection.forward(preds, labels)  # 计算当前batch的各项指标

        del batch_img1, batch_img2, labels

    epoch_metrics = metric_collection.compute()  # 计算当前epoch的各项指标

    epoch_loss /= n_iter

    # 记录各项指标
    # 替换 IoU 为 f1 / (2 - f1),之前的关于iou的计算是存在问题的
    f1_score = epoch_metrics['f1score']
    iou_from_f1 = f1_score / (2 - f1_score) if (2 - f1_score) != 0 else 0.0
    epoch_metrics['IoU'] = iou_from_f1
    # 记录当前epoch的学习率
    cur_lr = optimizer.param_groups[0]['lr']
    log_swanlab.log({f'curEpoch_learning rate': cur_lr, 'epoch': epoch})
    for k in epoch_metrics.keys():
        log_swanlab.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k], 'epoch': epoch})  # log epoch metric

    metric_collection.reset()  # 清空所有累积的中间统计量（如TP/FP/TN/FN），为下个epoch做准备。
    log_swanlab.log({f'epoch_{mode}_loss': epoch_loss, 'epoch': epoch})  # log epoch loss

    # 将当前epoch的各项指标打印下
    metrics_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in epoch_metrics.items()])
    # 添加 loss\lr 到字符串中
    print(f"[Epoch {epoch}][{mode.upper()}] loss: {epoch_loss:.4f} | {metrics_str} | lr: {cur_lr:.6f}")


    # 记录图片
    log_swanlab.log({
        f'{mode} t1_images': swanlab.Image(t1_img_log),
        f'{mode} t2_images': swanlab.Image(t2_img_log),
        f'{mode} masks': {
            'label': swanlab.Image(label_log),
            'pred': swanlab.Image(to_pilimg(pred_log)),
        },
        'epoch': epoch
    })  # log the t1_img, t2_img, pred and label

    # save best model and adjust learning rate according to learning rate scheduler
    if mode == 'val':
        f1_score = epoch_metrics['f1score']
        iou_from_f1 = f1_score / (2 - f1_score) if (2 - f1_score) != 0 else 0.0
        epoch_metrics['IoU'] = iou_from_f1

        current_f1 = epoch_metrics['f1score']
        best_f1 = best_metrics.get('best_f1score', 0)

        if current_f1 >= best_f1:
            # 更新最佳指标
            best_metrics['best_precision'] = epoch_metrics['precision']
            best_metrics['best_recall'] = epoch_metrics['recall']
            best_metrics['best_f1score'] = current_f1
            best_metrics['best_IoU'] = epoch_metrics['IoU']
            best_metrics['best_accuracy'] = epoch_metrics['accuracy']

            non_improved_epoch = 0

            if ph.save_best_model:
                save_model(net, best_f1score_model_path, epoch, 'multi_metric')

            # 记录指标到 W&B
            log_swanlab.log({
                f'{mode}_{epoch}_best_accuracy': epoch_metrics['accuracy'],
                f'{mode}_{epoch}_best_precision': epoch_metrics['precision'],
                f'{mode}_{epoch}_best_recall': epoch_metrics['recall'],
                f'{mode}_{epoch}_best_f1score': current_f1,
                f'{mode}_{epoch}_best_IoU': epoch_metrics['IoU']
            })
        # else: # 这一快是手动调节学习率
        #     non_improved_epoch += 1
        #     if non_improved_epoch == ph.patience:
        #         lr *= ph.factor
        #         for g in optimizer.param_groups:
        #             g['lr'] = lr
        #         non_improved_epoch = 0

        # Save checkpoint every few epochs
        if (epoch + 1) % ph.save_interval == 0 and ph.save_checkpoint:
            save_model(net, checkpoint_path, epoch, 'checkpoint', optimizer=optimizer)

    if mode == 'train':
        return log_swanlab, net, optimizer, grad_scaler, total_step, lr
    elif mode == 'val':
        val_f1 = epoch_metrics['f1score'].item()
        return log_swanlab, net, optimizer, total_step, lr, best_metrics, non_improved_epoch, val_f1
    else:
        raise NameError('mode should be train or val')