from pathlib import Path
import time
import numpy as np
import torch.nn.functional as F
from utils.path_hyperparameter import ph
import utils.visualization as visual
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


def train( train_loader, Eva_train, data_name,net, criterion, optimizer, epoch):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    epoch_loss = 0
    net.train(True)
    print(f'SET model mode to train!')
    # 这是一个用来记录已经遍历了多少个样本的计数器。虽然这个变量名叫 batch_iter，但它其实在这里记录的是「已经遍历了多少个样本」而不是 batch 数。
    # 如果 batch_size = 8，那每过一个 batch，这个 batch_iter 就会加上 8。
    batch_iter = 0

    # 是你传入的训练或验证集的 DataLoader， 是一个进度条库，用来美化训练输出，让你看到处理进度
    tbar = tqdm(train_loader)
    n_iter = len(train_loader)
    sample_batch = np.random.randint(low=0, high=n_iter) # 从 0 到 n_iter-1 的整数范围内随机抽取一个整数，并将结果赋值给 sample_batch

    length = 0
    # 在当前epoch中，遍历所有的图片，进行训练
    for i, (batch_img1, batch_img2, labels, name) in enumerate(tbar):
        # batch_img1和batch_img2的格式是:(B,3,H,W) 这里就是（B,3,H,W） H和W随着输入图片的尺寸大小改变而改变
        # labels的格式是(B,1,H,W)
        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + ph.batch_size))
        # 一开始是0，总计的加载图片的数量 = 已有的 + 每批次的图片数量
        batch_iter = batch_iter + ph.batch_size

        batch_img1 = batch_img1.cuda()  # (B,3,H,W)
        batch_img2 = batch_img2.cuda()  # (B,3,H,W)
        labels = labels.cuda()  # (B,1,H,W)
        # 在训练前，清除梯度
        optimizer.zero_grad()
        # preds格式为(B,1,H,W),不再是tuple元组了
        preds = net(batch_img1, batch_img2) # 前向传播
        loss = criterion(preds[0], labels) + criterion(preds[1], labels)
        # ---- loss function ----
        loss.backward() # 反向传播
        optimizer.step()
        epoch_loss += loss.item()

        output = F.sigmoid(preds[1])
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = labels.cpu().numpy().astype(int)

        Eva_train.add_batch(target, pred)

        length += 1
        #TODO：到时候坑定需要保存这个的
        # if i == sample_batch:
        #     sample_index = np.random.randint(low=0, high=batch_img1.shape[0])
        #     # ipdb.set_trace()
        #     # TODO：路径这一块还得修改
        #     t1_images_dir = Path(f'../datasets/{data_name}/{train}/t1/')
        #     t2_images_dir = Path(f'../datasets/{data_name}/{train}/t2/')
        #     labels_dir = Path(f'../datasets/{data_name}/{train}/label/')
        #     t1_img_log = Image.open(list(t1_images_dir.glob(name[sample_index] + '.*'))[0])
        #     t2_img_log = Image.open(list(t2_images_dir.glob(name[sample_index] + '.*'))[0])
        #     label_log = Image.open(list(labels_dir.glob(name[sample_index] + '.*'))[0])
        #     pred_log = torch.round(preds[sample_index]).cpu().clone().float()

    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
        epoch, epoch, train_loss, IoU, Pre, Recall, F1))



def val(val_loader, Eva_val, data_name,save_path, net, epoch,best_iou):
    vis = visual.Visualization()
    vis.create_summary(data_name)


    net.train(False)
    net.eval()
    print(f'SET model mode to val!')
    # 这是一个用来记录已经遍历了多少个样本的计数器。虽然这个变量名叫 batch_iter，但它其实在这里记录的是「已经遍历了多少个样本」而不是 batch 数。
    # 如果 batch_size = 8，那每过一个 batch，这个 batch_iter 就会加上 8。
    batch_iter = 0

    # 是你传入的训练或验证集的 DataLoader， 是一个进度条库，用来美化训练输出，让你看到处理进度
    tbar = tqdm(val_loader)
    n_iter = len(val_loader)
    sample_batch = np.random.randint(low=0, high=n_iter) # 从 0 到 n_iter-1 的整数范围内随机抽取一个整数，并将结果赋值给 sample_batch

    # 在当前epoch中，遍历所有的图片，进行训练
    for i, (batch_img1, batch_img2, labels, name) in enumerate(tbar):
        with torch.no_grad():
            # batch_img1和batch_img2的格式是:(B,3,H,W) 这里就是（B,3,H,W） H和W随着输入图片的尺寸大小改变而改变
            # labels的格式是(B,1,H,W)
            tbar.set_description(
                "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + ph.batch_size))
            # 一开始是0，总计的加载图片的数量 = 已有的 + 每批次的图片数量
            batch_iter = batch_iter + ph.batch_size

            batch_img1 = batch_img1.cuda()  # (B,3,H,W)
            batch_img2 = batch_img2.cuda()  # (B,3,H,W)
            labels = labels.cuda()  # (B,1,H,W)

            # preds格式为(B,1,H,W),不再是tuple元组了
            preds = net(batch_img1, batch_img2)[1] # 前向传播
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = labels.cpu().numpy().astype(int)
            Eva_val.add_batch(target, pred)
            #TODO：这一块实际上不需要的，只需要在test过程中进行保存即可
            # if i == sample_batch:
            #     sample_index = np.random.randint(low=0, high=batch_img1.shape[0])
            #     # ipdb.set_trace()
            #     # TODO：路径这一块还得修改
            #     t1_images_dir = Path(f'../datasets/{data_name}/{train}/t1/')
            #     t2_images_dir = Path(f'../datasets/{data_name}/{train}/t2/')
            #     labels_dir = Path(f'../datasets/{data_name}/{train}/label/')
            #     t1_img_log = Image.open(list(t1_images_dir.glob(name[sample_index] + '.*'))[0])
            #     t2_img_log = Image.open(list(t2_images_dir.glob(name[sample_index] + '.*'))[0])
            #     label_log = Image.open(list(labels_dir.glob(name[sample_index] + '.*'))[0])
            #     pred_log = torch.round(preds[sample_index]).cpu().clone().float()
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        torch.save(best_net, save_path + '_best_iou.pth')
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    vis.close_summary()
    return best_iou



def train_val(
        dataset_name,mode,
        dataloader, device, log_wandb, net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch, Eva_train, Eva_val,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_model_path=None, best_loss_model_path=None, non_improved_epoch=None
):
    assert mode in ['train', 'val'], 'mode should be train, val'
    epoch_loss = 0  # 初始化累计损失
    # Begin Training/Evaluating
    if mode == 'train':
        net.train()  # .train() 是用来切换成「训练模式」，确保 Dropout 和 BatchNorm 行为正确。
    else:
        net.eval()
    print(f'SET model mode to {mode}!')
    # 这是一个用来记录已经遍历了多少个样本的计数器。虽然这个变量名叫 batch_iter，但它其实在这里记录的是「已经遍历了多少个样本」而不是 batch 数。
    # 如果 batch_size = 8，那每过一个 batch，这个 batch_iter 就会加上 8。
    batch_iter = 0

    # 是你传入的训练或验证集的 DataLoader， 是一个进度条库，用来美化训练输出，让你看到处理进度
    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=n_iter)  # 从 0 到 n_iter-1 的整数范围内随机抽取一个整数，并将结果赋值给 sample_batch

    # 在当前epoch中，遍历所有的图片，进行训练
    for i, (batch_img1, batch_img2, labels, name) in enumerate(tbar):
        # batch_img1和batch_img2的格式是:(B,3,H,W) 这里就是（B,3,H,W） H和W随着输入图片的尺寸大小改变而改变
        # labels的格式是(B,1,H,W)
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

            # preds格式为(B,1,H,W),不再是tuple元组了
            preds = net(batch_img1, batch_img2)  # 前向传播
            # loss = criterion(preds, labels) # labels的格式为(B,h,w)
            loss = criterion(preds[0], labels) + criterion(preds[1], labels)

            loss.backward()  # 反向传播
            optimizer.step()
            output = F.sigmoid(preds[1])
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = labels.cpu().numpy().astype(int)

            Eva_train.add_batch(target, pred)
        else:
            preds = net(batch_img1, batch_img2)
            loss = criterion(preds, labels)
            # cd_loss = loss[0]

        epoch_loss += loss
        # preds从元组 -> (1B,1,H，W)
        # preds_bin = (torch.sigmoid(preds) > threshold)
        preds = torch.sigmoid(preds)

        # log the t1_img, t2_img, pred and label
        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=batch_img1.shape[0])
            # ipdb.set_trace()
            # TODO：路径这一块还得修改
            t1_images_dir = Path(f'../datasets/{dataset_name}/{mode}/t1/')
            t2_images_dir = Path(f'../datasets/{dataset_name}/{mode}/t2/')
            labels_dir = Path(f'../datasets/{dataset_name}/{mode}/label/')
            t1_img_log = Image.open(list(t1_images_dir.glob(name[sample_index] + '.*'))[0])
            t2_img_log = Image.open(list(t2_images_dir.glob(name[sample_index] + '.*'))[0])
            label_log = Image.open(list(labels_dir.glob(name[sample_index] + '.*'))[0])
            pred_log = torch.round(preds[sample_index]).cpu().clone().float()

        preds = preds.float()  # 格式为(1,1,h,w)
        labels = labels.int().unsqueeze(1)  # labels从(B,512,512)->变成 (B,1,512,512)
        # batch_metrices指的是当前batch的指标
        batch_metrics = metric_collection.forward(preds, labels)  # compute metric

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    # epoch_metric是一个dict,key是accuray/recall/f1score/precision,value是其对应的值
    # epoch_metric 指的是整epoch所有batch的累积平均指标
    epoch_metrics = metric_collection.compute()  # compute epoch metric
    epoch_loss /= n_iter  # n_iter代表图片的总数量，epoch_loss代表平均损失多少

    # 替换 IoU 为 f1 / (2 - f1),之前的关于iou的计算是存在问题的
    f1_score = epoch_metrics['f1score']
    iou_from_f1 = f1_score / (2 - f1_score) if (2 - f1_score) != 0 else 0.0
    epoch_metrics['IoU'] = iou_from_f1
    #  # 将每个指标按mode（如train/val）和epoch分类记录到W&B仪表盘。
    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k],
                       'epoch': epoch})  # log epoch metric
    metric_collection.reset()  # 清空所有累积的中间统计量（如TP/FP/TN/FN），为下个epoch做准备。

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

    if mode == 'val':
        f1_score = epoch_metrics['f1score']
        iou_from_f1 = f1_score / (2 - f1_score) if (2 - f1_score) != 0 else 0.0
        epoch_metrics['IoU'] = iou_from_f1

        improved_metrics = 0
        current_metrics = {
            'precision': epoch_metrics['precision'],
            'recall': epoch_metrics['recall'],
            'f1score': epoch_metrics['f1score'],
            'IoU': epoch_metrics['IoU']  # 使用新的 IoU
        }

        for metric_name, current_value in current_metrics.items():
            best_value = best_metrics.get(f'best_{metric_name}', 0)
            if current_value > best_value:
                improved_metrics += 1

        if improved_metrics >= 3:
            # 更新最佳指标
            for metric_name, current_value in current_metrics.items():
                best_metrics[f'best_{metric_name}'] = current_value

            non_improved_epoch = 0

            if ph.save_best_model:
                save_model(net, best_model_path, epoch, 'multi_metric')

            # 记录指标到 W&B
            log_wandb.log({
                f'{mode}_{epoch}_best_precision': current_metrics['precision'],
                f'{mode}_{epoch}_best_recall': current_metrics['recall'],
                f'{mode}_{epoch}_best_f1score': current_metrics['f1score'],
                f'{mode}_{epoch}_best_IoU': current_metrics['IoU']
            })
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

    # 返回最终的结果
    if mode == 'train':
        return log_wandb, net, optimizer, grad_scaler, total_step, lr
    elif mode == 'val':
        return log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch
    else:
        raise NameError('mode should be train or val')


