import sys
import time

import ipdb
import numpy as np
import torch.nn as nn
from torch import optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
from utils.path_hyperparameter import ph
import torch
from utils.losses import FCCDN_loss_without_seg
import os
import logging
import random
# import wandb
import swanlab
# 引入不同的model
from models.Models import DPCD
from models.FHLCDNet import FHLCDNet

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from utils.utils_wandb import train_val
from utils.dataset_process import compute_mean_std

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    """Using prefetch_generator to accelerate data loading

    原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
    使用 prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载。

    Parameter:
        DataLoader(class): torch.utils.data.DataLoader.
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# 基于训练集 loss 的早停策略：
# 如果 连续 10 个 epoch 的训练 loss 没有下降（或下降不显著），就提前终止训练。
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=1e-5):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True  # keep convolution algorithm deterministic
    # torch.backends.cudnn.benchmark = False  # using fixed convolution algorithm to accelerate training
    # if model and input are fixed, set True to search better convolution algorithm
    torch.backends.cudnn.benchmark = True

def auto_experiment():
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)


def train_net(dataset_name):
    """
    This is the workflow of training model and evaluating model,
    note that the dataset should be organized as
    :obj:`dataset_name`/`train` or `val`/`t1` or `t2` or `label`

    Parameter:
        dataset_name(str): name of dataset

    Return:
        return nothing
    """
    # 1. Create dataset, checkpoint and best model path

    # compute mean and std of train dataset to normalize train/val dataset
    t1_mean, t1_std = compute_mean_std(images_dir=f'./datasets/{dataset_name}/train/t1/')
    t2_mean, t2_std = compute_mean_std(images_dir=f'./datasets/{dataset_name}/train/t2/')

    # dataset path should be dataset_name/train or val/t1 or t2 or label
    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    train_dataset = BasicDataset(t1_images_dir=f'./datasets/{dataset_name}/train/t1/',
                                 t2_images_dir=f'./datasets/{dataset_name}/train/t2/',
                                 labels_dir=f'./datasets/{dataset_name}/train/label/',
                                 train=True, **dataset_args)
    val_dataset = BasicDataset(t1_images_dir=f'./datasets/{dataset_name}/val/t1/',
                               t2_images_dir=f'./datasets/{dataset_name}/val/t2/',
                               labels_dir=f'./datasets/{dataset_name}/val/label/',
                               train=False, **dataset_args)

    # 2. Markdown dataset size
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders

    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True,
                       # pin_memeory=True,
                       )
    train_loader = DataLoaderX(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoaderX(val_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 4. Initialize logging

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # working device
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    # using wandb to log hyperparameter, metrics and output
    # resume=allow means if the id is identical with the previous one, the run will resume
    # (anonymous=must) means the id will be anonymous
    # log_swanlab = swanlab.init(project=ph.log_wandb_project, resume='allow', anonymous='must',
    #                        settings=swanlab.Settings(start_method='thread'),
    #                        config=hyperparameter_dict)
    log_swanlab = swanlab.init(
        # 设置项目名
        project=ph.log_wandb_project,
        # 设置超参数
        config={
            "learning_rate": ph.learning_rate,
            "dataset": ph.dataset_name,
            "epochs": ph.batch_size
        }
    )

    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
        Mixed Precision: {ph.amp}
    ''')

    # 5. Set up model, optimizer, warm_up_scheduler, learning rate scheduler, loss function and other things

    # net = DPCD()  # change detection model
    net = FHLCDNet()
    net = net.to(device=device)
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer
    warmup_lr = np.arange(1e-7, ph.learning_rate,
                          (ph.learning_rate - 1e-7) / ph.warm_up_step)  # warm up learning rate
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=ph.patience,
    #                                                  factor=ph.factor)  # learning rate scheduler
    grad_scaler = torch.cuda.amp.GradScaler()  # loss scaling for amp

    # load model and optimizer
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate

    # criterion = FCCDN_loss_without_seg  # loss function
    criterion = nn.BCEWithLogitsLoss().cuda()

    best_metrics = dict.fromkeys(['best_f1score', 'best_recall','best_precision','best_IoU'], 0)  # best evaluation metrics
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task='binary', mdmc_average='global').to(device=device),
        'precision': Precision(task='binary', mdmc_average='global').to(device=device),
        'recall': Recall(task='binary', mdmc_average='global').to(device=device),
        'f1score': F1Score(task='binary', mdmc_average='global').to(device=device)
    })  # metrics calculator

    to_pilimg = T.ToPILImage()  # convert to PIL image to log in wandb

    # model saved path
    checkpoint_path = f'./{dataset_name}_checkpoint/'
    best_f1score_model_path = f'./{dataset_name}_best_f1score_model/'


    non_improved_epoch = 0  # adjust learning rate when non_improved_epoch equal to patience

    early_stopper = EarlyStopping(patience=ph.patience)

    # 5. Begin training


    for epoch in range(ph.epochs):

        log_swanlab, net, optimizer, grad_scaler, total_step, lr, train_loss = \
            train_val(
                mode='train', dataset_name=dataset_name,
                dataloader=train_loader, device=device, log_swanlab=log_swanlab, net=net,
                optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                warmup_lr=warmup_lr, grad_scaler=grad_scaler
            )

        # 判断是否触发早停
        early_stopper(train_loss)
        if early_stopper.early_stop:
            logging.info(f"Early stopping triggered on train loss at epoch {epoch}.")
            break

        # 6. 开始  val
        # starting validation from evaluate epoch to minimize time
        if epoch >= ph.evaluate_epoch:
            with torch.no_grad():
                log_swanlab, net, optimizer, total_step, lr, best_metrics, non_improved_epoch = \
                    train_val(
                        mode='val', dataset_name=dataset_name,
                        dataloader=val_loader, device=device, log_swanlab=log_swanlab, net=net,
                        optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                        best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_f1score_model_path=best_f1score_model_path,
                        non_improved_epoch=non_improved_epoch
                    )

    log_swanlab.finish()





if __name__ == '__main__':

    auto_experiment()
