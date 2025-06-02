import sys
import time
from pathlib import Path

import ipdb
import numpy as np
from torch import optim
import torch.nn as nn
import torchvision.transforms as T

from utils.path_hyperparameter import ph
import torch
from utils import data_loader

import os
import logging
import random
import wandb
# from models.Models import DPCD
from models.Models_trans import DPCD # 这个model废弃了主要是
from models.HSANet import HSANet

from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score,JaccardIndex
from utils.utils import train,val
from utils.metrics import Evaluator

from torch.utils.data import DataLoader # 继承 dataloader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    """Using prefetch_generator to accelerate data loading

    原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
    使用 prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载。

    Parameter:
        DataLoader(class): torch.utils.data.DataLoader.
    """

    # 继承自PyTorch原生 DataLoader，但通过 BackgroundGenerator 实现异步预加载：
    # 默认DataLoader需等当前batch全部消费完才会加载下一批，而 BackgroundGenerator 确保总有数据在后台加载，减少GPU等待时间。
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)  # 禁用Python哈希随机化
    np.random.seed(SEED) # NumPy随机数生成
    torch.manual_seed(SEED)  # PyTorch CPU随机种子
    torch.cuda.manual_seed(SEED)  # 当前GPU的随机种子
    torch.cuda.manual_seed_all(SEED)  # 所有GPU的随机种子（多卡时）

    # 固定卷积算法，确保结果可复现
    torch.backends.cudnn.deterministic = True  # keep convolution algorithm deterministic
    # torch.backends.cudnn.benchmark = False  # using fixed convolution algorithm to accelerate training
    # if model and input are fixed, set True to search better convolution algorithm
    torch.backends.cudnn.benchmark = True

def auto_experiment():
    # 设置随机种子，保证实验可重复性
    random_seed(SEED=ph.random_seed)
    try:
        # 负责模型的训练
        # 参数：数据集的名称
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

    dataset_name = ph.dataset_name
    # 这一块关于dataset的路径配置采用了相对路径，更加好用
    base_path = Path('../datasets')  # 或 Path.cwd() / 'datasets' / 'CD_datasets'
    train_root = base_path / dataset_name / 'train'
    val_root = base_path / dataset_name / 'val'

    train_loader = data_loader.get_loader(train_root, ph.batch_size, ph.patch_size, num_workers=2, shuffle=True,
                                          pin_memory=True)
    val_loader = data_loader.get_test_loader(val_root, ph.batch_size, ph.patch_size, num_workers=2, shuffle=False,
                                             pin_memory=True)

    # 初始化两个评估器，用于训练与验证阶段评估模型性能（比如 IoU、Precision、Recall 等）。
    Eva_train = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)

    # 4. Initialize logging
    # 自动检测可用硬件，优先使用GPU（CUDA），否则回退到CPU。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # working device
    # 设置Python日志系统，只记录INFO级别及以上的消息（如INFO/WARNING/ERROR）
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime



    # 5. Set up model, optimizer, warm_up_scheduler, learning rate scheduler, loss function and other things
    # net = DPCD()  # change detection model
    net = HSANet().cuda() # chaneg cd model

    # 原始的损失函数
    # criterion = FCCDN_loss_without_seg  # loss function 定义的一个损失函数
    # 使用 二分类的带 Logits 的交叉熵损失，适合输出未经过 Sigmoid 的二分类输出。
    criterion = nn.BCEWithLogitsLoss().cuda()

    # 将模型 net 移动到指定的设备 device 上，例如如果 device 是 "cuda"，它会将模型移动到 GPU 上，如果是 "cpu"，则模型会在 CPU 上运行。
    net = net.to(device=device)
    # 码创建了一个优化器 optimizer，使用的是 AdamW 优化算法，它是一种常用的基于梯度的优化器
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer

    # CosineAnnealingWarmRestarts: 余弦退火策略 + 周期性重启，有利于模型更好收敛。
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # model saved path
    # 创建了模型保存的路径，用于存储模型的训练检查点和最佳模型
    save_path = ph.save_path + dataset_name
    if not os.path.exists(ph.save_path):
        os.makedirs(ph.save_path)


    print(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
    ''')
    best_iou = 0.0
    # 5. Begin training
    # 比如说ph.epochs设置成250的话,epoch【0,249】
    for epoch in range(ph.epochs):
        print("/n")
        print(f" ********************* Epoch {epoch} 开始 ********************* ")  # 添加调试输出
        start_time = time.time()
        # 打印当前学习率（因为学习率会变化）。
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch} learning rate : {param_group['lr']}")
        Eva_train.reset()
        train(train_loader, Eva_train, dataset_name, net, criterion, optimizer,epoch)
        # 6. Begin evaluation
        if epoch >= ph.evaluate_epoch:

            best_iou = val(val_loader, Eva_val, dataset_name, save_path, net, epoch,best_iou)

        # 更新学习率（执行调度器）
        lr_scheduler.step()

        # 当前epoch执行所使用的时间
        epoch_time = time.time() - start_time
        print(f" ********************* Epoch {epoch} 结束: {epoch_time:.2f}秒 ********************* ")
        print("/n")

if __name__ == '__main__':
    # 程序执行入口
    auto_experiment()

