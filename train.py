import sys
import time

import ipdb
import numpy as np
from torch import optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
from utils.path_hyperparameter import ph
import torch
from utils.losses import FCCDN_loss_without_seg
# from utils.losses_new import FCCDN_loss_without_seg

import os
import logging
import random
import wandb
# from models.Models import DPCD
from models.Models_trans import DPCD

from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score,JaccardIndex
from utils.utils import train_val
from utils.dataset_process import compute_mean_std
from utils.dataset_process import image_shuffle, split_image
import onnx
import onnx.utils
import onnx.version_converter
import netron
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
    # 1. Create dataset, checkpoint and best model path

    # 分别计算 t1文件中所有图像的均值、标准差
    t1_mean, t1_std = compute_mean_std(images_dir=f'../datasets/{dataset_name}/train/t1/')
    t2_mean, t2_std = compute_mean_std(images_dir=f'../datasets/{dataset_name}/train/t2/')


    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)

    train_dataset = BasicDataset(t1_images_dir=f'../datasets/{dataset_name}/train/t1/',
                                 t2_images_dir=f'../datasets/{dataset_name}/train/t2/',
                                 labels_dir=f'../datasets/{dataset_name}/train/label/',
                                 train=True, **dataset_args)
    val_dataset = BasicDataset(t1_images_dir=f'../datasets/{dataset_name}/val/t1/',
                               t2_images_dir=f'../datasets/{dataset_name}/val/t2/',
                               labels_dir=f'../datasets/{dataset_name}/val/label/',
                               train=False, **dataset_args)

    # 2. 通过调用 BasicDataset 类中的 __len__() 方法来获取数据集的大小，即训练集和验证集的样本数量。
    # train_dataset 和 val_dataset 作为 BasicDataset 类的实例，调用 len() 函数时会自动触发 __len__() 方法
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders

    loader_args = dict(num_workers=8,  # 使用8个子进程并行加载数据
                       prefetch_factor=5, # 每个worker预加载5个batch
                       persistent_workers=True,  # 训练过程中保持Worker进程存活（默认每个epoch后销毁重建），避免重复创建进程的开销。
                       # pin_memeory=True,
                       )
    # shuffle=True 表示个epoch打乱数据顺序，防止模型记忆样本顺序；drop_last=False 不丢弃最后一个不完整的batch；
    train_loader = DataLoaderX(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoaderX(val_dataset, shuffle=False, drop_last=False, batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 4. Initialize logging
    # 自动检测可用硬件，优先使用GPU（CUDA），否则回退到CPU。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # working device
    # 设置Python日志系统，只记录INFO级别及以上的消息（如INFO/WARNING/ERROR）
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    # using wandb to log hyperparameter, metrics and output
    # resume=allow means if the id is identical with the previous one, the run will resume
    # (anonymous=must) means the id will be anonymous
    log_wandb = wandb.init(project=ph.log_wandb_project, resume='allow', anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=hyperparameter_dict)
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

    net = DPCD()  # change detection model
    # 将模型 net 移动到指定的设备 device 上，例如如果 device 是 "cuda"，它会将模型移动到 GPU 上，如果是 "cpu"，则模型会在 CPU 上运行。
    net = net.to(device=device)
    # 码创建了一个优化器 optimizer，使用的是 AdamW 优化算法，它是一种常用的基于梯度的优化器
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer
    # 用于生成一个学习率预热（warm-up）数组，主要目的是在训练开始时逐步增大学习率，避免初期训练时学习率过高导致的训练不稳定。
    warmup_lr = np.arange(1e-7, ph.learning_rate,
                          (ph.learning_rate - 1e-7) / ph.warm_up_step)  # warm up learning rate
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=ph.patience,factor=ph.factor)  # learning rate scheduler
    #是用于启用自动混合精度（AMP）训练的梯度缩放器。AMP 是 PyTorch 提供的一种加速训练的技术，通过使用半精度（16位）浮点数来减少内存使用并加速训练。为了保持训练的稳定性，需要对梯度进行缩放。
    grad_scaler = torch.cuda.amp.GradScaler()  # loss scaling for amp

    # load model and optimizer
    # 主要用于加载预训练模型的权重，并恢复模型的优化器状态
    if ph.load:
        # 加载存储的模型检查点（checkpoint），并将其存储到 checkpoint 变量中
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True


    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate 学习率


    criterion = FCCDN_loss_without_seg  # loss function 定义的一个损失函数

    # 初始化用于模型评估的最佳指标和计算指标的集合，这里['best_f1score', 'lowest loss'] 是字典的键，0 是每个键的默认值。
    best_metrics = dict.fromkeys(['best_f1score','best_precision','best_recall','best_IoU'], 0)  # best evaluation metrics
    # 一个用来组织和计算多种评估指标的工具。在 PyTorch 中，MetricCollection 是一种便捷方式，它允许我们在训练过程中同时计算多个不同的评估指标。
    metric_collection = MetricCollection({
        'accuracy': Accuracy().to(device=device),
        'precision': Precision().to(device=device),
        'recall': Recall().to(device=device),
        'f1score': F1Score().to(device=device),
        'IoU': JaccardIndex(num_classes=2, task="binary").to(device=device)

    })  # metrics calculator

    #  PyTorch 中的一个工具函数，它将 PyTorch Tensor 转换为 PIL 图像对象。PIL 图像是 Python Imaging Library（或 Pillow）使用的一种图像格式，可以用于显示或保存图像
    to_pilimg = T.ToPILImage()  # convert to PIL image to log in wandb

    # model saved path
    # 创建了模型保存的路径，用于存储模型的训练检查点和最佳模型
    checkpoint_path = f'./{dataset_name}_checkpoint/'
    best_model_path = f'./{dataset_name}_best_model/'
    best_loss_model_path = f'./{dataset_name}_best_loss_model/'

    # 用于跟踪连续多少个 epoch 中模型的性能（例如，F1 分数或损失）没有改善。如果模型在一定的 patience 轮次内没有提升（即 F1 分数没有变好或损失没有降低），可以采取一些操作，比如降低学习率或提前停止训练。
    non_improved_epoch = 0  # adjust learning rate when non_improved_epoch equal to patience

    # 5. Begin training
    # 比如说ph.epochs设置成250的话,epoch【0,249】
    for epoch in range(ph.epochs):
        # 输出后的内容
        log_wandb, net, optimizer, grad_scaler, total_step, lr = \
            train_val(
                mode='train', dataset_name=dataset_name,
                dataloader=train_loader, device=device, log_wandb=log_wandb, net=net,
                optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                warmup_lr=warmup_lr, grad_scaler=grad_scaler
            )
        # 6. Begin evaluation

        # starting validation from evaluate epoch to minimize time
        # 验证过程（只有当 epoch 达到指定数量后才执行），前几轮训练模型还不稳定，为了节省时间，不做验证。
        if epoch >= ph.evaluate_epoch:
            with torch.no_grad():
                log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch = \
                    train_val(
                        mode='val', dataset_name=dataset_name,
                        dataloader=val_loader, device=device, log_wandb=log_wandb, net=net,
                        optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                        best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_model_path=best_model_path, best_loss_model_path=best_loss_model_path,
                        non_improved_epoch=non_improved_epoch
                    )
    # 放在整个训练结束后
    wandb.finish()


if __name__ == '__main__':
    # 程序执行入口
    auto_experiment()

