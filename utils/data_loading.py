import torch
import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.path_hyperparameter import ph
import torchvision.transforms as transforms
import ipdb


class BasicDataset(Dataset):

    # 初始化函数
    def __init__(self, t1_images_dir: str, t2_images_dir: str, labels_dir: str, train: bool,
                 t1_mean: list, t1_std: list, t2_mean: list, t2_std: list):

        self.t1_images_dir = Path(t1_images_dir)
        self.t2_images_dir = Path(t2_images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train

        # 获取 t1_images_dir 目录中所有文件的文件名，并将文件名中的扩展名去掉，最后将处理后的文件名存储在 self.t1_ids 列表中
        self.t1_ids = [splitext(file)[0] for file in listdir(t1_images_dir) if not file.startswith('.')]
        self.t2_ids = [splitext(file)[0] for file in listdir(t2_images_dir) if not file.startswith('.')]
        self.t1_ids.sort()
        self.t2_ids.sort()


        # 判断文件内容的合理性
        # 判断传来的t1\t2所在文件夹是否为空
        if not self.t1_ids:
            raise RuntimeError(f'No input file found in {t1_images_dir}, make sure you put your images there')
        if not self.t2_ids:
            raise RuntimeError(f'No input file found in {t2_images_dir}, make sure you put your images there')
        # 判断t1和t2图片的数量是否一致
        assert len(self.t1_ids) == len(self.t2_ids), 'number of t1 images is not equivalent to number of t2 images'
        logging.info(f'Creating dataset with {len(self.t1_ids)} examples')

        # 使用 Albumentations 库来定义一系列图像增强（augmentation）操作，目的是提高模型的泛化能力，防止过拟合。
        # Albumentations 是一个强大的图像处理库，专门用于图像增强，常用于深度学习模型的训练过程中，特别是处理视觉数据时
        self.train_transforms_all = A.Compose([
            #  水平翻转 图像
            A.Flip(p=0.5),
            # 这个操作是 矩阵转置，即交换图像的宽度和高度（也就是旋转 90 度）
            A.Transpose(p=0.5),
            # 这个操作用于 旋转 图像 45 度。
            A.Rotate(45, p=0.3),
            # 这个操作将图像同时进行 平移、缩放 和 旋转
            A.ShiftScaleRotate(p=0.3),
        ], additional_targets={'image1': 'image'}) # dditional_targets 参数的作用是告诉 Albumentations 除了对 image（即 t1_img）和 image1（即 t2_img）进行增强外，还需要对 其他目标 数据（例如 label，即 mask）进行增强

        self.train_transforms_image = A.Compose(
            # A.OneOf 允许你定义多个增强操作
            [A.OneOf([
                # 向图像添加 高斯噪声，模拟图像中出现的随机噪声
                A.GaussNoise(p=1),
                # 随机改变图像的色相（Hue）、饱和度（Saturation）和亮度（Value），模拟图像颜色的变化。
                A.HueSaturationValue(p=1),
                # 随机调整图像的亮度和对比度。p=1 表示每次增强时都会进行这种调整。
                A.RandomBrightnessContrast(p=1),
                # 随机调整图像的 伽马（gamma）值，这会影响图像的整体亮度
                A.RandomGamma(p=1),
                # 该操作模拟 浮雕效果，给图像添加光照和阴影的效果。
                A.Emboss(p=1),
                # 向图像添加 运动模糊，模拟相机抖动或物体快速移动造成的模糊效果。
                A.MotionBlur(p=1),
            ], p=ph.noise_p)],
            # 这表示除了对主图像（image）进行增强操作外，还会对额外的目标数据（例如，image1）进行相同的增强操作。
            additional_targets={'image1': 'image'})

        # 将多个图像预处理操作组合成一个管道。
        self.t1_normalize = A.Compose([
            # 对图像进行 标准化 操作，通常是将图像的像素值调整到某个均值（mean）和标准差（std）下。这样可以帮助模型更好地收敛，因为不同的图像可能会有不同的像素分布。
            A.Normalize(
                mean=t1_mean,
                std=t1_std)
        ])

        self.t2_normalize = A.Compose([
            A.Normalize(
                mean=t2_mean,
                std=t2_std)
        ])

        # 这是 Albumentations 库中的一个操作，用于将图像从 NumPy 数组转换为 PyTorch 的 Tensor 格式。
        # Tensor 格式是 (C, H, W)，即通道数（C）、高度（H）和宽度（W）
        self.to_tensor = A.Compose([
            ToTensorV2()
        ], additional_targets={'image1': 'image'})

    # 调用时机：每次访问数据集长度时 num_samples = len(dataset)
    def __len__(self):
        """ Return length of dataset."""

        return len(self.t1_ids)

    # 标签二值化处理：在 __getitem__ 中每次加载图像和标签时，会调用 label_preprocess 函数。
    # 遍历这张图片的每个像素，这个函数将标签图像中的非零像素值转换为 1，以将标签图像二值化。这样可以确保只有目标区域（通常标记为 1）和背景区域（标记为 0）在标签图像中。
    @classmethod
    def label_preprocess(cls, label):
        """ Binaryzation label."""

        label[label != 0] = 1
        return label

    # 读取并转换图像文件为数组。 在 __getitem__ 中加载图像/标签时
    @classmethod
    def load(cls, filename):
        """Open image and convert image to array."""

        img = Image.open(filename)
        # 将加载的图像对象转换为 NumPy 数组。NumPy 数组的格式使得图像可以方便地进行数值计算和处理。
        # 图像的每个像素值将被存储在数组中，通常是一个 3D 数组：[height, width, channels]，对应于图像的高度、宽度和颜色通道（RGB）
        img = np.array(img)

        return img

    # 每次按索引获取数据时（由 DataLoader 触发）
    # for batch in dataloader:  # 每次迭代触发__getitem__
    #     t1, t2, label = batch
    def __getitem__(self, idx):
        # 图片的前缀名字:00060
        t1_name = self.t1_ids[idx]
        # 图片的前缀名字:00060
        t2_name = self.t2_ids[idx]
        # 于 T1 和 T2 图像是一对一的，必须确保它们的文件名一致。如果不一致，就抛出一个错误。
        assert t1_name == t2_name, f't1 name{t1_name} not equal to t2 name{t2_name}'

        # glob 方法根据文件名的模式查找文件，并返回文件路径列表。
        # 在指定目录 self.t1_images_dir 中，查找所有文件名前缀匹配 t1_name 且扩展名任意的文件，返回匹配文件的完整路径列表。
        # 示例值：CLCD/train/t1/00060.png
        t1_img_file = list(self.t1_images_dir.glob(t1_name + '.*'))
        t2_img_file = list(self.t2_images_dir.glob(t2_name + '.*'))
        label_file = list(self.labels_dir.glob(t1_name + '.*'))

        # 确保每个图像都有且只有一个对应的标签文件。如果没有或有多个文件，会抛出错误。
        assert len(label_file) == 1, f'Either no label or multiple labels found for the ID {t1_name}: {label_file}'
        assert len(t1_img_file) == 1, f'Either no image or multiple images found for the ID {t1_name}: {t1_img_file}'

        # 使用 self.load 方法加载 T1、T2 图像和标签图像，将它们转换为 NumPy 数组或其他合适的格式
        # 经过load()，t1_img 和 t2_img 输出的格式就是 (H,W,3)

        t1_img = self.load(t1_img_file[0])
        t2_img = self.load(t2_img_file[0])


        label = self.load(label_file[0]) # label的格式就是(H,W)
        label = self.label_preprocess(label)



        # 针对数据进行增强的，不改变输入输出的格式
        if self.train:
            # sample是一个dict，
            sample = self.train_transforms_all(image=t1_img, image1=t2_img, mask=label)
            # sample['image']：这是增强后的 t1_img，它是一个 numpy 数组，形状和原始图像相同，但已经应用了增强操作（例如翻转、旋转等）。
            t1_img, t2_img, label = sample['image'], sample['image1'], sample['mask']
            sample = self.train_transforms_image(image=t1_img, image1=t2_img)
            t1_img, t2_img = sample['image'], sample['image1']

        # 对图像进行标准化操作
        t1_img = self.t1_normalize(image=t1_img)['image']
        t2_img = self.t2_normalize(image=t2_img)['image']

        # 在train模式下，才进行图片随机替换的操作
        if self.train:
            # 训练阶段，随机交换 T1 和 T2 图像的位置。通过 random.choice([0, 1]) 随机决定是否交换。
            # random exchange t1_img and t2_img
            if random.choice([0, 1]):
                t1_img, t2_img = t2_img, t1_img


        # 使用 ToTensorV2 转换 T1 和 T2 图像以及标签为 PyTorch 张量。
        # 原始的t1_img t2_img 格式 从（H，W,C） -> (C,H,W)
        sample = self.to_tensor(image=t1_img, image1=t2_img,mask=label)
        # ipdb.set_trace()
        t1_tensor, t2_tensor, label_tensor = sample['image'].contiguous(),\
                                             sample['image1'].contiguous(),sample['mask'].contiguous()
        name = t1_name

        # 输出格式：t1_tensor 和 t2_tensor :(3,H,W) label_tensor:(2,H,W)
        return t1_tensor, t2_tensor, label_tensor, name


