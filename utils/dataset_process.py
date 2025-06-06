import sys
import numpy as np
from os import listdir
from os.path import splitext
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import time
import wandb
import shutil


def verify_correspondence(dataset_name, mode=None):
    """
        验证数据集中 t1、t2 和 label 目录中的图像是否一一对应，即确保每个图像在这三个目录中都有相同的文件名。
        dataset_name：要验证的数据集名称，例如 'CLCD'。
        mode：指定验证的数据集类型，是训练集（train）、验证集（val）还是测试集（test）。如果 mode 为 None，则验证整个数据集中的图像。
    """

    if mode is None:
        images_dir = {'t1_images_dir': Path(f'./{dataset_name}/t1/'),
                      't2_images_dir': Path(f'./{dataset_name}/t2/'),
                      'label_images_dir': Path(f'./{dataset_name}/label/')
                      }
    else:
        images_dir = {'t1_images_dir': Path(f'./{dataset_name}/{mode}/t1/'),
                      't2_images_dir': Path(f'./{dataset_name}/{mode}/t2/'),
                      'label_images_dir': Path(f'./{dataset_name}/{mode}/label/')
                      }
    image_names = []
    for dir_path in images_dir.values():
        image_name = [splitext(file)[0] for file in listdir(dir_path) if not file.startswith('.')]
        image_names.append(image_name)
    image_names = np.unique(np.array(image_names))
    if len(image_names) != 1:
        print('Correspondence False')
        return False
    else:
        print('Correspondence Verified')
        return True


def delete_monochrome_image(dataset_name, mode=None):
    """ Delete monochrome images in dataset.
    作用是删除数据集中的单色图像（即全黑或全白的图像）。具体来说，它会删除标签目录（label）中全黑或全白的图像以及对应的 t1 和 t2 目录中的图像。

    Parameter:
        dataset_name(str): 指定的数据集名称。
        mode(str): 指定要操作的子集，可以是 'train'、'val' 或 'test'，默认为 None，表示操作整个数据集。

    Return:
        return nothing
    """

    if mode is None:
        t1_images_dir = Path(f'./{dataset_name}/t1/')
        t2_images_dir = Path(f'./{dataset_name}/t2/')
        label_images_dir = Path(f'./{dataset_name}/label/')
    else:
        t1_images_dir = Path(f'./{dataset_name}/{mode}/t1/'),
        t2_images_dir = Path(f'./{dataset_name}/{mode}/t2/'),
        label_images_dir = Path(f'./{dataset_name}/{mode}/label/')

    ids = [splitext(file)[0] for file in listdir(t1_images_dir) if not file.startswith('.')]
    img_name_sample = listdir(t1_images_dir)[0]
    img_sample = Image.open(str(t1_images_dir) + str(img_name_sample))
    img_size = img_sample.size[0]  # the image's height and width should be same

    if not ids:
        raise RuntimeError(f'No input file found in {t1_images_dir}, make sure you put your images there')
    for name in tqdm(ids):
        label_img_dir = list(label_images_dir.glob(str(name) + '.*'))
        assert len(label_img_dir) == 1, f'Either no mask or multiple masks found for the ID {name}: {label_img_dir}'
        img = Image.open(label_img_dir[0])
        img_array = np.array(img)
        array_sum = np.sum(img_array)
        if array_sum == 0 or array_sum == (255 * img_size * img_size):
            path = label_img_dir[0]
            path.unlink()

            t1_img_dir = list(t1_images_dir.glob(str(name) + '.*'))
            assert len(t1_img_dir) == 1, f'Either no mask or multiple masks found for the ID {name}: {t1_img_dir}'
            path = t1_img_dir[0]
            path.unlink()

            t2_img_dir = list(t2_images_dir.glob(str(name) + '.*'))
            path = t2_img_dir[0]
            path.unlink()
    print('Over')


def compute_mean_std(images_dir):
    """Compute the mean and std of dataset images.
        计算给定目录下所有图像的 均值（mean） 和 标准差（std） 的函数，通常用于数据预处理，尤其是在训练神经网络时进行图像归一化。
    Parameter:
        dataset_name(str): name of the specified dataset.

    Return:
        计算的均值和标准差列表，分别对应每个通道（R、G、B）。
        means(list): means in three channel(RGB) of images in :obj:`images_dir`
        stds(list): stds in three channel(RGB) of images in :obj:`images_dir`
    """
    # 将 images_dir 转换为 Path 对象，方便对目录和文件的操作
    images_dir = Path(images_dir)

    # 初始化均值和标准差：
    means = [0, 0, 0]
    stds = [0, 0, 0]
    # 使用 listdir() 获取目录中的所有文件名，并通过 splitext(file)[0] 获取去掉扩展名的文件名。
    ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
    # num_imgs 计算图像的数量。
    num_imgs = len(ids)

    # 果目录为空（没有图像），抛出异常。
    if not ids:
        raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
    # 环计算每张图像的均值和标准差：
    for name in tqdm(ids):
        img_file = list(images_dir.glob(str(name) + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # 使用 PIL.Image.open() 读取图像并转为 NumPy 数组。
        # img举个示例：[[[255, 0, 0], [0, 255, 0]],
        #         [[0, 0, 255], [255, 255, 255]]]
        img = Image.open(img_file[0])
        # img_array 是一个 NumPy 数组，表示一张图像，形状为 (height, width, channels)，即三维数组。对于 RGB 图像，channels 的大小是 3
        img_array = np.array(img)
        # 归一化图像：将图像数据标准化为 [0, 1] 范围
        img_array = img_array.astype(np.float32) / 255.
        for i in range(3):
            # img_array[:, :, i] 使用了 切片（slicing）来提取图像中第 i 个通道的数据（RGB 中分别是红色、绿色、蓝色通道）
            # : 表示取所有高度/宽度
            means[i] += img_array[:, :, i].mean()
            stds[i] += img_array[:, :, i].std()
    # 除以总数就得到所有图片的均值、标准差
    means = np.asarray(means) / num_imgs
    stds = np.asarray(stds) / num_imgs

    print("normMean = {}".format(means))
    print("normStd = {}".format(stds))

    return means, stds


def crop_img(dataset_name, pre_size, after_size, overlap_size):
    """
        将数据集中的图像裁剪成多个较小的图像。根据设定的裁剪大小（after_size），它可以根据给定的重叠大小（overlap_size）进行训练集裁剪，或者在验证集和测试集上进行无重叠的裁剪。
        dataset_name(str)：指定数据集的名称。
        pre_size(int)：裁剪前的原始图像大小（假设是正方形图像），例如，原图为1024x1024。
        after_size(int)：裁剪后的目标图像大小，例如裁剪成512x512的图像。
        overlap_size(int)：训练集裁剪时的重叠区域的大小，验证集和测试集则没有重叠。
    """

    if (pre_size - after_size % after_size - overlap_size != 0) or (pre_size % after_size != 0):
        print(f'ERROR: the pre_size - after size should be multiple of after_size - overlap_size, '
              f'and pre_size should be multiple of after size')
        sys.exit()

    # data path should be dataset_name/train or val or test/t1 or t2 or label
    train_image_dirs = [
        Path(f'./{dataset_name}/train/t1/'),
        Path(f'./{dataset_name}/train/t2/'),
        Path(f'./{dataset_name}/train/label/'),
    ]
    train_save_dirs = [
        f'./{dataset_name}_crop/train/t1/',
        f'./{dataset_name}_crop/train/t2/',
        f'./{dataset_name}_crop/train/label/',
    ]
    val_test_image_dirs = [
        Path(f'./{dataset_name}/val/t1/'),
        Path(f'./{dataset_name}/val/t2/'),
        Path(f'./{dataset_name}/val/label/'),
        Path(f'./{dataset_name}/test/t1/'),
        Path(f'./{dataset_name}/test/t2/'),
        Path(f'./{dataset_name}/test/label/'),
    ]
    val_test_save_dirs = [
        f'./{dataset_name}_crop/val/t1/',
        f'./{dataset_name}_crop/val/t2/',
        f'./{dataset_name}_crop/val/label/',
        f'./{dataset_name}_crop/test/t1/',
        f'./{dataset_name}_crop/test/t2/',
        f'./{dataset_name}_crop/test/label/',
    ]
    slide_size = after_size - overlap_size
    slide_times_with_overlap = (pre_size - after_size) // slide_size + 1
    slide_times_without_overlap = pre_size // after_size

    print('Start crop training images')
    # crop train images
    for images_dir, save_dir in zip(train_image_dirs, train_save_dirs):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ids = [splitext(file) for file in listdir(images_dir) if not file.startswith('.')]
        if not ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        for name, suffix in tqdm(ids):
            img_file = list(images_dir.glob(str(name) + '.*'))
            assert len(img_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {img_file}'
            img = Image.open(img_file[0])
            for i in range(slide_times_with_overlap):
                for j in range(slide_times_with_overlap):
                    box = (i * slide_size, j * slide_size,
                           i * slide_size + after_size, j * slide_size + after_size)
                    region = img.crop(box)
                    region.save(save_dir + f'/{name}_{i}_{j}{suffix}')

    print('Start crop val and test images')
    # crop val and test images
    for images_dir, save_dir in zip(val_test_image_dirs, val_test_save_dirs):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ids = [splitext(file) for file in listdir(images_dir) if not file.startswith('.')]
        if not ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        for name, suffix in tqdm(ids):
            img_file = list(images_dir.glob(str(name) + '.*'))
            assert len(img_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {img_file}'
            img = Image.open(img_file[0])
            for i in range(slide_times_without_overlap):
                for j in range(slide_times_without_overlap):
                    box = (i * after_size, j * after_size, (i + 1) * after_size, (j + 1) * after_size)
                    region = img.crop(box)
                    region.save(save_dir + f'/{name}_{i}_{j}{suffix}')
    print('Over')


def image_shuffle(dataset_name):
    """ Shuffle dataset images.

    随机打乱图像：代码通过打乱图像文件的顺序，将数据集中的每张图像重新命名。图像包括 t1（例如源图像）、t2（例如目标图像）和 label（例如标签图像），它们通常用于变化检测任务。

    Return:
        return nothing.
    """

    # data path should be dataset_name/t1 or t2 or label
    t1_images_dir = Path(f'./{dataset_name}/t1/')
    t2_images_dir = Path(f'./{dataset_name}/t2/')
    label_images_dir = Path(f'./{dataset_name}/label/')

    ids = [splitext(file) for file in listdir(t1_images_dir) if not file.startswith('.')]
    Imgnum = len(ids)
    L = random.sample(range(0, Imgnum), Imgnum)

    if not ids:
        raise RuntimeError(f'No input file found in {t1_images_dir}, make sure you put your images there')
    for i, (name, suffix) in tqdm(enumerate(ids)):
        t1_img_dir = list(t1_images_dir.glob(str(name) + '.*'))
        assert len(t1_img_dir) == 1, f'Either no mask or multiple masks found for the ID {name}: {t1_img_dir}'
        path = Path(t1_img_dir[0])
        new_file = path.with_name('shuffle_' + str(L[i]) + str(suffix))
        path.replace(new_file)

        t2_img_dir = list(t2_images_dir.glob(str(name) + '.*'))
        path = Path(t2_img_dir[0])
        new_file = path.with_name('shuffle_' + str(L[i]) + str(suffix))
        path.replace(new_file)

        label_img_dir = list(label_images_dir.glob(str(name) + '.*'))
        path = Path(label_img_dir[0])
        new_file = path.with_name('shuffle_' + str(L[i]) + str(suffix))
        path.replace(new_file)
    print('Over')


def split_image(dataset_name, fixed_ratio=True):
    """
        将数据集中的图像按一定比例（默认为 7:2:1，训练集、验证集和测试集的比例）或者指定数量（通过 fixed_ratio=False 设置）划分到三个子集：train（训练集）、val（验证集）和 test（测试集）。
    """
    source_image_dirs = [
        Path(f'./{dataset_name}/t1'),
        Path(f'./{dataset_name}/t2'),
        Path(f'./{dataset_name}/label'),
    ]
    train_save_dirs = [
        f'./{dataset_name}_split/train/t1/',
        f'./{dataset_name}_split/train/t2/',
        f'./{dataset_name}_split/train/label/',
    ]
    val_save_dirs = [
        f'./{dataset_name}_split/val/t1/',
        f'./{dataset_name}_split/val/t2/',
        f'./{dataset_name}_split/val/label/',
    ]
    test_save_dirs = [
        f'./{dataset_name}_split/test/t1/',
        f'./{dataset_name}_split/test/t2/',
        f'./{dataset_name}_split/test/label/',
    ]

    for i in range(3):
        Path(train_save_dirs[i]).mkdir(parents=True, exist_ok=True)
        Path(val_save_dirs[i]).mkdir(parents=True, exist_ok=True)
        Path(test_save_dirs[i]).mkdir(parents=True, exist_ok=True)

        ids = [splitext(file) for file in listdir(source_image_dirs[i]) if not file.startswith('.')]
        ids.sort()
        if not ids:
            raise RuntimeError(f'No input file found in {source_image_dirs[i]}, make sure you put your images there')

        if fixed_ratio:
            whole_num = len(ids)
            train_num = int(0.7 * whole_num)
            val_num = int(0.2 * whole_num)
            test_num = int(0.1 * whole_num)
        else:
            train_num = 540
            val_num = 152
            test_num = 1828

        for step, (name, suffix) in tqdm(enumerate(ids)):
            img_file = list(source_image_dirs[i].glob(str(name) + '.*'))
            assert len(img_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {img_file}'
            if step <= train_num:
                img_path = Path(img_file[0])
                new_path = Path(train_save_dirs[i] + str(name) + str(suffix))
                img_path.replace(new_path)
            elif step <= train_num + val_num:
                img_path = Path(img_file[0])
                new_path = Path(val_save_dirs[i] + str(name) + str(suffix))
                img_path.replace(new_path)
            else:
                img_path = Path(img_file[0])
                new_path = Path(test_save_dirs[i] + str(name) + str(suffix))
                img_path.replace(new_path)
    print('Over')


def crop_whole_image(dataset_name, crop_size):
    """
        这段代码的作用是将整个大图像裁剪成小块，每个小块的大小为 crop_size × crop_size，并且没有重叠。裁剪后的图像将分别保存为 t1、t2 和 label 子目录下的图像。

    """

    Image.MAX_IMAGE_PIXELS = None
    # images_path and suffix should be set
    images_path = [Path('./njds/T1_img/2014.tif'),
                   Path('./njds/T2_img/2018.tif'),
                   Path('./njds/Change_Label/gt.tif')
                   ]
    suffix = '.tif'
    save_path = [f'./{dataset_name}/t1/',
                 f'./{dataset_name}/t2/',
                 f'./{dataset_name}/label/'
                 ]
    # 确保存储路径存在
    for path in save_path:
        Path(path).mkdir(parents=True, exist_ok=True)

    for n in tqdm(range(len(images_path))):
        image = Image.open(images_path[n])
        w, h = image.size # 读取每个图像文件，获取其宽度 (w) 和高度 (h)。
        print(f'image size: {image.size}')
        for j in range(w // crop_size + 1):
            for i in range(h // crop_size + 1):
                if i == h // crop_size:
                    y1 = h - crop_size
                    y2 = h
                else:
                    y1 = i * crop_size
                    y2 = (i + 1) * crop_size
                if j == w // crop_size:
                    x1 = w - crop_size
                    x2 = w
                else:
                    x1 = j * crop_size
                    x2 = (j + 1) * crop_size

                box = (x1, y1, x2, y2)
                region = image.crop(box)
                region.save(save_path[n] + f'/{j}_{i}{suffix}')


def compare_predset():
    """
    比较两个预测结果集，并计算它们之间的差异，最后将这些差异存储到一个文件中。
    具体来说，它比较两个不同的预测结果（比如 pred_set_1 和 pred_set_2）之间的差异，并根据差异的大小将它们排序，然后保存到一个 .npy 文件。

    """

    # two pred set should be set first
    pred_set_1 = Path('./njds_val_dedf_pred_dir')  # dedf path
    pred_set_2 = Path('./njds_val_ded_pred_dir')  # ded path

    step = 0
    difference_dict = {}

    ids = [splitext(file) for file in listdir(pred_set_1) if not file.startswith('.')]
    if not ids:
        raise RuntimeError(f'No input file found in {pred_set_1}, make sure you put your images there')
    for name, suffix in ids:
        step += 1
        print(f'step: {step}')
        pred_1_file = list(pred_set_1.glob(str(name) + '.*'))
        assert len(pred_1_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {pred_1_file}'
        pred_1_image = Image.open(pred_1_file[0])
        pred_1_array = np.array(pred_1_image)

        pred_2_file = list(pred_set_2.glob(str(name) + '.*'))
        assert len(pred_2_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {pred_2_file}'
        pred_2_image = Image.open(pred_2_file[0])
        pred_2_array = np.array(pred_2_image)

        difference = np.sum(np.abs(pred_2_array - pred_1_array))

        difference_dict[str(name)] = difference

    ordered_difference_list = sorted(difference_dict.items(), key=lambda x: x[1], reverse=True)
    np.save('njds_ordered_val_difference.npy', np.array(ordered_difference_list))
    print('Over')


def display_dataset_image(dataset_name, mode=None):
    """ Display dataset image in wandb to inspect images.

    Notice that if mode is None,
    image path should be organized as :obj:`dataset_name`/`t1` or `t2` or `label`/ , else
    be organized as :obj:`dataset_name`/:obj:`mode`/`t1` or `t2` or `label`/ .

    Parameter:
        dataset_name(str): name of the specified dataset.
        mode(str): ensure whether sample train, val or test dataset.

    Return:
        return nothing.
    """

    if mode is not None:
        display_img_path = [f'./{dataset_name}/{mode}/t1/',
                            f'./{dataset_name}/{mode}/t2/',
                            f'./{dataset_name}/{mode}/label/'
                            ]
    else:
        display_img_path = [f'./{dataset_name}/t1/',
                            f'./{dataset_name}/t2/',
                            f'./{dataset_name}/label/'
                            ]
    localtime = time.asctime(time.localtime(time.time()))
    log_wandb = wandb.init(project='dpcd_last', resume='allow', anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=dict(time=localtime))
    ids = [splitext(file)[0] for file in listdir(display_img_path[0]) if not file.startswith('.')]
    if not ids:
        raise RuntimeError(f'No input file found in {display_img_path[0]}, make sure you put your images there')
    for name in tqdm(ids):
        display_img1 = list(Path(display_img_path[0]).glob(str(name) + '.*'))
        assert len(display_img1) == 1, f'Either no mask or multiple masks found for the ID {name}: {display_img1}'
        display_img1 = Image.open(display_img1[0])

        display_img2 = list(Path(display_img_path[1]).glob(str(name) + '.*'))
        assert len(display_img2) == 1, f'Either no mask or multiple masks found for the ID {name}: {display_img2}'
        display_img2 = Image.open(display_img2[0])

        display_img3 = list(Path(display_img_path[2]).glob(str(name) + '.*'))
        assert len(display_img3) == 1, f'Either no mask or multiple masks found for the ID {name}: {display_img3}'
        display_img3 = Image.open(display_img3[0])

        log_wandb.log({
            f'{display_img_path[0]}': wandb.Image(display_img1),
            f'{display_img_path[1]}': wandb.Image(display_img2),
            f'{display_img_path[2]}': wandb.Image(display_img3)
        })

    print('Over')


def sample_dataset(dataset_name, mode=None, ratio=None, num=None):
    """ Random sample specified ratio or number of dataset.

    从指定的数据集中随机采样一定比例或者数量的图像，并将这些图像复制到另一个目录

    Parameter:
        dataset_name(str): name of the specified dataset.
        mode(str): ensure whether sample train, val or test dataset.
        ratio(float): if not None, sample dataset with :math:`ratio` times :math:`dataset_size`.
        num(int): if not None, sample dataset with this num.
            if ratio and num are both not None, sample dataset with specified ratio.

    Return:
        return nothing.
    """

    if mode is not None:
        source_img_path = [
            f'./{dataset_name}/{mode}/t1/',
            f'./{dataset_name}/{mode}/t2/',
            f'./{dataset_name}/{mode}/label/'
        ]
        save_sample_img_path = [
            f'./{dataset_name}_sample/{mode}/t1/',
            f'./{dataset_name}_sample/{mode}/t2/',
            f'./{dataset_name}_sample/{mode}/label/'
        ]
    else:
        source_img_path = [
            f'./{dataset_name}/t1/',
            f'./{dataset_name}/t2/',
            f'./{dataset_name}/label/'
        ]
        save_sample_img_path = [
            f'./{dataset_name}_sample/t1/',
            f'./{dataset_name}_sample/t2/',
            f'./{dataset_name}_sample/label/'
        ]

    assert not (ratio is None and num is None), 'ratio and num are None at the same time'

    ids = [splitext(file) for file in listdir(source_img_path[0]) if not file.startswith('.')]
    Imgnum = len(ids)
    if ratio is not None:
        num = Imgnum * ratio
    img_index = random.sample(range(0, Imgnum), num)
    sample_imgs = [ids[i] for i in img_index]
    if not ids:
        raise RuntimeError(f'No input file found in {source_img_path[0]}, make sure you put your images there')
    for name, suffix in tqdm(sample_imgs):
        for i in range(len(source_img_path)):
            source_img = list(Path(source_img_path[i]).glob(str(name) + '.*'))
            assert len(source_img) == 1, f'Either no mask or multiple masks found for the ID {name}: {source_img}'
            source_file = Path(source_img[0])
            new_file = Path(save_sample_img_path[i] + str(name) + str(suffix))
            shutil.copyfile(source_file, new_file)
    print('Over')
