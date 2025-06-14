import sys
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
import logging
from utils.path_hyperparameter import ph
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, JaccardIndex
# from models.Models import DPCD
from models.Models_trans import DPCD
from models.FHLCDNet import FHLCDNet
from utils.dataset_process import compute_mean_std
from tqdm import tqdm

'''

'''


def test_models(dataset_name, load_checkpoint=True):
    # 1. Create dataset

    # compute mean and std of train dataset to normalize train/val/test dataset
    t1_mean, t1_std = compute_mean_std(images_dir=f'../autodl-tmp/datasets/{dataset_name}/train/t1/')
    t2_mean, t2_std = compute_mean_std(images_dir=f'../autodl-tmp/datasets/{dataset_name}/train/t2/')

    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    test_dataset = BasicDataset(t1_images_dir=f'../autodl-tmp/datasets/{dataset_name}/test/t1/',
                                t2_images_dir=f'../autodl-tmp/datasets/{dataset_name}/test/t2/',
                                labels_dir=f'../autodl-tmp/datasets/{dataset_name}/test/label/',
                                train=False, **dataset_args)
    # 2. Create data loaders
    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True
                       )

    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    print(f'Using device {device}')
    net = FHLCDNet()
    net.to(device=device)

    load_model = torch.load(ph.load_best_pth, map_location=device)
    if load_checkpoint:
        net.load_state_dict(load_model['net'])
    else:
        net.load_state_dict(load_model)
    print(f'Model loaded from {ph.load_best_pth}')
    torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')

    metric_collection = MetricCollection({
        'accuracy': Accuracy(task='binary').to(device=device),
        'precision': Precision(task='binary').to(device=device),
        'recall': Recall(task='binary').to(device=device),
        'f1score': F1Score(task='binary').to(device=device)
    })  # metrics calculator

    net.eval()
    print('SET model mode to test!')

    with torch.no_grad():
        for batch_img1, batch_img2, labels, name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.float().to(device)

            cd_preds = net(batch_img1, batch_img2)

            # Calculate and log other batch metrics
            cd_preds = torch.sigmoid(cd_preds).float()
            labels = labels.int()
            metric_collection.update(cd_preds, labels)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        test_metrics = metric_collection.compute()
        # 替换 IoU 为 f1 / (2 - f1)
        f1 = test_metrics['f1score'].item()
        iou_from_f1 = f1 / (2 - f1) if (2 - f1) != 0 else 0.0
        test_metrics['IoU'] = torch.tensor(iou_from_f1)

        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()

    print('over')


if __name__ == '__main__':

    try:
        # TODO:需要修改成自己的数据集路径
        test_models(dataset_name=ph.dataset_name, load_checkpoint=False)
    except KeyboardInterrupt:
        print('Error')
        sys.exit(0)
