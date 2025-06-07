class Path_Hyperparameter:
    random_seed = 42
    save_path = './output/'
    # dataset hyper-parameter
    dataset_name = 'test_cd'

    # training hyper-parameter
    epochs: int = 220  # Number of epochs
    batch_size: int = 1  # Batch size
    inference_ratio = 1  # batch_size in val and test equal to batch_size*inference_ratio
    learning_rate: float = 5e-4  # Learning rate
    factor = 0.1  # learning rate decreasing factor
    patience = 10  # schedular patience
    warm_up_step = 500  # warm up step
    weight_decay: float = 0.0025  # AdamW optimizer weight decay
    amp: bool = True  # if use mixed precision or not
    load: str = False
    # 用于测试已经训练好的 模型
    # load: str = '/home/chenyuxiang/Desktop/ShenPei/change_detection/SGSLN/SYSU-CD_best_f1score_model/best_f1score_epoch203_Mon Apr 28 22:45:22 2025.pth'  # Load model and/or optimizer from a .pth file for testing or continuing training
    max_norm: float = 20  # gradient clip max norm

    # evaluate hyper-parameter
    evaluate_epoch: int = 0  # start evaluate after training for evaluate epochs
    stage_epoch = [0, 0, 0, 0, 0]  # adjust learning rate after every stage epoch
    save_checkpoint: bool = True  # if save checkpoint of model or not
    save_interval: int = 10  # save checkpoint every interval epoch
    save_best_model: bool = True  # if save best model or not

    # log wandb hyper-parameter
    # log_wandb_project: str = 'change_detection'  # wandb project name

    # data transform hyper-parameter
    noise_p: float = 0.1  # probability of adding noise

    # model hyper-parameter
    dropout_p: float = 0.5  # probability of dropout
    patch_size: int = 256  # size of input image

    y = 2  # ECA-net parameter
    b = 1  # ECA-net parameter

    # inference parameter
    log_path = './log_feature/'

    # 最终的 hyperparameter_dict 字典会包含类中所有非下划线开头的类属性（即所有公开配置参数）
    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}


ph = Path_Hyperparameter()
