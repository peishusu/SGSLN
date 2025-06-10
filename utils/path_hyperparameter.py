class Path_Hyperparameter:
    random_seed = 10

    # dataset hyper-parameter
    dataset_name = 'S2Looking'

    early_stop_patience = 20  # 连续多少个 epoch 未提升后早停

    # training hyper-parameter
    epochs: int = 150  # Number of epochs
    batch_size: int = 32  # Batch size
    inference_ratio = 1  # batch_size in val and test equal to batch_size*inference_ratio
    learning_rate: float = 1e-4  # Learning rate
    factor = 0.1  # learning rate decreasing factor
    patience = 20  # schedular patience
    warm_up_step = 500  # warm up step
    weight_decay: float = 0.0025  # AdamW optimizer weight decay
    amp: bool = True  # if use mixed precision or not
    load: str = False  # Load model and/or optimizer from a .pth file for testing or continuing training
    load_best_pth:str = "./S2Looking_best_f1score_model/best_multi_metric_epoch54_Tue Jun 10 23:06:39 2025.pth" # 设置加载pth文件的路径
    max_norm: float = 20  # gradient clip max norm

    # evaluate hyper-parameter
    evaluate_epoch: int = 0  # start evaluate after training for evaluate epochs
    stage_epoch = [0, 0, 0, 0, 0]  # adjust learning rate after every stage epoch
    save_checkpoint: bool = True  # if save checkpoint of model or not
    save_interval: int = 10  # save checkpoint every interval epoch
    save_best_model: bool = True  # if save best model or not

    # log wandb hyper-parameter
    log_wandb_project: str = 'shenpei_cd'  # wandb project name

    # data transform hyper-parameter
    noise_p: float = 0.2  # probability of adding noise

    # model hyper-parameter
    dropout_p: float = 0.1  # probability of dropout
    patch_size: int = 256  # size of input image

    y = 2  # ECA-net parameter
    b = 1  # ECA-net parameter

    # inference parameter
    log_path = './log_feature/'

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}


ph = Path_Hyperparameter()