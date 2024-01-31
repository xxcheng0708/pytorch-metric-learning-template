import os

os.environ["TORCH_HOME"] = "./pretrained_models"
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from utils.data_transforms import ZeroOneNormalize, RandomGaussianBlur, LetterBoxResize
from models.backbone_attention import Backbone
# from models.sync_bn import convertBNtoSyncBN
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.neighbors import KNeighborsClassifier
from utils.losses import get_loss, get_xbm_loss_func
import warnings
from utils.utils import seed_torch
from train import train
from pytorch_metric_learning.utils.inference import CustomKNN

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    seed_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config/embedding_cifar100.yaml", type=str, help="config file path")
    args = parser.parse_args()
    cfg_path = args.cfg

    with open(cfg_path, "r", encoding="utf8") as f:
        cfg_dict = yaml.safe_load(f)
    print(cfg_dict)

    in_channels = cfg_dict.get("in_channels")
    train_dataset_dir = cfg_dict.get("train_dataset_dir")
    save_dir = cfg_dict.get("save_dir")
    pretrained_model_path = cfg_dict.get("pretrained_models")
    validate_loss = cfg_dict.get("validate_loss")
    model_name = cfg_dict.get("model_name")
    use_pretrained = cfg_dict.get("use_pretrained")
    pretrained_ckpt_path = cfg_dict.get("pretrained_ckpt_path")

    xbm_enable = cfg_dict.get("xbm_enable")
    xbm_size = cfg_dict.get("xbm_size")
    xbm_start_iteration = cfg_dict.get("xbm_start_iteration")
    xbm_close = cfg_dict.get("xbm_close")

    mixup_enable = cfg_dict.get("mixup_enable")
    mixup_close = cfg_dict.get("mixup_close")

    xbm_mixup_alternate = cfg_dict.get("xbm_mixup_alternate", 0)

    out_dimension = cfg_dict.get("out_dimension")
    visible_device = cfg_dict.get("device")
    m_class_per_batch = cfg_dict.get("m_class_per_batch")
    m_sample_per_class = cfg_dict.get("m_sample_per_class")
    train_ratio = cfg_dict.get("train_ratio")
    val_ratio = cfg_dict.get("val_ratio")
    test_ratio = cfg_dict.get("test_ratio")

    invert_ratio = cfg_dict.get("invert_ratio")
    left_right_flip = cfg_dict.get("left_right_flip")
    up_down_flip = cfg_dict.get("up_down_flip")
    use_letterbox = cfg_dict.get("use_letterbox")

    num_workers = cfg_dict.get("num_workers")
    num_epoches = cfg_dict.get("epoch")
    lr = cfg_dict.get("lr")
    step_size = cfg_dict.get("step_size")
    gamma = cfg_dict.get("gamma")
    weight_decay = cfg_dict.get("weight_decay")
    momentum = cfg_dict.get("momentum", 0.9)

    if use_letterbox:
        # 使用 LetterBox 方式缩放图片
        resize_op_train = LetterBoxResize(dst_size=(256, 256))
        resize_op_val = LetterBoxResize(dst_size=(224, 224))
    else:
        resize_op_train = torchvision.transforms.Resize(size=(256, 256))
        resize_op_val = torchvision.transforms.Resize(size=(224, 224))

    # 训练集数据处理策略
    train_transforms_list = [
        resize_op_train,
        torchvision.transforms.RandomCrop(size=(224, 224)),
        # RandomGaussianBlur(kernel_size=(5, 5), p=0.8),
        # torchvision.transforms.RandomInvert(p=invert_ratio),
        # torchvision.transforms.ColorJitter(brightness=0.5, hue=0.0, saturation=0.0, contrast=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=left_right_flip),
        # torchvision.transforms.RandomVerticalFlip(p=up_down_flip),
        # torchvision.transforms.RandomRotation(degrees=15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    # 验证集数据处理策略
    val_transforms_list = [
        resize_op_val,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_transforms = torchvision.transforms.Compose(train_transforms_list)
    val_transforms = torchvision.transforms.Compose(val_transforms_list)

    train_dataset = torchvision.datasets.CIFAR100(root="./datasets", train=True, transform=train_transforms,
                                                  download=True)
    val_dataset = torchvision.datasets.CIFAR100(root="./datasets", train=False, transform=val_transforms, download=True)
    classes = train_dataset.classes

    m_class_per_batch = min(len(classes), m_class_per_batch)
    batchsize = m_class_per_batch * m_sample_per_class

    train_data_loader = DataLoader(train_dataset, batch_size=batchsize, drop_last=True, shuffle=True,
                                   num_workers=num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=batchsize, drop_last=False, shuffle=False,
                                 num_workers=num_workers)

    print("m_class_per_batch: {}, m_sample_per_class: {}, batch size: {}".format(m_class_per_batch, m_sample_per_class,
                                                                                 batchsize))
    print("train dataloader: {}, val dataloader: {}".format(len(train_data_loader), len(val_data_loader)))

    backbone = Backbone(out_dimension=out_dimension, model_name=model_name, pretrained=use_pretrained,
                        model_path=pretrained_ckpt_path)
    model, train_params, pretrained_params = backbone.build_model()

    # 当 batch size 比较小的时候，需要使用同步的 BatchNorm，收集多卡之间的训练数据信息
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # optimizer = torch.optim.SGD(
    #     [
    #         {"params": train_params, "lr": lr},
    #         {"params": pretrained_params, "lr": lr / 10}
    #     ],
    #     lr=lr,
    #     weight_decay=weight_decay
    # )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum
    )

    model = torch.nn.DataParallel(model, device_ids=visible_device).cuda()
    # print(model)
    print(model.device_ids)

    loss_func, xbm_loss_func, loss_miner, loss_optimizer, distance = get_loss("CircleLoss", xbm_enable, lr,
                                                                              weight_decay, classes_num=0,
                                                                              out_dimension=out_dimension,
                                                                              xbm_size=xbm_size)
    # print("classes: {}".format(len(classes)))
    # loss_func, xbm_loss_func, loss_miner, loss_optimizer, distance = get_loss("SupervisedContrastiveLoss", xbm_enable,
    #                                                                           lr, weight_decay, classes_num=0,
    #                                                                           out_dimension=out_dimension,
    #                                                                           xbm_size=xbm_size)

    accuracy_calculator = AccuracyCalculator(
        include=(
            "precision_at_1",
            "r_precision",
            "mean_average_precision_at_r"
        ),
        k="max_bin_count",
        knn_func=CustomKNN(distance=distance)
    )

    knn_model = KNeighborsClassifier(n_neighbors=5, p=2, n_jobs=-1)
    train_iter, val_iter, test_iter = train_data_loader, val_data_loader, None

    writer = SummaryWriter(save_dir)
    train(model, train_iter, train_dataset, val_iter, val_dataset, [loss_func, xbm_loss_func],
          num_epoches, accuracy_calculator, save_dir, optimizer=optimizer, loss_optimizer=loss_optimizer,
          mining_func=loss_miner, writer=writer, knn_model=None, mixup_enable=mixup_enable,
          xbm_enable=xbm_enable, xbm_start_iteration=xbm_start_iteration, xbm_close=xbm_close,
          validate_loss=validate_loss, ref_includes_query=False, print_step=10)
    writer.close()
