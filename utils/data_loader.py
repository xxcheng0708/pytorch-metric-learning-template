import random

from torchvision.io import read_image
import numpy as np
from torchvision.io.image import ImageReadMode
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision
from pytorch_metric_learning import samplers


class MusicNpyDataset(Dataset):
    def __init__(self, filenames, labels, transforms, down_sample=False):
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms
        self.down_sample = down_sample

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = np.load(self.filenames[idx]).astype(np.float32)
        if self.down_sample:
            data = data[:, ::4]
        data = self.transforms(data)
        # print(data.shape)
        return data, self.labels[idx]


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, transforms, use_letterbox=False, sample_count=-1):
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms
        self.use_letterbox = use_letterbox

        if sample_count != -1:
            sample_count = min(len(self.filenames), sample_count)
            random_choice = np.random.choice(len(self.filenames), size=sample_count, replace=False)
            self.filenames = np.array(self.filenames)[random_choice].tolist()
            self.labels = np.array(self.labels)[random_choice].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.use_letterbox:
            image = cv2.imread(self.filenames[idx])
        else:
            image = read_image(self.filenames[idx], mode=ImageReadMode.RGB)
        image = self.transforms(image)
        return image, self.labels[idx]


def load_dataset(dataset_dir, ratio):
    """
    数据加载
    :param dataset_dir:
    :param ratio:
    :param seed:
    :return:
    """
    # 加载按照类别目录存放的数据
    dataset = torchvision.datasets.ImageFolder(dataset_dir)
    classes = dataset.classes
    character = [[] for _ in range(len(classes))]
    random.shuffle(dataset.samples)
    sample_count = {}

    for x, y in dataset.samples:
        if y not in sample_count:
            sample_count[y] = 0

        character[y].append(x)
        sample_count[y] += 1
    print(classes)
    print(sample_count)

    # 按照比例划分训练集/验证集/测试集
    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):
        num_sample_train = int(len(data) * ratio[0])
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val
        print(classes[i], num_sample_train, num_sample_val, num_val_index)

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
        for x in data[num_val_index:]:
            test_inputs.append(str(x))
            test_labels.append(i)

    print("train_inputs: {}, train_labels: {}".format(len(train_inputs), len(train_labels)))
    print("val_inputs: {}, val_labels: {}".format(len(val_inputs), len(val_labels)))
    print("test_inputs: {}, test_labels: {}".format(len(test_inputs), len(test_labels)))

    data_info = {
        "train_inputs": train_inputs,
        "train_labels": train_labels,
        "val_inputs": val_inputs,
        "val_labels": val_labels,
        "test_inputs": test_inputs,
        "test_labels": test_labels
    }
    return data_info, classes


def load_dataloader(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels,
                    batchsize=512, num_workers=8,
                    train_transforms=None, val_transforms=None,
                    use_letterbox=False, sample_count=-1, m_sample_per_class=1):
    """
    数据预处理
    :param train_inputs:
    :param train_labels:
    :param val_inputs:
    :param val_labels:
    :param test_inputs:
    :param test_labels:
    :param batchsize:
    :param num_workers:
    :param train_transforms:
    :param val_transforms:
    :return:
    """
    train_dataset = ImageDataset(train_inputs, train_labels, train_transforms, use_letterbox, sample_count=sample_count)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, drop_last=True,
                                  shuffle=False, num_workers=num_workers,
                                  sampler=samplers.MPerClassSampler(train_labels,
                                                                    m=m_sample_per_class,
                                                                    batch_size=batchsize,
                                                                    length_before_new_iter=len(train_labels)),
                                  pin_memory=True)
    val_dataset = None
    val_dataloader = None
    test_dataset = None
    test_dataloader = None

    if len(val_inputs) >= batchsize:
        val_dataset = ImageDataset(val_inputs, val_labels, val_transforms, use_letterbox, sample_count=sample_count)
        val_dataloader = DataLoader(val_dataset, batch_size=batchsize, drop_last=False,
                                    shuffle=False, num_workers=num_workers,
                                    sampler=samplers.MPerClassSampler(val_labels,
                                                                      m=m_sample_per_class,
                                                                      batch_size=batchsize,
                                                                      length_before_new_iter=len(val_labels)),
                                    pin_memory=True)

    if len(test_inputs) >= batchsize:
        test_dataset = ImageDataset(test_inputs, test_labels, val_transforms, use_letterbox, sample_count=sample_count)
        test_dataloader = DataLoader(test_dataset, batch_size=batchsize, drop_last=False,
                                     shuffle=False, num_workers=num_workers,
                                     sampler=samplers.MPerClassSampler(test_labels,
                                                                       m=m_sample_per_class,
                                                                       batch_size=batchsize,
                                                                       length_before_new_iter=len(test_labels)),
                                     pin_memory=True)

    loader = {}
    loader["train_set"] = train_dataset
    loader["val_set"] = val_dataset
    loader["test_set"] = test_dataset
    loader["train_loader"] = train_dataloader
    loader["val_loader"] = val_dataloader
    loader["test_loader"] = test_dataloader
    return loader
