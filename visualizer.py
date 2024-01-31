from utils.evaluate import get_all_embeddings
import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler
import torchvision
from models.backbone_attention import Backbone
import torch
import os
import time
from utils.data_loader import load_dataset, ImageDataset
from utils.data_transforms import ZeroOneNormalize
from sklearn.decomposition import PCA


def visualizer(embeddings, labels, save_path):
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        # if i == 20:
        #     break
        idx = labels == label_set[i]
        print(idx)
        plt.plot(embeddings[idx.squeeze(), 0], embeddings[idx.squeeze(), 1], ".", markersize=2)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    val_transforms_list = [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),    # cifar10 cifar100
        # ZeroOneNormalize(),  # flower_photos hand_geature
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    val_transforms = torchvision.transforms.Compose(val_transforms_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_path = "./results/model_cifar10_SupervisedContrastiveLoss.pth"
    # val_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=val_transforms, download=True)
    # save_path = "./results/visualizer_cifar10.jpg"

    model_path = "./results/model_cifar100_CircleLoss.pth"
    val_dataset = torchvision.datasets.CIFAR100(root="./datasets", train=True, transform=val_transforms, download=True)
    save_path = "./results/visualizer_cifar100.jpg"

    # model_path = "./results/model_flower_photos_SupervisedContrastiveLoss.pth"
    # dataset_info, classes = load_dataset("./datasets/flower_photos", ratio=[1.0, 0.0, 0.0])
    # val_dataset = ImageDataset(dataset_info["train_inputs"], dataset_info["train_labels"], val_transforms, False)
    # save_path = "./results/visualizer_flower_photos.jpg"

    # model_path = "./results/model_hand_gesture_CircleLoss.pth"
    # dataset_info, classes = load_dataset("./datasets/hand_keypoints", ratio=[1.0, 0.0, 0.0])
    # val_dataset = ImageDataset(dataset_info["train_inputs"], dataset_info["train_labels"], val_transforms, False)
    # save_path = "./results/visualizer_hand_gesture.jpg"

    # model_path = "./results/ImageNet_40_32_CircleLoss/model-acc-000-0.3200-0.7040-0.4171.pth"
    # dataset_info, classes = load_dataset("/aidata/dataset/ImageNet/val/val", ratio=[1.0, 0.0, 0.0])
    # val_dataset = ImageDataset(dataset_info["train_inputs"], dataset_info["train_labels"], val_transforms, False)
    # save_path = "./results/visualizer_imagenet.jpg"

    backbone = Backbone(out_dimension=256, model_name="resnet18", pretrained=False)
    model, _, _ = backbone.build_model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # print(model)

    embeddings, labels = get_all_embeddings(val_dataset, model)
    embeddings = embeddings.cpu()
    labels = labels.cpu()

    # 对高维特征进行PCA降维
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(embeddings)

    visualizer(embeddings, labels, save_path)
