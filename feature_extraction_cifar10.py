import torch
import numpy as np
import os
from models.backbone_attention import Backbone
import torchvision
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from imutils.video import fps
from sklearn.metrics.pairwise import cosine_distances
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
from utils.utils import normalize


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "./results/model_cifar10_SupervisedContrastiveLoss.pth"

    backbone = Backbone(out_dimension=256, model_name="resnet18", pretrained=False)
    model, _, _ = backbone.build_model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # print(model)

    val_transforms_list = [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    val_transforms = torchvision.transforms.Compose(val_transforms_list)

    val_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=val_transforms, download=True)
    classes = val_dataset.classes
    val_data_loader = DataLoader(val_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=4)

    feature_data = {}
    cls_idx_list = []
    count = 0
    for data, cls_idx in val_data_loader:
        cls_idx = cls_idx.item()
        if cls_idx not in feature_data:
            feature_data[cls_idx] = []
        if cls_idx in feature_data and len(feature_data[cls_idx]) == 10:
            continue

        feature = model(data.cuda())
        feature = normalize(feature, dim=1)
        feature = feature.cpu().detach().numpy()
        print("feature: {}".format(feature.shape))
        feature_data[cls_idx].append(feature)
        count += 1
        print(count, cls_idx)

# print(list(feature_data.values()))
feature_data = np.concatenate(list(feature_data.values()), axis=0).squeeze()
print(feature_data.shape)
print(np.linalg.norm(feature_data, ord=2, axis=1))

distance = cosine_distances(feature_data, feature_data)
similarity = 1.0 - distance
print(similarity)

plt.figure(figsize=(50, 50))
plt.imshow(similarity)
for i in range(similarity.shape[0]):
    for j in range(similarity.shape[1]):
        plt.text(j, i, "{:.1f}".format(similarity[i, j]), ha='center', va='center', color='w')
plt.savefig("./results/plot_cifar10_with_value.jpg")
plt.show()
