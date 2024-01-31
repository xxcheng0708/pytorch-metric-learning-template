import json
import os

import torch
import torchvision
from models.backbone_attention import Backbone
from pytorch_metric_learning.utils.inference import InferenceModel
import random
from utils.data_loader import ImageDataset
from utils.data_transforms import ZeroOneNormalize
from torchvision.io.image import read_image, ImageReadMode
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import MatchFinder
from matplotlib import pyplot as plt

# 验证集数据处理策略
val_transforms_list = [
    torchvision.transforms.Resize(size=(224, 224)),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
val_transforms = torchvision.transforms.Compose(val_transforms_list)

# 加载训练集
ratio = [0.8, 0.2, 0.0]
dataset = torchvision.datasets.ImageFolder("./datasets/flower_photos")
classes = dataset.classes
character = [[] for _ in range(len(classes))]
random.shuffle(dataset.samples)
sample_count = {}

for x, y in dataset.samples:
    if y not in sample_count:
        sample_count[y] = 0
    character[y].append(x)
    sample_count[y] += 1

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

train_dataset = ImageDataset(train_inputs, train_labels, val_transforms, False, sample_count=-1)
val_dataset = ImageDataset(val_inputs, val_labels, val_transforms, False, sample_count=-1)

# 加载训练好的模型
model_path = "results/model_flower_photos_SupervisedContrastiveLoss.pth"
backbone = Backbone(out_dimension=256, model_name="resnet18", pretrained=False, model_path=None)
model, _, _ = backbone.build_model()
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.eval()

# 加载待识别的图像数据
queries = []
labels = []
test_data_path = "./datasets/test_data/flower_photos"
for dir_name in os.listdir(test_data_path):
    for idx, image_name in enumerate(os.listdir(os.path.join(test_data_path, dir_name))):
        # if idx == 2:
        #     break
        image = read_image(os.path.join(test_data_path, dir_name, image_name), mode=ImageReadMode.RGB)
        image = val_transforms(image)
        queries.append(image)
        labels.append(dir_name)

# initialize with a model
im = InferenceModel(model, match_finder=MatchFinder(distance=CosineSimilarity(), threshold=0.70),
                    normalize_embeddings=True)

if os.path.exists("./results/inference_flower_photos.index") and \
        os.path.exists("./results/inference_flower_photos_labels.txt"):
    im.load_knn_func("./results/inference_flower_photos.index")
    with open("./results/inference_flower_photos_labels.txt", "r", encoding="utf8") as f:
        ref_data_labels = json.loads(f.read())
else:
    # 把之前的训练集作为reference，分两批次加入到参考特征，使用query与reference进行特征匹配查询
    train_dataset_imgs = []
    train_data_labels = []
    for idx in range(len(train_dataset)):
        data, label = train_dataset[idx]
        train_dataset_imgs.append(data)
        train_data_labels.append(label)
    # pass in a dataset to serve as the search space for k-nn
    im.train_knn(train_dataset_imgs)

    val_dataset_imgs = []
    val_data_labels = []
    for idx in range(len(val_dataset)):
        data, label = val_dataset[idx]
        val_dataset_imgs.append(data)
        val_data_labels.append(label)
    # add another dataset to the index
    im.add_to_knn(val_dataset_imgs)

    im.save_knn_func("./results/inference_flower_photos.index")
    with open("./results/inference_flower_photos_labels.txt", "w", encoding="utf8") as f:
        ref_data_labels = train_data_labels + val_data_labels
        f.write(json.dumps(ref_data_labels))

# get the 10 nearest neighbors of a query，在reference中查找与query最匹配的10个特征，返回特征对应的距离和样本编号
distances, indices = im.get_nearest_neighbors(queries, k=10)
print("distances: {}".format(distances))
print("indices: {}".format(indices))
print("data labels: {}".format(len(ref_data_labels)))

# 根据样本编号确定匹配到的样本的标签
for idx in range(len(queries)):
    distance = distances[idx]
    indice = indices[idx]

    matched_labels = []
    for i in indice:
        matched_labels.append(classes[ref_data_labels[i]])
    print(labels[idx], matched_labels, distance)

# determine if inputs are close to each other，判断queries中的样本特征是否有与之匹配的样本特征
is_match = im.is_match(queries, queries)
print("is_match: {}".format(is_match))

# determine "is_match" pairwise for all elements in a batch，判断queries中的样本特征是否两两相互匹配
match_matrix = im.get_matches(queries)
print("match_matrix: {}".format(match_matrix))

plt.imshow(match_matrix)
plt.show()

# save and load the knn function (which is a faiss index by default)
# im.save_knn_func("./datasets/test_data/hand_gesture.index")
# im.load_knn_func("./datasets/test_data/hand_gesture.index")
