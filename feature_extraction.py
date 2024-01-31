import torch
import numpy as np
import os
from models.backbone_attention import Backbone
import torchvision
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from imutils.video import fps
from utils.data_transforms import ZeroMeanNormalize, ZeroOneNormalize, LetterBoxResize
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from matplotlib import pyplot as plt
import cv2
from utils.utils import normalize

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_path = "./results/model_hand_gesture_CircleLoss.pth"
    # data_path = "./datasets/test_data/hand_gesture"

    model_path = "./results/model_flower_photos_SupervisedContrastiveLoss.pth"
    data_path = "./datasets/test_data/flower_photos"

    # model_path = "./results/cat_and_dog_2_512_SupervisedContrastiveLoss/model-acc-004-0.7579-0.9692-0.7909.pth"
    # data_path = "/aidata/dataset/ImageNet/val"

    backbone = Backbone(out_dimension=256, model_name="resnet18", pretrained=False)
    model, _, _ = backbone.build_model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # print(model)

    use_letterbox = False
    val_transforms_list = [
        LetterBoxResize(dst_size=(224, 224)) if use_letterbox else torchvision.transforms.Resize(size=(224, 224)),
        ZeroOneNormalize(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    val_transforms = torchvision.transforms.Compose(val_transforms_list)

    fps = fps.FPS()
    fps.start()

    feature_data = []
    cls_idx_list = []
    for cls_idx, dir_name in enumerate(os.listdir(data_path)):
        if cls_idx == 50:
            break
        dir_path = os.path.join(data_path, dir_name)

        for sample_idx, image_name in enumerate(os.listdir(dir_path)):
            if sample_idx == 20:
                break
            image_path = os.path.join(dir_path, image_name)
            print(image_path)
            try:
                if use_letterbox:
                    data = cv2.imread(image_path)
                else:
                    data = read_image(image_path, mode=ImageReadMode.RGB)
                    data = data.to(device)

                data = val_transforms(data).unsqueeze(dim=0).cuda()
                # print(data.shape)

                feature = model(data)
                feature = normalize(feature, dim=1)
                feature = feature.cpu().detach().numpy()
                print("sample_idx: {}, feature: {}".format(sample_idx, feature.shape))
                feature_data.append(feature)
                cls_idx_list.append(cls_idx)
                fps.update()

            except Exception as e:
                print(image_name, e)
                raise Exception(e)

    fps.stop()
    print("FPS: {}".format(fps.fps()))
    print("duration: {}".format(fps.elapsed()))

    feature_data = np.concatenate(feature_data, axis=0)
    # print(feature_data.shape)
    # print(np.linalg.norm(feature_data, ord=2, axis=1))

    distance = cosine_distances(feature_data, feature_data)
    # distance = euclidean_distances(feature_data, feature_data)
    similarity = 1.0 - distance
    print(similarity)

    plt.figure(figsize=(50, 50))
    plt.imshow(similarity)
    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            plt.text(j, i, "{:.1f}".format(similarity[i, j]), ha='center', va='center', color='w')
    plt.savefig("./results/plot_flower_photos_with_value.jpg")
    plt.show()
