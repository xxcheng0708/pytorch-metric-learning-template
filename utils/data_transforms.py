import numpy as np
import torchvision.transforms
from PIL import ImageOps as plops
import torch
import random
from PIL import Image, ImageFilter
import cv2
from torch.nn import functional as F


class InvertTransform(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        random_prob = np.random.uniform(0, 1)
        if 1.0 - random_prob < self.p:
            img = plops.invert(img)
        return img


class UpDownFlipTransform(object):
    """
        Image图片数据随机上下翻转
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        random_prob = np.random.uniform(0, 1)
        if 1.0 - random_prob < self.p:
            img = img[::-1, :].copy()
        return img


class LeftRightFlipTransform(object):
    """
        Image图片数据随机左右翻转
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        random_prob = np.random.uniform(0, 1)
        if 1.0 - random_prob < self.p:
            img = img[:, ::-1].copy()
        return img


class MinMaxNormalize(object):
    """
        最大-最小归值一化

    """

    def __call__(self, x):
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)
        return x


class MeanStdNormalize(object):
    def __call__(self, x):
        x_mean = x.mean()
        x_std = x.std()
        x = (x - x_mean) / x_std
        return x


class ZeroMeanNormalize(object):
    def __call__(self, x):
        x_mean = x.mean()
        x = x - x_mean
        return x


class AddSaltPepperNoise(object):
    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            mask = np.repeat(mask, c, axis=2)
            img[mask == 0] = 0
            img[mask == 1] = 255
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class AddBlur(object):
    def __init__(self, p=0.5, blur="normal"):
        self.p = p
        self.blur = blur

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            if self.blur == "normal":
                img = img.filter(ImageFilter.BLUR)
                return img
            if self.blur == "Gaussian":
                img = img.filter(ImageFilter.GaussianBlur)
                return img
            if self.blur == "mean":
                img = img.filter(ImageFilter.BoxBlur)
                return img
        return img


class ZeroOneNormalize(object):
    def __call__(self, img):
        # return img.float().div(1)
        return img.float().div(255)


class RandomGaussianBlur(object):
    def __init__(self, kernel_size, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = torchvision.transforms.GaussianBlur(kernel_size=self.kernel_size)(img)
            return img
        return img


class LetterBoxResize(object):
    def __init__(self, dst_size, pad_color=(114, 114, 114), device=None):
        """
        :param dst_size:
        :param pad_color:
        """
        self.dst_size = dst_size
        self.pad_color = pad_color
        self.device = device

    def __call__(self, image_src):
        """
        image resize, keep aspect ratio
        :param image_src: image (numpy or torch)
        :return:
        """
        src_h, src_w = image_src.shape[:2]
        dst_h, dst_w = self.dst_size
        scale = min(dst_h / src_h, dst_w / src_w)
        pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))
        # print("image src: {}, {}, {}".format(image_src.shape, pad_h, pad_w))

        if image_src.shape[0:2] != (pad_w, pad_h):
            image_dst = cv2.resize(image_src, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_dst = image_src

        top = int((dst_h - pad_h) / 2)
        down = int((dst_h - pad_h + 1) / 2)
        left = int((dst_w - pad_w) / 2)
        right = int((dst_w - pad_w + 1) / 2)

        # add border
        image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=self.pad_color)

        # print("image_dst: {}".format(image_dst.shape))
        # x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
        if self.device:
            image_dst = torch.as_tensor(image_dst, device=self.device).permute(2, 0, 1)
        else:
            image_dst = torch.as_tensor(image_dst).permute(2, 0, 1)
        # print("image_dst: {}, {}".format(image_dst.shape, image_dst.device))
        # return image_dst, x_offset, y_offset
        return image_dst