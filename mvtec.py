import math
from pathlib import Path
from typing import List

import cv2
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numpy import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import math

import yaml
import cv2
import os

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np





class CutPaste(object):

    def __init__(self, colorJitter=None, transform=None):
        self.transform = transform

    def __call__(self, img_org, img_cp, gt_augment, gt_normal):


        if self.transform:
            img_org = self.transform(img_org)
            img_cp = self.transform(img_cp)

        return img_org, img_cp, gt_augment, gt_normal

class Augmentation(CutPaste):

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        super(Augmentation, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def q_x(self, x_0, t):
        # x_0 to x_t
        num_steps = 1000

        betas = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        # noise = torch.randn_like(x_0)
        noise = np.random.normal(size=x_0.shape)
        noise = np.uint8(255 * noise)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]


        return np.uint8(alphas_t * x_0 + alphas_1_m_t * noise)


    def __call__(self, img):

        h = img.size[0]
        w = img.size[1]

        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]

        patch = img.crop(box)
        patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
        steps = np.random.randint(0, 1000)
        patch = self.q_x(patch, steps)
        patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

        augmented = img.copy()
        augmented.paste(patch, insert_box)

        mask = Image.fromarray(np.ones_like(patch))

        gt_augment = Image.fromarray(np.zeros_like(img))
        gt_augment.paste(mask, insert_box)
        gt_augment = gt_augment.convert('L')
        gt_augment = np.array(gt_augment)
        gt_augment[gt_augment != 0] = 1

        gt_normal = Image.fromarray(np.zeros_like(img))
        gt_normal = gt_normal.convert('L')
        gt_normal = np.array(gt_normal)

        return super().__call__(img, augmented, gt_augment, gt_normal)


def get_data_transforms(size):

    basic_transforms = transforms.Compose([])
    basic_transforms.transforms.append(transforms.ToTensor())
    basic_transforms.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    train_transforms = transforms.Compose([])
    train_transforms.transforms.append(transforms.Resize((size, size)))
    train_transforms.transforms.append(Augmentation(transform=basic_transforms))

    test_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])

    return train_transforms, test_transforms, gt_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type

