# coding:utf-8

""" Docstring
"""

import os
from PIL import Image
from skimage import io
import numpy as np

import torch
from torch import nn  # For neural network
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch.nn.functional as F


data_path = 'data/'


class SaltDataset(Dataset):
    def __init__(self, root='./data', train=True, transform=None):

        self.items = [(
            os.path.join(os.path.join(root, 'images', file_name)),
            os.path.join(os.path.join(root, 'masks', file_name)))
            for file_name in os.listdir(os.path.join(root, 'images')
                                        )]
        if train:
            self.items = self.items[:int(0.9*len(self.items))]
        else:
            self.items = self.items[int(0.9*len(self.items)):]

        self.images_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.masks_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, mask_path = self.items[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.images_transform(image)
        mask = self.masks_transform(mask).long()/65535

        return image, mask


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)
        self.relu10 = nn.ReLU()

    def forward(self, x):
        c1 = self.relu1(self.conv1_2(self.conv1_1(x)))
        p1 = self.pool1(c1)

        c2 = self.relu2(self.conv2_2(self.conv2_1(p1)))
        p2 = self.pool2(c2)

        c3 = self.relu3(self.conv3_2(self.conv3_1(p2)))
        p3 = self.pool3(c3)

        c4 = self.relu4(self.conv4_2(self.conv4_1(p3)))
        p4 = self.pool4(c4)

        c5 = self.relu5(self.conv5_2(self.conv5_1(p4)))

        u6 = self.upsample1(c5)
        u6 = torch.cat((u6, c4), dim=1)
        c6 = self.relu6(self.conv6_2(self.conv6_1(u6)))

        u7 = self.upsample2(c6)
        u7 = torch.cat((u7, c3), dim=1)
        c7 = self.relu7(self.conv7_2(self.conv7_1(u7)))

        u8 = self.upsample3(c7)
        u8 = torch.cat((u8, c2), dim=1)
        c8 = self.relu8(self.conv8_2(self.conv8_1(u8)))

        u9 = self.upsample4(c8)
        u9 = torch.cat((u9, c1), dim=1)
        c9 = self.relu9(self.conv9_2(self.conv9_1(u9)))

        c10 = self.relu10(self.conv10(c9))
        return c10


class UNETWithoutPooling(nn.Module):
    """UNET without pooling"""
    def __init__(self):
        super(UNETWithoutPooling, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)
        self.relu10 = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1_1(x)
        p1 = self.conv1_2(c1)
        p1 = self.relu1(p1)

        c2 = self.conv2_1(p1)
        p2 = self.conv2_2(c2)
        p2 = self.relu2(p2)

        c3 = self.conv3_1(p2)
        p3 = self.conv3_2(c3)
        p3 = self.relu3(p3)

        c4 = self.conv4_1(p3)
        p4 = self.conv4_2(c4)
        p4 = self.relu4(p4)

        c5 = self.conv5_1(p4)
        p5 = self.conv5_2(c5)
        p5 = self.relu5(p5)

        u6 = self.upsample1(p5)
        u6 = torch.cat((u6, c4), 1)
        c6 = self.relu6(self.conv6_2(self.conv6_1(u6)))

        u7 = self.upsample2(c6)
        u7 = torch.cat((u7, c3), 1)
        c7 = self.relu7(self.conv7_2(self.conv7_1(u7)))

        u8 = self.upsample3(c7)
        u8 = torch.cat((u8, c2), 1)
        c8 = self.relu8(self.conv8_2(self.conv8_1(u8)))

        u9 = self.upsample4(c8)
        u9 = torch.cat((u9, c1), 1)
        c9 = self.relu9(self.conv9_2(self.conv9_1(u9)))

        c10 = self.relu10(self.conv10(c9))
        return c10


class UNETWithoutConcat(nn.Module):
    """UNET Without concatenation during decoding"""

    def __init__(self):
        super(UNETWithoutConcat, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)
        self.relu10 = nn.ReLU()

    def forward(self, x):
        c1 = self.relu1(self.conv1_2(self.conv1_1(x)))
        p1 = self.pool1(c1)

        c2 = self.relu2(self.conv2_2(self.conv2_1(p1)))
        p2 = self.pool2(c2)

        c3 = self.relu3(self.conv3_2(self.conv3_1(p2)))
        p3 = self.pool3(c3)

        c4 = self.relu4(self.conv4_2(self.conv4_1(p3)))
        p4 = self.pool4(c4)

        c5 = self.relu5(self.conv5_2(self.conv5_1(p4)))

        u6 = self.upsample1(c5)
        # u6 = torch.cat((u6, c4), dim=1)
        c6 = self.relu6(self.conv6_2(self.conv6_1(u6)))

        u7 = self.upsample2(c6)
        # u7 = torch.cat((u7, c3), dim=1)
        c7 = self.relu7(self.conv7_2(self.conv7_1(u7)))

        u8 = self.upsample3(c7)
        # u8 = torch.cat((u8, c2), dim=1)
        c8 = self.relu8(self.conv8_2(self.conv8_1(u8)))

        u9 = self.upsample4(c8)
        # u9 = torch.cat((u9, c1), dim=1)
        c9 = self.relu9(self.conv9_2(self.conv9_1(u9)))

        c10 = self.relu10(self.conv10(c9))
        return c10


class UNETAdd(nn.Module):
    """UNET Without concatenation during decoding"""

    def __init__(self):
        super(UNETAdd, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.SELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.SELU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.SELU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.SELU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.SELU()

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.SELU()

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.SELU()

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.SELU()

        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.SELU()

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)
        self.relu10 = nn.SELU()

    def forward(self, x):
        c1 = self.relu1(self.conv1_2(self.conv1_1(x)))
        p1 = self.pool1(c1)

        c2 = self.relu2(self.conv2_2(self.conv2_1(p1)))
        p2 = self.pool2(c2)

        c3 = self.relu3(self.conv3_2(self.conv3_1(p2)))
        p3 = self.pool3(c3)

        c4 = self.relu4(self.conv4_2(self.conv4_1(p3)))
        p4 = self.pool4(c4)

        c5 = self.relu5(self.conv5_2(self.conv5_1(p4)))

        u6 = self.upsample1(c5)
        u6 = torch.add(u6, c4)
        c6 = self.relu6(self.conv6_2(self.conv6_1(u6)))

        u7 = self.upsample2(c6)
        u7 = torch.add(u7, c3)
        c7 = self.relu7(self.conv7_2(self.conv7_1(u7)))

        u8 = self.upsample3(c7)
        u8 = torch.add(u8, c2)
        c8 = self.relu8(self.conv8_2(self.conv8_1(u8)))

        u9 = self.upsample4(c8)
        u9 = torch.add(u9, c1)
        c9 = self.relu9(self.conv9_2(self.conv9_1(u9)))

        c10 = self.relu10(self.conv10(c9))
        return c10


class UNETMax(nn.Module):
    """UNET Without concatenation during decoding"""

    def __init__(self):
        super(UNETMax, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)
        self.relu10 = nn.ReLU()

    def forward(self, x):
        c1 = self.relu1(self.conv1_2(self.conv1_1(x)))
        p1 = self.pool1(c1)

        c2 = self.relu2(self.conv2_2(self.conv2_1(p1)))
        p2 = self.pool2(c2)

        c3 = self.relu3(self.conv3_2(self.conv3_1(p2)))
        p3 = self.pool3(c3)

        c4 = self.relu4(self.conv4_2(self.conv4_1(p3)))
        p4 = self.pool4(c4)

        c5 = self.relu5(self.conv5_2(self.conv5_1(p4)))

        u6 = self.upsample1(c5)
        u6 = torch.max(u6, c4)
        c6 = self.relu6(self.conv6_2(self.conv6_1(u6)))

        u7 = self.upsample2(c6)
        u7 = torch.max(u7, c3)
        c7 = self.relu7(self.conv7_2(self.conv7_1(u7)))

        u8 = self.upsample3(c7)
        u8 = torch.max(u8, c2)
        c8 = self.relu8(self.conv8_2(self.conv8_1(u8)))

        u9 = self.upsample4(c8)
        u9 = torch.max(u9, c1)
        c9 = self.relu9(self.conv9_2(self.conv9_1(u9)))

        c10 = self.relu10(self.conv10(c9))
        return c10


class UNETMin(nn.Module):
    """UNET Without concatenation during decoding"""

    def __init__(self):
        super(UNETMin, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)
        self.relu10 = nn.ReLU()

    def forward(self, x):
        c1 = self.relu1(self.conv1_2(self.conv1_1(x)))
        p1 = self.pool1(c1)

        c2 = self.relu2(self.conv2_2(self.conv2_1(p1)))
        p2 = self.pool2(c2)

        c3 = self.relu3(self.conv3_2(self.conv3_1(p2)))
        p3 = self.pool3(c3)

        c4 = self.relu4(self.conv4_2(self.conv4_1(p3)))
        p4 = self.pool4(c4)

        c5 = self.relu5(self.conv5_2(self.conv5_1(p4)))

        u6 = self.upsample1(c5)
        u6 = torch.min(u6, c4)
        c6 = self.relu6(self.conv6_2(self.conv6_1(u6)))

        u7 = self.upsample2(c6)
        u7 = torch.min(u7, c3)
        c7 = self.relu7(self.conv7_2(self.conv7_1(u7)))

        u8 = self.upsample3(c7)
        u8 = torch.min(u8, c2)
        c8 = self.relu8(self.conv8_2(self.conv8_1(u8)))

        u9 = self.upsample4(c8)
        u9 = torch.min(u9, c1)
        c9 = self.relu9(self.conv9_2(self.conv9_1(u9)))

        c10 = self.relu10(self.conv10(c9))
        return c10


class FullyCNN(nn.Module):
    """UNET Without concatenation during decoding"""

    def __init__(self):
        super(FullyCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(
            in_channels=12, out_channels=6, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(
            in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(
            in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu7 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        return x


def train_model(model, trainloader, testloader, criterion, optimizer, device,
                nb_epoch):
    """Run this function to train the model"""

    train_losses, test_losses = [], []

    for epoch in range(nb_epoch):
        print("epoch", epoch)
        # Test the model
        test_loss = do_one_epoch(model=model, loader=testloader, train=False,
                                 criterion=criterion, device=device)
        test_losses.append(test_loss)
        print('test', test_loss)

        # Train the model
        train_loss = do_one_epoch(model=model, loader=trainloader, train=True,
                                  criterion=criterion, device=device,
                                  optimizer=optimizer)
        train_losses.append(train_loss)
        print('train', train_loss)

    return train_losses, test_losses


def do_one_epoch(model, loader, train, criterion, device, optimizer=None):

    losses = []
    for idx_batch, (X, Y) in enumerate(loader):

        if train:
            model.train()
        else:
            model.eval()

        # Zero optimizer grad
        if train:
            optimizer.zero_grad()

        # Zero model grad
        if train:
            model.zero_grad()

        # Copy to GPU
        gpu_X = X.to(device=device)
        gpu_Y = Y.to(device=device).view(-1, 128*128)

        # Output and loss computation
        gpu_output = model(gpu_X).view(-1, 2, 128*128)

        loss = criterion(gpu_output, gpu_Y)
        losses.append(float(loss))

        if train:
            # Backward step
            loss.backward()

            # Optimizer step
            optimizer.step()
    return np.mean(losses)


# Device
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Train dataset and loader
trainset = SaltDataset(root=data_path, train=True)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

testset = SaltDataset(root=data_path, train=False)
testloader = DataLoader(testset, batch_size=16, shuffle=True)

# Network
model = UNETMin()
print(model)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define criterion
criterion = torch.nn.CrossEntropyLoss()

# Train the network
print(train_model(model=model, trainloader=trainloader, testloader=testloader,
                  criterion=criterion, optimizer=optimizer, device=device,
                  nb_epoch=30))
