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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.conv5_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)

    def forward(self, x):
        c1 = F.relu(self.conv1_2(self.conv1_1(x)))
        p1 = self.pool1(c1)

        c2 = F.relu(self.conv2_2(self.conv2_1(p1)))
        p2 = self.pool2(c2)

        c3 = F.relu(self.conv3_2(self.conv3_1(p2)))
        p3 = self.pool3(c3)

        c4 = F.relu(self.conv4_2(self.conv4_1(p3)))
        p4 = self.pool4(c4)

        c5 = F.relu(self.conv5_2(self.conv5_1(p4)))

        u6 = self.upsample1(c5)
        u6 = torch.cat((u6, c4), 1)
        c6 = F.relu(self.conv6_2(self.conv6_1(u6)))

        u7 = self.upsample2(c6)
        u7 = torch.cat((u7, c3), 1)
        c7 = F.relu(self.conv7_2(self.conv7_1(u7)))

        u8 = self.upsample3(c7)
        u8 = torch.cat((u8, c2), 1)
        c8 = F.relu(self.conv8_2(self.conv8_1(u8)))

        u9 = self.upsample4(c8)
        u9 = torch.cat((u9, c1), 1)
        c9 = F.relu(self.conv9_2(self.conv9_1(u9)))

        c10 = F.relu(self.conv10(c9))
        return c10


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.conv5_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)

    def forward(self, x):
        c1 = F.relu(self.conv1_2(self.conv1_1(x)))
        p1 = self.pool1(c1)

        c2 = F.relu(self.conv2_2(self.conv2_1(p1)))
        p2 = self.pool2(c2)

        c3 = F.relu(self.conv3_2(self.conv3_1(p2)))
        p3 = self.pool3(c3)

        c4 = F.relu(self.conv4_2(self.conv4_1(p3)))
        p4 = self.pool4(c4)

        c5 = F.relu(self.conv5_2(self.conv5_1(p4)))

        u6 = self.upsample1(c5)
        u6 = torch.cat((u6, c4), 1)
        c6 = F.relu(self.conv6_2(self.conv6_1(u6)))

        u7 = self.upsample2(c6)
        u7 = torch.cat((u7, c3), 1)
        c7 = F.relu(self.conv7_2(self.conv7_1(u7)))

        u8 = self.upsample3(c7)
        u8 = torch.cat((u8, c2), 1)
        c8 = F.relu(self.conv8_2(self.conv8_1(u8)))

        u9 = self.upsample4(c8)
        u9 = torch.cat((u9, c1), 1)
        c9 = F.relu(self.conv9_2(self.conv9_1(u9)))

        c10 = F.relu(self.conv10(c9))
        return c10


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

        # Train the model
        train_loss = do_one_epoch(model=model, loader=trainloader, train=True,
                                  criterion=criterion, device=device,
                                  optimizer=optimizer)
        train_losses.append(train_loss)

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
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = SaltDataset(root=data_path, train=False)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

# Network
model = NeuralNetwork2()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define criterion
criterion = torch.nn.CrossEntropyLoss()

# Train the network
print(train_model(model=model, trainloader=trainloader, testloader=testloader,
                  criterion=criterion, optimizer=optimizer, device=device,
                  nb_epoch=30))
