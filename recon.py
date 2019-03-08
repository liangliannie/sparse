from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class RepairNet():

    def __init__(self):
        self.network = Net()
        # self.network.cuda()
        self.loss_func = torch.nn.L1Loss()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_batch(self, input_img, target_img):
        output = self.network.forward(input_img)
        loss = self.loss_func(output, target_img)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_network(self):
        print("saving network parameters")
        folder_path = os.path.join('/home/liang/PycharmProjects/sparse/output', 'model')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.network.state_dict(), os.path.join(folder_path, "model_dict_{}".format(self.model_num)))
        self.model_num += 1
        if self.model_num >= 5: self.model_num = 0


class ReconDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = 'hello'

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        reconimage = imread(data_dir + "/phantom.png", as_gray=True)
        reconimage = rescale(reconimage, scale=0.4, mode='reflect', multichannel=False)
        theta = np.linspace(0., 180., max(reconimage.shape), endpoint=False)
        sinogram = radon(reconimage, theta=theta, circle=True)
        sinogram, reconimage = np.expand_dims(sinogram, axis=0), np.expand_dims(reconimage, axis=0)
        sinogram = sinogram/sinogram.max()
        reconimage = reconimage/reconimage.max()
        sample = {'image': sinogram, 'landmarks': reconimage}

        return sample

if __name__ == "__main__":
    TrainNet = RepairNet()
    optimizer = torch.optim.Adam(TrainNet.network.parameters(), lr=0.005, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985)
    TrainNet.set_optimizer(optimizer)

    transformed_dataset = ReconDataset()
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        sino, img = (sample_batched['image']), (sample_batched['landmarks'])
        sino, img = sino.type(torch.float32), img.type(torch.float32)

        TrainNet.train_batch(sino, img)

        print(sino.shape, img.shape)
