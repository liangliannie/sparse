from __future__ import print_function, division
import os
from pytorch_msssim import MSSSIM
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import visdom
import sparse_adam
import adam
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch._jit_internal import weak_module, weak_script_method
import torch
import pickle
import torch_sparse
from collections import OrderedDict
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils
batch_size = 15
wl = 16
w, l, n = wl, wl, 30

class Net(torch.nn.Module):

    def __init__(self, shape1, shape2):
        super(Net, self).__init__()
        self.shape1 = shape1
        self.shape2 = shape2
        weight = init.kaiming_uniform_(torch.Tensor(wl*wl, wl*wl), a=math.sqrt(5))
        self.weight = torch.nn.Parameter(weight.to_sparse())
        # print(self.weight)

    def forward(self, x, size):

        x = x.reshape(size, -1)
        x = torch.sparse.mm(self.weight, x.t())

        return x.reshape(size, w, l)



class RepairNet():

    def __init__(self, shape1, shape2):
        self.network = Net(shape1, shape2)
        self.network.cuda()
        self.loss_func = torch.nn.L1Loss(reduction='mean')
        self.mssim_loss = MSSSIM(window_size=11, size_average=True)
        self.loss_func_MSE = torch.nn.MSELoss()
        self.model_num = 0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_batch(self, input_img, target_img, size):
        output = self.network.forward(input_img, size)
        loss = self.loss_func(output, target_img) #- self.mssim_loss.forward(output.reshape(batch_size,1,w,l), target_img.reshape(batch_size,1,w,l))

        self.optimizer.zero_grad()
        loss.backward()
        # weight = self.optimizer.param_groups[0]['params']
        self.optimizer.step()
        return loss.detach(), output

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

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def __init__(self, root):
        self.root = root
        self.dic = self.unpickle(self.root)
        self.count = 0
        self.img = self.dic['data']

    def __len__(self):
        return 5*batch_size

    def __getitem__(self, idx):
        if self.count >= 50000:
            self.count = 0

        if self.count % 1 == 2:
            reconimage = imread(data_dir + "/phantom.png", as_gray=True)
            reconimage = rescale(reconimage, scale=w/400.0, mode='reflect', multichannel=False)
        else:
            reconimage = self.img[self.count].reshape(3, 64, 64)
            reconimage = 0.2989*reconimage[0] + 0.5870*reconimage[1] + 0.1140*reconimage[2]
            reconimage = rescale(reconimage, scale=w/64.0, mode='reflect', multichannel=False)

        self.count += 1
        theta = np.linspace(0., 180., max(reconimage.shape), endpoint=False)
        sinogram = radon(reconimage, theta=theta, circle=True)

        sinogram = sinogram /sinogram.max()
        reconimage = reconimage /reconimage.max()
        sample = {'sino': sinogram, 'img': reconimage}

        return sample

if __name__ == "__main__":
    vis = visdom.Visdom()
    win = None

    TrainNet = RepairNet((w, l), (w, l))
    # optimizer = adam.Adam(TrainNet.network.parameters(), lr=1e-3, betas=(0.9, 0.999))
    optimizer = sparse_adam.SparseAdam(TrainNet.network.parameters(), lr=4e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985)
    TrainNet.set_optimizer(optimizer)

    transformed_dataset = ReconDataset('/home/liang/Desktop/imagenet/val_data')
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=True)
    TrainNet.network.train()


    for epoch in range(500000):

        if epoch % 5 == 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print("Current learning rate: {}\n".format(param_group['lr']))

        print("Starting epoch: " + str(epoch) + '\n')
        for i_batch, sample_batched in enumerate(dataloader):
            sino, img = (sample_batched['sino']), (sample_batched['img'])
            sino, img = sino.type(torch.float32).cuda(), img.type(torch.float32).cuda()
            loss, out = TrainNet.train_batch(sino, img, batch_size)
            print('loss:', loss.item())

        with torch.no_grad():
            for name, param in TrainNet.network.named_parameters():
                if 'weight' in name:
                    weight = param.data
        #     if epoch % 50 == 51:
        #         print('Revise the zeros in weight')
        #         # TODO change the sparse to make it visible for one pixel
        #         weight_min = weight.min()
        #         weight_dis = weight.max() - weight_min
        #         m = torch.nn.Threshold(weight_dis*0.1, 0)
        #         weight = m(weight-weight_min) + weight_min
        #         # optimizer.param_groups[0]['params'][0].data = weight
        #     for name, param in TrainNet.network.named_parameters():
        #         if 'weight' in name:
        #             param.data = weight
        #             # print(len(set(weight2 - weight)))

            testimage = imread(data_dir + "/phantom.png", as_gray=True)
            testimage = rescale(testimage, scale=w/400.0, mode='reflect', multichannel=False)
            theta = np.linspace(0., 180., max(testimage.shape), endpoint=False)
            tesgram = radon(testimage, theta=theta, circle=True)
            tesgram, testimage = torch.tensor(tesgram), torch.tensor(testimage)
            tesgram = tesgram / tesgram.max()
            testimage = testimage / testimage.max()
            tesgram, testimage = tesgram.type(torch.float32).cuda(), testimage.type(torch.float32).cuda()
            output = TrainNet.network.forward(tesgram, 1)

            sino_show_v = sino.reshape(batch_size, w, l).detach().cpu().numpy()[0:1]
            img_show_v = img.reshape(batch_size, w, l).detach().cpu().numpy()[0:1]
            out_show_v = out.reshape(batch_size, w, l).detach().cpu().numpy()[0:1]

            sino_show = tesgram.reshape(1, w, l).detach().cpu().numpy()[0:1]
            img_show = testimage.reshape(1, w, l).detach().cpu().numpy()[0:1]
            out_show = output.reshape(1, w, l).detach().cpu().numpy()[0:1]

            w_show1 = weight.to_dense()[n].reshape(1, w, l).detach().cpu().numpy()
            w_show2 = weight.to_dense()[n+20].reshape(1, w, l).detach().cpu().numpy()
            w_show3 = weight.to_dense()[n+10].reshape(1, w, l).detach().cpu().numpy()

            w_show1 = w_show1 - w_show1.min()
            w_show2 = w_show2 - w_show2.min()
            w_show3 = w_show3 - w_show3.min()

            images = np.stack(
                [sino_show_v / sino_show_v.max() * 255, img_show_v / img_show_v.max() * 255, out_show_v / out_show_v.max() * 255,
                 sino_show / sino_show.max() * 255, img_show / img_show.max() * 255, out_show / out_show.max() * 255,
                 w_show1 / w_show1.max() * 255, w_show2 / w_show2.max() * 255, w_show3 / w_show3.max() * 255])

            if not win:
                win = vis.images(images, padding=5, nrow=3, opts=dict(title='Sino, Img, Out, Weight'))
            else:
                vis.images(images, padding=5, win=win, nrow=3, opts=dict(title='Sino, Img, Out, Weight'))
