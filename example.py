# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
from pytorch_msssim import MSSSIM
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import visdom
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch._jit_internal import weak_module, weak_script_method
import torch
import pickle
from torch_sparse import coalesce

wl = 16
w, l = wl,wl


class ReconDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root):
        self.root = root
        self.dic = self.unpickle(self.root)
        self.count = 0
        self.img = self.dic['data']

    def __len__(self):
        return 50*batch_size

    def __getitem__(self, idx):
        if self.count >= 50000:
            self.count = 0

        if self.count % 50 == 51:
            reconimage = imread(data_dir + "/phantom.png", as_gray=True)
            reconimage = rescale(reconimage, scale=w/400.0, mode='reflect', multichannel=False)
        else:
            reconimage = self.img[self.count].reshape(3, 64, 64)
            reconimage = reconimage[0]
            reconimage = rescale(reconimage, scale=w/64.0, mode='reflect', multichannel=False)

        self.count += 1
        theta = np.linspace(0., 180., max(reconimage.shape), endpoint=False)
        sinogram = radon(reconimage, theta=theta, circle=True)
        # sinogram, reconimage = np.expand_dims(sinogram, axis=0), np.expand_dims(reconimage, axis=0)

        sinogram = sinogram /sinogram.max()
        reconimage = reconimage /reconimage.max()
        sample = {'sino': sinogram.flatten(), 'img': reconimage.flatten()}

        return sample

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict


class DynamicNet(torch.nn.Module):

    def __init__(self, D_in, D_out, bias=True):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.in_features = D_in
        self.out_features = D_out
        self.weight = Parameter(torch.randn(self.out_features, self.in_features).to_sparse().requires_grad_(True))
        if bias:
            self.bias = Parameter(torch.randn(self.out_features).to_sparse().requires_grad_(True))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        y_pred = torch.sparse.mm(weight, x.t()).t()
        if self.bias:
            y_pred.add(bias)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, D_out = 64, w*l, w*l
batch_size = N

# # Create random Tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, D_out, bias=False)
# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SparseAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985)

weight = torch.randn(D_out, D_in).to_sparse().to('cuda').requires_grad_(True)
bias = torch.randn(D_out).to_sparse().requires_grad_(True).cuda()
filter = torch.nn.Threshold(0, 0, inplace=False)

transformed_dataset = ReconDataset('/home/liang/Desktop/imagenet/val_data')
dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=True)
learning_rate = 1e-5
vis = visdom.Visdom()
win = None
show = True
for t in range(500):
    for i_batch, sample_batched in enumerate(dataloader):
        sino, img = (sample_batched['sino']), (sample_batched['img'])
        sino, img = sino.type(torch.float32).cuda(), img.type(torch.float32).cuda()
        x, y = sino, img

        # out = model(x)
        out = torch.sparse.mm(weight, x.t()).t()
        loss = criterion(out, y)
        print(t, ': ', 'weight number: ', weight._nnz(), 'loss:', loss.item())
        # print(weight)
        # model.zero_grad()

        loss.backward()
        with torch.no_grad():
            weight -= learning_rate * weight.grad
            index, value = coalesce(weight._indices(), weight._values(), m=weight.shape[0], n=weight.shape[1])
            weight = torch.sparse.FloatTensor(index, value, torch.Size(weight.shape)).requires_grad_(True)
            # weight.coalesce()
            # print(weight.is_coalesced())
            # weight = filter(torch.sparse.FloatTensor(weight._indices(), weight._values(), torch.Size(weight.shape)).to_dense()).to_sparse().requires_grad_(True)

        learning_rate *= 0.985
            # weight.grad.zero_()

        if show:
            sino_show = x.reshape(N, w, l).detach().cpu().numpy()[0:1]
            img_show = y.reshape(N, w, l).detach().cpu().numpy()[0:1]
            out_show = out.reshape(N, w, l).detach().cpu().numpy()[0:1]
            ww = torch.sparse.FloatTensor(weight._indices(), weight._values(), torch.Size(weight.shape)).to_dense()
            w_show1 = ww[0].reshape(1, w, l).detach().cpu().numpy()
            w_show2 = ww[1].reshape(1, w, l).detach().cpu().numpy()
            w_show3 = ww[2].reshape(1, w, l).detach().cpu().numpy()

            images = np.stack(
                [sino_show / sino_show.max() * 255, img_show / img_show.max() * 255, out_show / out_show.max() * 255,
                 w_show1 / w_show1.max() * 255, w_show2 / w_show2.max() * 255, w_show3 / w_show3.max() * 255])
            if not win:
                win = vis.images(images, padding=5, nrow=3, opts=dict(title='Sino, Img, Out, Weight'))
            else:
                vis.images(images, padding=5, win=win, nrow=3, opts=dict(title='Sino, Img, Out, Weight'))
