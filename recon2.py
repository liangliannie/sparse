# -*- coding: utf-8 -*-
import torch
import numpy as np
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import visdom
from torch.utils.data import Dataset, DataLoader
import pickle
dtype = torch.float
device = torch.device("cuda")

ww, ll = 128,128
N, D_in, H, D_out = 1, ww*ll , 1, ww*ll


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
        # print(self.img.shape)


    def __len__(self):
        return 36

    def __getitem__(self, idx):

        # reconimage = imread(data_dir + "/phantom.png", as_gray=True)
        if self.count >= 50000:
            self.count = 0
        reconimage = self.img[self.count].reshape(3,64,64)
        self.count +=1
        reconimage = reconimage[0]
        reconimage = rescale(reconimage, scale=2, mode='reflect', multichannel=False)
        theta = np.linspace(0., 180., max(reconimage.shape), endpoint=False)
        sinogram = radon(reconimage, theta=theta, circle=True)
        # sinogram, reconimage = np.expand_dims(sinogram, axis=0), np.expand_dims(reconimage, axis=0)

        sinogram = sinogram /sinogram.max()
        reconimage = reconimage /reconimage.max()
        sample = {'image': sinogram.flatten(), 'landmarks': reconimage.flatten()}

        return sample


# reconimage = imread(data_dir + "/phantom.png", as_gray=True)
# reconimage = rescale(reconimage, scale=0.4, mode='reflect', multichannel=False)
# theta = np.linspace(0., 180., max(reconimage.shape), endpoint=False)
# sinogram = radon(reconimage, theta=theta, circle=True)
#
# sinogram = sinogram /sinogram.max()
# reconimage = reconimage /reconimage.max()


# sino = torch.tensor(sinogram, requires_grad=True, device=device)
# img = torch.tensor(reconimage, requires_grad=True, device=device)
#
# sino, img = sino.type(torch.float32), img.type(torch.float32)

# k = img.clone().detach().requires_grad_(False)
# Pixel = torch.zeros(1, 160, 160, device=device, dtype=dtype, requires_grad=False)
# Pixel[0,0,0] = img[0,0,0]


# x = sino.reshape(1, -1)
# y = img.reshape(1, -1)
# # k = Pixel.reshape(1, -1)
#
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
w = torch.randn(D_in, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-5
vis = visdom.Visdom()
win = None
n=0

filter = torch.nn.Threshold(-2, 0, inplace=False)
batchsize = 12
transformed_dataset = ReconDataset('/home/liang/Desktop/imagenet/val_data')
dataloader = DataLoader(transformed_dataset, batch_size=batchsize,
                            shuffle=True)

for t in range(1000):
    # w = filter(w)
    # if t%20==10:
    #     print(w)
    #     w = filter(w)
    #     print(w)
    #     # w = torch.nn.functional.relu(w-0.1)+0.1
    #     w = w.clone().detach().requires_grad_(True)
    for i_batch, sample_batched in enumerate(dataloader):
        sino, img = (sample_batched['image']), (sample_batched['landmarks'])
        sino, img = sino.type(torch.float32).cuda(), img.type(torch.float32).cuda()
        print(sino.shape)
        out = sino.mm(w).clamp(min=0)
        loss = (out -img).pow(2).sum()
        # print(t, loss.item(), out.shape, y.shape, x.shape, w.shape)
        print(t, loss.item())
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            w.grad.zero_()
            learning_rate *= 0.985

        sino_show = sino.reshape(batchsize, ww, ll).detach().cpu().numpy()[0:1]
        img_show = img.reshape(batchsize, ww, ll).detach().cpu().numpy()[0:1]
        out_show = out.reshape(batchsize, ww, ll).detach().cpu().numpy()[0:1]
        w_show = w[n].reshape(1, ww, ll).detach().cpu().numpy()

        images = np.stack(
            [sino_show / sino_show.max() * 255, img_show / img_show.max() * 255, out_show / out_show.max() * 255,
             w_show / w_show.max() * 255])
        if not win:
            win = vis.images(images, padding=5, nrow=1, opts=dict(title='Sino, Img, Out, Weight'))
        else:
            vis.images(images, padding=5, win=win, nrow=1, opts=dict(title='Sino, Img, Out, Weight'))
