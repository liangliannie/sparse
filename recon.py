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
from collections import OrderedDict
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils
batch_size = 60
wl = 64
w, l, n = wl, wl, 30

# def to_torch_sparse_tensor(M):
#     M = M.tocoo().astype(np.float32)
#     indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
#     values = torch.from_numpy(M.data)
#     shape = torch.Size(M.shape)
#     T = torch.sparse.FloatTensor(indices, values, shape)
#     return T


#
# class Parameter(torch.Tensor):

#     def __new__(cls, data=None, requires_grad=True):
#         if data is None:
#             data = torch.Tensor()
#         return torch.Tensor._make_subclass(cls, data, requires_grad)
#
#     def __deepcopy__(self, memo):
#         if id(self) in memo:
#             return memo[id(self)]
#         else:
#             result = type(self)(self.data.clone(), self.requires_grad)
#             memo[id(self)] = result
#             return result
#
#     def __repr__(self):
#         return 'Parameter containing:\n' + super(Parameter, self).__repr__()
#
#     def __reduce_ex__(self, proto):
#         # See Note [Don't serialize hooks]
#         return (
#             torch._utils._rebuild_parameter,
#             (self.data, self.requires_grad, OrderedDict())
#         )

# def sparse_linear(input, weight, bias=None):
#     # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
#     # TODO change it into the sparse
#     # weight = torch.sparse.FloatTensor(weight._indices(), weight._values(), torch.Size([25600,25600])).to_dense()
#     if input.dim() == 2 and bias is not None:
#         # fused op is marginally faster
#         ret = torch.addmm(torch.jit._unwrap_optional(bias), input, weight.t())
#         # ret = torch.sparse.addmm(torch.jit._unwrap_optional(bias), input, weight.t())
#     else:
#         output = input.matmul(weight.t())
#         if bias is not None:
#             output += torch.jit._unwrap_optional(bias)
#         ret = output
#     return ret
#
#
# class Linear(torch.nn.Module):
#     # TODO change it into the sparse
#     __constants__ = ['bias']
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(Linear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)
#
#     @weak_script_method
#     def forward(self, input):
#         return sparse_linear(input, self.weight, self.bias)
#
#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )

class Net(torch.nn.Module):

    def __init__(self, shape1, shape2):
        super(Net, self).__init__()
        self.shape1 = shape1
        self.shape2 = shape2
        self.fc = torch.nn.Linear(shape1[0]*shape1[1], shape2[0]*shape2[1], bias=False)

    def forward(self, x, size):
        x = x.reshape(size, -1)
        x = self.fc(x)
        return x.reshape(size, w, l)



class RepairNet():

    def __init__(self, shape1, shape2):
        self.network = Net(shape1, shape2)
        self.network.cuda()
        self.loss_func = torch.nn.L1Loss()
        self.mssim_loss = MSSSIM(window_size=11, size_average=True)
        self.loss_func_MSE = torch.nn.MSELoss()
        self.model_num = 0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_batch(self, input_img, target_img, size):
        output = self.network.forward(input_img, size)
        # self.loss_func(output, target_img) +
        loss = self.loss_func(output, target_img) #- self.mssim_loss.forward(output.reshape(batch_size,1,w,l), target_img.reshape(batch_size,1,w,l))
        self.optimizer.zero_grad()
        loss.backward()
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
        # print(self.img.shape)


    def __len__(self):
        return 5*batch_size

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
        sample = {'sino': sinogram, 'img': reconimage}

        return sample

if __name__ == "__main__":
    vis = visdom.Visdom()
    win = None

    # n = 30

    TrainNet = RepairNet((w, l), (w, l))
    optimizer = torch.optim.Adam(TrainNet.network.parameters(), lr= 5e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3 , gamma=0.985)
    TrainNet.set_optimizer(optimizer)

    transformed_dataset = ReconDataset('/home/liang/Desktop/imagenet/val_data')
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=True)
    TrainNet.network.train()
    m = torch.nn.Threshold(0.5, float('inf'))

    for epoch in range(5000):
        if epoch%5 == 0:
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

            # TODO change the sparse to make it visiable for one pixel
            for name, param in TrainNet.network.named_parameters():
                if 'weight' in name:
                    weight = param.data

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

            w_show1 = weight[n].reshape(1, w, l).detach().cpu().numpy()
            w_show2 = weight[n+100].reshape(1, w, l).detach().cpu().numpy()
            w_show3 = weight[n+200].reshape(1, w, l).detach().cpu().numpy()


            images = np.stack(
                [sino_show_v / sino_show_v.max() * 255, img_show_v / img_show_v.max() * 255, out_show_v / out_show_v.max() * 255,
                 sino_show / sino_show.max() * 255, img_show / img_show.max() * 255, out_show / out_show.max() * 255,
                 w_show1 / w_show1.max() * 255, w_show2 / w_show2.max() * 255, w_show3 / w_show3.max() * 255])
            # images = np.stack(
            #     [sino_show_v / sino_show_v.max() * 255, img_show_v / img_show_v.max() * 255,
            #      out_show_v / out_show_v.max() * 255,
            #      w_show1 / w_show1.max() * 255, w_show2 / w_show2.max() * 255, w_show3 / w_show3.max() * 255])
            if not win:
                win = vis.images(images, padding=5, nrow=3, opts=dict(title='Sino, Img, Out, Weight'))
            else:
                vis.images(images, padding=5, win=win, nrow=3, opts=dict(title='Sino, Img, Out, Weight'))

        if epoch%50 == 0:
            TrainNet.save_network()

