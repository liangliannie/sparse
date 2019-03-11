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
import visdom
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch._jit_internal import weak_module, weak_script_method


# def to_torch_sparse_tensor(M):
#     M = M.tocoo().astype(np.float32)
#     indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
#     values = torch.from_numpy(M.data)
#     shape = torch.Size(M.shape)
#     T = torch.sparse.FloatTensor(indices, values, shape)
#     return T


def sparse_linear(input, weight, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    # TODO change it into the sparse
    # weight = torch.sparse.FloatTensor(weight._indices(), weight._values(), torch.Size([25600,25600])).to_dense()
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(torch.jit._unwrap_optional(bias), input, weight.t())
        # ret = torch.sparse.addmm(torch.jit._unwrap_optional(bias), input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += torch.jit._unwrap_optional(bias)
        ret = output
    return ret


class Linear(torch.nn.Module):
    # TODO change it into the sparse
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        #
        # weight = torch.zeros([out_features, in_features], dtype=torch.float32, requires_grad=True)
        # self.weight = weight.to_sparse().requires_grad_(True).cuda()
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # weight = torch.sparse.FloatTensor(self.weight._indices(), self.weight._values(), torch.Size([25600,25600])).to_dense()
        weight = self.weight
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    @weak_script_method
    def forward(self, input):
        # print(input, self.weight)
        return sparse_linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = Linear(160*160, 160*160, bias=False)

    def forward(self, x):
        x = x.reshape(1, -1)
        x = self.fc(x)
        x = x.reshape(1, 1, 160, 160)
        return x



class RepairNet():

    def __init__(self):
        self.network = Net()
        self.network.cuda()
        self.loss_func = torch.nn.L1Loss()
        self.loss_func_MSE = torch.nn.MSELoss()
        self.model_num = 0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_batch(self, input_img, target_img):
        output = self.network.forward(input_img)
        loss = self.loss_func(output, target_img) + self.loss_func_MSE(output, target_img)
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
    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = 'hello'
        reconimage = imread(data_dir + "/phantom.png", as_gray=True)
        reconimage = rescale(reconimage, scale=0.4, mode='reflect', multichannel=False)
        theta = np.linspace(0., 180., max(reconimage.shape), endpoint=False)
        sinogram = radon(reconimage, theta=theta, circle=True)
        self.sinogram, self.reconimage = np.expand_dims(sinogram, axis=0), np.expand_dims(reconimage, axis=0)

    def __len__(self):
        return 30

    def __getitem__(self, idx):
        sinogram = self.sinogram /self.sinogram.max()
        reconimage = self.reconimage /self.reconimage.max()
        sample = {'image': sinogram, 'landmarks': reconimage}
        return sample

if __name__ == "__main__":
    vis = visdom.Visdom()
    win = None
    w, l,n = 80, 80, 30
    TrainNet = RepairNet()
    optimizer = torch.optim.Adam(TrainNet.network.parameters(), lr=0.005, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985)
    TrainNet.set_optimizer(optimizer)

    transformed_dataset = ReconDataset()
    dataloader = DataLoader(transformed_dataset, batch_size=1,
                            shuffle=True)
    TrainNet.network.train()
    m = torch.nn.Threshold(0.5, float('inf'))


    for epoch in range(15):
        if epoch == 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print("Current learning rate: {}\n".format(param_group['lr']))
        print("Starting epoch: " + str(epoch) + '\n')
        for i_batch, sample_batched in enumerate(dataloader):
            sino, img = (sample_batched['image']), (sample_batched['landmarks'])
            sino, img = sino.type(torch.float32).cuda(), img.type(torch.float32).cuda()

            loss, out = TrainNet.train_batch(sino, img)
            k = torch.zeros(img.shape).cuda()
            k = img.detach()
            # k[0, 0, :, :] = img[0, 0, :, :]
            # print(k.shape, torch.inverse(TrainNet.network.fc3.weight).shape)
            pixel = k.flatten()
            #TODO change the sparse to make it visiable for one pixel
            for name, param in TrainNet.network.named_parameters():
                if 'weight' in name:
                    weight = param.data
                # elif 'bias' in name:
                #     bias = param.data
                # print(name, param.data.shape)
            # print(pixel-bias)
            # weight = TrainNet.network.named_parameters()
            # bias = TrainNet.network.fc.bias.data()
            # value = torch.mean(weight)
            print(weight)
            weight_dense = torch.Tensor(list(map(lambda x: 1/x if x > 0.1 else x, weight.flatten())))
            print(weight_dense.shape)
            # weight_dense = torch.inverse(m(weight))
            # weight_dense = torch.zeros(weight.shape)
            # for i in range(weight.shape[0]):
            #     for j in range(weight.shape[1]):
            #         if weight[i][j] >= value:
            #             weight_dense = 1.0/weight[i][j]

            one_pixel = torch.matmul(weight_dense.t(), pixel) #torch.inverse()
            one_pixel = one_pixel.reshape(1, 1, 160, 160)
            one_pixel_show = one_pixel[0].detach().cpu().numpy()
            sino_show = sino[0].detach().cpu().numpy()
            img_show = img[0].detach().cpu().numpy()
            out_show = out[0].detach().cpu().numpy()
            # one_pixel_show= out_show
            images = np.stack([sino_show/sino_show.max()*255, img_show/img_show.max()*255, out_show/out_show.max()*255, one_pixel_show/one_pixel_show.max()*255])
            if not win:
                win = vis.images(images, padding=5, nrow=1, opts=dict(title='Sino, Image'))
            else:
                vis.images(images, padding=5, win=win, nrow=1, opts=dict(title='Sino, Image'))
        if epoch == n:
            TrainNet.save_network()
