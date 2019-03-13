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


def sparse_linear(input, weight, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    # TODO change it into the sparse
    # weight = torch.sparse.FloatTensor(weight._indices(), weight._values(), torch.Size([25600,25600])).to_dense()
    weight = torch.sparse.FloatTensor(weight._indices(), weight._values(),
                                      torch.Size(weight.shape)).to_dense()
    if bias is not None:
        bias = torch.sparse.FloatTensor(bias._indices(), bias._values(),
                                        torch.Size([len(bias)])).to_dense()
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
        self.weight = Parameter(torch.randn(out_features, in_features).to_sparse().requires_grad_(True))
        if bias:
            self.bias = Parameter(torch.randn(out_features).to_sparse().requires_grad_(True))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.sparse.FloatTensor(self.weight._indices(), self.weight._values(),
                                          torch.Size([self.in_features, self.out_features])).to_dense()
        bias = torch.sparse.FloatTensor(self.bias._indices(), self.bias._values(),
                                        torch.Size([self.out_features])).to_dense()
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    @weak_script_method
    def forward(self, input):
        return sparse_linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


l = Linear(2, 2)
net = torch.nn.Sequential(Linear(2, 2))

optimizer = torch.optim.SparseAdam(net.parameters(), lr=1e-6, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985)


device = 'cuda'
x = torch.tensor([[1.0, 1.0],
                  [1.0, 2.1],
                  [1.0, 3.6],
                  [1.0, 4.2],
                  [1.0, 6.0],
                  [1.0, 7.0]], requires_grad=True, device=device)
y = torch.tensor([[1.0, 2.1, 3.6, 4.2, 6.0, 7.0],[1.0, 2.1, 3.6, 4.2, 6.0, 7.0]], requires_grad=True, device=device)
# y = x

net.cuda()
# weights = torch.zeros(2, 1, requires_grad=True, device=device)
# print(weights)

filter = torch.nn.Threshold(0.001, 0, inplace=False)

for i in range(500):
    # weights = filter(weights)
    # weights = torch.tensor(weights, requires_grad=True, device=device)
    # weights = weights.clone().detach().requires_grad_(True)
    # net_input = x.mm(weights)
    for name, para in net.named_parameters():
        # print(name, para)
        if 'weight' in name:
            weights = para
        if 'bias' in name:
            bias = para

    # print(weights)
    # print(net_input)
    net_input = net(x)
    loss = torch.mean((net_input - y.t()) ** 2)
    print(loss.item())
    # print(weights.grad)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # loss.backward()
    # weights.add_(-0.0001 * weights.grad.data)
    # weights.zero_grad()
    # if loss.data < 1e-3:
    #     break
print('n_iter', i)
# print(loss.data)