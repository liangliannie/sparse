
import torch
from torch import nn

class TrainNet(torch.nn.Module):


    def __init__(self, in_features, out_features, bias=True):
        super(TrainNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features).to_sparse().requires_grad_(True))
        self.bias = torch.nn.Parameter(torch.randn(out_features, out_features).to_sparse().requires_grad_(True))


    def forward(self, input):

        # weight = self.weight.to_dense().requires_grad_(True)
        x = torch.sparse.mm(self.weight, input)
        # x = torch.sparse.addmm(self.weight,input)
        # x = torch.nn.functional.linear(input, weight, None)

        return x


# a = torch.randn(2,3).to_sparse().requires_grad_(True)
x = torch.randn(3,3, requires_grad=False)
y = torch.randn(3,3, requires_grad=False)
Net = TrainNet(3,3)
# for para in Net.parameters():
#     print(para)
optimizer = torch.optim.SparseAdam(Net.parameters(), lr=1e-3, betas=(0.9, 0.999))
l1 = torch.nn.L1Loss(reduction='sum')
for epoth in range(5000):
    out = Net(x)
    loss = l1(out, y)
    # loss = out.sum()
    # print(Net.weight.grad)
    # Net.weight.grad.zero_()
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# param = nn.Parameter(x) #param does not have backward information now!
# optimizer = torch.optim.sparse_adam(param)
# print(param.data)
