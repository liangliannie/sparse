import torch
import torch.nn as nn
from synapses import SETLayer


inp = torch.ones((32, 1024)).cuda()

layer = SETLayer(1024, 1024).cuda()

for name, para in layer.named_parameters():
    print(name, para.data.shape)

optimizer = torch.optim.Adam(layer.parameters(), lr=5e-3, betas=(0.9, 0.999))
# optimizer = torch.optim.SGD(layer.parameters(), lr=0.01, momentum=0.9)

out = layer(inp)
# tot = out.sum()

# criterion = nn.MSELoss()
# tr = torch.tensor(0).float()
loss_func = nn.L1Loss(reduction='mean')
loss = loss_func(inp, out)

loss.backward(retain_graph=True)

optimizer.step()
for name, para in layer.named_parameters():
    print(name, para.grad.shape)
layer.evolve_connections()
print(loss)
