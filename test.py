import torch

a = torch.randn(200, 300).to_sparse().requires_grad_(True)
b = torch.randn(300, 200, requires_grad=True).t()
a_dense = torch.tensor(a.to_dense()).requires_grad_()
# print(a.to_dense(), a_dense)
y1 = torch.sparse.mm(a, b.t())

# print('yyy', y)
y1.sum().backward()
# print(a.grad.to_dense())

y = b.matmul(a_dense.t())
y.sum().backward()
# print('yyy', y)
# print(a_dense.grad)

print(y-y1.t())
# print(a_dense.grad)