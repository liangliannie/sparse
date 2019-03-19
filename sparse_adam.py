import math
import torch
from torch.optim.optimizer import Optimizer

# from torch_sparse import coalesce

class SparseAdam(Optimizer):
    r"""Implements lazy version of Adam algorithm suitable for sparse tensors.

    In this variant, only moments that show up in the gradient get updated, and
    only those portions of the gradient get applied to the parameters.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if not grad.is_sparse:
                    raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # print(p.data.size())
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                print('nnz####### ', p.data._nnz())
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                grad = grad.coalesce()  # the update is non-linear so indices must be unique
                exp_avg, exp_avg_sq = exp_avg.coalesce(), exp_avg_sq.coalesce()

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1-beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1-beta2, grad.mul_(grad))

                numer = exp_avg.to_dense()
                denom = exp_avg_sq.to_dense().sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.add_((-step_size * numer.div_(denom)).sparse_mask(grad))
                del numer, denom

                p.data = p.data.coalesce()

                # if state['step'] % 200 == 0:
                #     weight = p.data
                #     print('Before:', p.data._nnz())
                #     dense_weight = weight.to_dense()
                #     mask = torch.lt(abs(dense_weight), 0.01)
                #     dense_weight[mask] = 0
                #     p.data = dense_weight.to_sparse().to('cuda').requires_grad_(True)
                #     print('After:', p.data._nnz())


        return loss
