from torch import einsum
from torch import matmul
from backpack.core.derivatives.batchnorm2d import BatchNorm2dDerivatives
from backpack.extensions.firstorder.fisher.fisher_base import FisherBase
# import time

class FisherBatchNorm2d(FisherBase):
    def __init__(self, silent):
        self.silent = silent
        super().__init__(derivatives=BatchNorm2dDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        if not self.silent:
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            input = module.input0
            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            var = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            xhat = (input - mean) / (var + module.eps).sqrt()
            dw = g_out_sc * xhat
            out = einsum("nihw,lihw->nl", dw, dw)

            # compute vector jacobian product in optimization method
            grad = module.weight.grad
            grad_prod = einsum("nihw,i->n", (dw, grad))

            # en = time.time()
            # print('Elapsed Time in BatchNorm2d:', en - st)
            return (out, grad_prod)
        else:
            # st = time.time()
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            input = module.input0
            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            var = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            xhat = (input - mean) / (var + module.eps).sqrt()
            dw = g_out_sc * xhat

            # compute vector jacobian product in optimization method
            grad = module.weight.grad
            grad_prod = einsum("nihw,i->n", (dw, grad))
            return grad_prod


    def bias(self, ext, module, g_inp, g_out, backproped):
        if not self.silent:
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            # compute vector jacobian product in optimization method
            grad = module.bias.grad
            grad_prod = einsum("nihw,i->n", (g_out_sc, grad))

            out = einsum("nihw,lihw->nl", g_out_sc, g_out_sc)
            return (out, grad_prod)
        else:
            # print('x'*100)

            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            # compute vector jacobian product in optimization method
            grad = module.bias.grad
            grad_prod = einsum("nihw,i->n", (g_out_sc, grad))

            return  grad_prod


