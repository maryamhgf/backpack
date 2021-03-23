from torch import einsum
from torch import matmul
from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.firstorder.fisher.fisher_base import FisherBase


class FisherBatchNorm1d(FisherBase):
    def __init__(self, silent):
        self.silent = silent
        super().__init__(derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        if not self.silent:
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            input = module.input0
            mean = input.mean(dim=0)
            var = input.var(dim=0, unbiased=False)
            xhat = (input - mean) / (var + module.eps).sqrt()
            dw = g_out_sc * xhat

            # compute vector jacobian product in optimization method
            grad = module.weight.grad
            grad_prod = einsum("nk,k->n", (dw, grad))

            return (0.* matmul(dw, dw.t()), grad_prod)
        else:
            
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            input = module.input0
            mean = input.mean(dim=0)
            var = input.var(dim=0, unbiased=False)
            xhat = (input - mean) / (var + module.eps).sqrt()
            dw = g_out_sc * xhat

            # compute vector jacobian product in optimization method
            grad = module.weight.grad
            grad_prod = einsum("nk,k->n", (dw, grad))

            return grad_prod


    def bias(self, ext, module, g_inp, g_out, backproped):
        if not self.silent:
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            # compute vector jacobian product in optimization method
            grad = module.bias.grad
            grad_prod = einsum("no,o->n", (g_out_sc, grad))

            out = einsum("no,lo->nl", g_out_sc, g_out_sc)
            return (0.*out, grad_prod)
        else:
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            # compute vector jacobian product in optimization method
            grad = module.bias.grad
            grad_prod = einsum("no,o->n", (g_out_sc, grad))

            return grad_prod


