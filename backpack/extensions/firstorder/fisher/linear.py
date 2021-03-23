from torch import einsum

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.fisher.fisher_base import FisherBase


class FisherLinear(FisherBase):
    def __init__(self, silent=False):
        self.silent = silent
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):

        if not self.silent:
            grad = module.weight.grad
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]
            B =  einsum("ni,li->nl", (module.input0, module.input0))   
            A =  einsum("no,lo->nl", (g_out_sc, g_out_sc))

            # compute vector jacobian product in optimization method
            grad_prod = einsum("ni,oi->no", (module.input0, grad))
            grad_prod = einsum("no,no->n", (grad_prod, g_out_sc))

            return (A * B, grad_prod)
        else:
            grad = module.weight.grad
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]
            # compute vector jacobian product in optimization method
            grad_prod = einsum("ni,oi->no", (module.input0, grad))
            grad_prod = einsum("no,no->n", (grad_prod, g_out_sc))

            return grad_prod


    def bias(self, ext, module, g_inp, g_out, backproped):

        if not self.silent:
            grad = module.bias.grad
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            # compute vector jacobian product in optimization method
            grad_prod = einsum("no,o->n", (g_out_sc, grad))
            out = einsum("no,lo->nl", g_out_sc, g_out_sc)
            return (out, grad_prod)
        else:
            
            grad = module.bias.grad
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            # compute vector jacobian product in optimization method
            grad_prod = einsum("no,o->n", (g_out_sc, grad))
            return grad_prod

