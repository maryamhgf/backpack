from backpack.core.derivatives.batchnorm2d import BatchNorm2dDerivatives
from backpack.extensions.firstorder.fisher_block.fisher_block_base import FisherBlockBase

from torch import einsum, eye, matmul, ones_like, norm
from torch.linalg import inv

class FisherBlockBatchNorm2d(FisherBlockBase):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=BatchNorm2dDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]
        G = g_out_sc

        I = module.input0
        mean = I.mean(dim=(0, 2, 3), keepdim=True)
        var = I.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        xhat = (I - mean) / (var + module.eps).sqrt()
        dw = g_out_sc * xhat
        out = einsum("nihw,lihw->nl", dw, dw)

        # compute vector jacobian product in optimization method
        grad = module.weight.grad
        dw_reduced = einsum("nihw->ni", dw)
        # grad_prod = einsum("nihw,i->n", (dw, grad))
        grad_prod = einsum("ni,i->n", (dw_reduced, grad))

        NGD_kernel = out / n
        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

        # gv = einsum("n,nihw->i", (v, G))
        ### multiply with Jacobian
        gv = einsum("n,ni->i", (v, dw_reduced))
        gv = gv / n

        update = (grad - gv)/self.damping

        module.dw = dw_reduced
        module.NGD_inv = NGD_inv
        return (out, grad_prod, update)
        

    def bias(self, ext, module, g_inp, g_out, backproped):
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]
        g_out_sc_reduced = einsum("nihw->ni", g_out_sc)
        # compute vector jacobian product in optimization method
        grad = module.bias.grad
        grad_prod = einsum("ni,i->n", (g_out_sc_reduced, grad))

        out = einsum("nihw,lihw->nl", g_out_sc, g_out_sc)

        NGD_kernel = out / n
        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
        gv = einsum("n,ni->i", (v, g_out_sc_reduced))
        gv = gv / n

        update = (grad - gv)/self.damping
        return (out, grad_prod, update)
        

