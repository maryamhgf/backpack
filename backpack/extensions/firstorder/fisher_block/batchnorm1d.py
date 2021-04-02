from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.firstorder.fisher_block.fisher_block_base import FisherBlockBase

from torch import einsum, eye, matmul, ones_like, norm
from torch.linalg import inv

class FisherBlockBatchNorm1d(FisherBlockBase):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]
        G = g_out_sc

        I = module.input0
        mean = I.mean(dim=0)
        var = I.var(dim=0, unbiased=False)
        xhat = (I - mean) / (var + module.eps).sqrt()
        dw = g_out_sc * xhat

        # compute vector jacobian product in optimization method
        grad = module.weight.grad
        grad_prod = einsum("nk,k->n", (dw, grad))

        out = matmul(dw, dw.t())
        NGD_kernel = out / n
        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

        # gv = einsum("n,nk->k", (v, G))
        ### multiply with Jacobian
        gv = einsum("n,nk->k", (v, dw))
        gv = gv / n

        update = (grad - gv)/self.damping

        module.I = I
        module.G = G
        module.NGD_inv = NGD_inv

        return (out, grad_prod, update)


    def bias(self, ext, module, g_inp, g_out, backproped):
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]

        # compute vector jacobian product in optimization method
        grad = module.bias.grad
        grad_prod = einsum("no,o->n", (g_out_sc, grad))

        out = einsum("no,lo->nl", g_out_sc, g_out_sc)

        NGD_kernel = out / n
        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
        gv = einsum("n,no->o", (v, g_out_sc))
        gv = gv / n

        update = (grad - gv)/self.damping

        return (out, grad_prod, update)
        


