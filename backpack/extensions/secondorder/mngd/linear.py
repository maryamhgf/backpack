import backpack.utils.linear as LinUtils
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.mngd.mngd_base import MNGDBaseModule


class MNGDLinear(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    # TODO: FIX these functions for NGD
    def bias(self, ext, module, grad_inp, grad_out, backproped):
        grad = module.bias.grad
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]

        # compute vector jacobian product in optimization method
        grad_prod = einsum("no,o->n", (g_out_sc, grad))
        out = einsum("no,lo->nl", g_out_sc, g_out_sc)

        NGD_kernel = out / n
        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
        gv = einsum("n,no->o", (v, g_out_sc))
        gv = gv / n

        update = (grad - gv)/self.damping
        return (out, grad_prod, update)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        I = module.input0
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]
        G = g_out_sc
        grad = module.weight.grad
        
        
        B =  einsum("ni,li->nl", (I, I))   
        A =  einsum("no,lo->nl", (G, G))

        # compute vector jacobian product in optimization method
        grad_prod = einsum("ni,oi->no", (I, grad))
        grad_prod = einsum("no,no->n", (grad_prod, G))
        out = A * B 
        NGD_kernel = out / n
        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

        gv = einsum("n,no->no", (v, G))
        gv = einsum("no,ni->oi", (gv, I))
        gv = gv / n

        update = (grad - gv)/self.damping
        
        module.I = I
        module.G = G
        module.NGD_inv = NGD_inv
        return (out, grad_prod, update)
