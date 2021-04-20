from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.firstorder.fisher_block_eff.fisher_block_eff_base import FisherBlockEffBase

from torch import einsum, eye, matmul, ones_like, norm
from torch.linalg import inv

class FisherBlockEffBatchNorm1d(FisherBlockEffBase):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
    
        return module.weight.grad


    def bias(self, ext, module, g_inp, g_out, backproped):
        

        return module.bias.grad
        


