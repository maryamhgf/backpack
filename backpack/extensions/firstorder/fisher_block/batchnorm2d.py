from backpack.core.derivatives.batchnorm2d import BatchNorm2dDerivatives
from backpack.extensions.firstorder.fisher_block.fisher_block_base import FisherBlockBase

from torch import einsum, eye, matmul, ones_like, norm
from torch.linalg import inv

class FisherBlockBatchNorm2d(FisherBlockBase):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=BatchNorm2dDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        
        return module.weight.grad
        

    def bias(self, ext, module, g_inp, g_out, backproped):
        
        return module.bias.grad
        

