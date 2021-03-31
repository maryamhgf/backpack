from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.mngd.mngd_base import MNGDBaseModule
from backpack.utils import conv as convUtils
from torch import sqrt, zeros
import torch

from torch import einsum

class MNGDConv2d(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])
    
    # TODO: FIX these functions for NGD
    def bias(self, ext, module, grad_inp, grad_out, backproped):
        # sqrt_ggn = backproped
        # return convUtils.extract_bias_ngd(module, sqrt_ggn, self.MODE)
        return None

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        # weight_diag= convUtils.extract_weight_ngd(module, backproped, self.MODE)
        # return weight_diag
        return None


