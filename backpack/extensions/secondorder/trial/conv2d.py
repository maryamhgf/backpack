from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.trial.trial_base import TRIALBaseModule
from backpack.utils import conv as convUtils
from torch import sqrt, zeros
import torch

class TRIALConv2d(TRIALBaseModule):
    def __init__(self, MODE):
        self.MODE = MODE
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])
    
    # TODO: FIX these functions for NGD
    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped
        return convUtils.extract_bias_ngd(module, sqrt_ggn, self.MODE)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        # mask_shape = module.input0.shape
        # mask = self.create_mask_conv2d(module, mask_shape)
        # X = convUtils.unfold_func(module)(module.input0)
        # weight_diag = convUtils.extract_weight_ngd(module, X, backproped, mask)
        weight_diag = convUtils.extract_weight_ngd(module, backproped, self.MODE)
        return weight_diag


