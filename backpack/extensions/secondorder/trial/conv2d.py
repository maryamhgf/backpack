from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.trial.trial_base import TRIALBaseModule
from backpack.utils import conv as convUtils
from torch import sqrt, zeros
import torch

from torch import einsum

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
        if self.MODE == 666: # not good because of repeating
            dw = self.derivatives.weight_jac_t_mat_prod(module, grad_inp, grad_out, backproped, sum_batch=False)
            dw = dw.reshape(dw.shape[0], dw.shape[1], dw.shape[2], -1)
            res_ = dw.permute(0,1,3,2)
            return einsum("vnkm,zqkm->vnzq", (res_, res_))
        else:
            weight_diag= convUtils.extract_weight_ngd(module, backproped, self.MODE)
            return weight_diag


