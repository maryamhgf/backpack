import backpack.utils.linear as LinUtils
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.mngd.mngd_base import MNGDBaseModule


class MNGDLinear(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    # TODO: FIX these functions for NGD
    def bias(self, ext, module, grad_inp, grad_out, backproped):
        # print(backproped.shape)
        # print(module.bias.grad)
        # return LinUtils.extract_bias_ngd(module, backproped, self.MODE)
        return None

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        # return LinUtils.extract_weight_ngd(module, backproped, self.MODE)
        return None
