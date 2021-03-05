import backpack.utils.linear as LinUtils
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.trial.trial_base import TRIALBaseModule


class TRIALLinear(TRIALBaseModule):
    def __init__(self, MODE):
        self.MODE = MODE
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    # TODO: FIX these functions for NGD
    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return LinUtils.extract_bias_ngd(module, backproped, self.MODE)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        return LinUtils.extract_weight_ngd(module, backproped, self.MODE)
