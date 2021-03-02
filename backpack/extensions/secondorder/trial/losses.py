from functools import partial

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.secondorder.trial.trial_base import TRIALBaseModule
from torch import softmax
from torch import rand


class TRIALLoss(TRIALBaseModule):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        hess_func = self.make_loss_hessian_func(ext)
        # print(hess_func(module, grad_inp, grad_out))
        return hess_func(module, grad_inp, grad_out)

    def make_loss_hessian_func(self, ext):
        """Get function that produces the backpropagated quantity."""
        return self.derivatives.sqrt_hessian


# class DiagGGNMSELoss(DiagGGNLoss):
#     def __init__(self):
#         super().__init__(derivatives=MSELossDerivatives())


# class TRIALCrossEntropyLoss(TRIALLoss):
#     def __init__(self):
#         super().__init__(derivatives=CrossEntropyLossDerivatives())


class TRIALCrossEntropyLoss(TRIALLoss):
    def __init__(self):
    	# set normalized to True for NGD
        super().__init__(derivatives=MSELossDerivatives(True))        
