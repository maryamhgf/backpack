from functools import partial

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.secondorder.mngd.mngd_base import MNGDBaseModule
from torch import softmax
from torch import rand


class MNGDLoss(MNGDBaseModule):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        
        return rand(10,10)
