import backpack.utils.linear as LinUtils
from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.secondorder.mngd.mngd_base import MNGDBaseModule
from torch import einsum
import torch

class MNGDBatchNorm1d(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"])

    # TODO: FIX these functions for NGD
    def weight(self, ext, module, grad_inp, grad_out, backproped):
        # dgamma = self.derivatives._weight_jac_t_mat_prod(module, grad_inp, grad_out, backproped, sum_batch=False)
        
        # # fake
        # new_bp = self.derivatives._my_jac_t_mat_prod(module, grad_inp, grad_out, backproped)
        # print('new_bp :\n', new_bp)

        # return einsum("vni,zqi->vnzq", (dgamma, dgamma))
        return None

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        # dbeta =  self.derivatives._bias_jac_t_mat_prod(module, grad_inp, grad_out, backproped, sum_batch=False)
        # print(torch.norm(dbeta))
        # fake
        # new_bp = self.derivatives._my_jac_t_mat_prod(module, grad_inp, grad_out, backproped)
        # print('new_bp bias:\n', new_bp)
        # return einsum("vni,zqi->vnzq", (dbeta, dbeta))
        return None