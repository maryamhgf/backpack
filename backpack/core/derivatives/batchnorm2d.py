from warnings import warn

from torch import einsum
from torch.nn import BatchNorm2d
import torch
from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class BatchNorm2dDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return BatchNorm2d

    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return False

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        return self._jac_t_mat_prod(module, g_inp, g_out, mat)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        return None

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        return None

    def get_normalized_input_and_var(self, module):
        input = module.input0
        mean = input.mean(dim=(0, 2, 3), keepdim=True)
        var = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        return (input - mean) / (var + module.eps).sqrt(), var

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch):
        # TODO: complete this function
        if not sum_batch:
            warn(
                "BatchNorm batch summation disabled."
                "This may not compute meaningful quantities"
            )

        x_hat, _ = self.get_normalized_input_and_var(module)
        equation = "vnihw,nihw->v{}i".format("" if sum_batch is True else "n")
        operands = [mat, x_hat]
        return einsum(equation, operands)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        return None

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        # TODO: complete this function
        if not sum_batch:
            warn(
                "BatchNorm batch summation disabled."
                "This may not compute meaningful quantities"
            )
            N_axis = 3,4 
            return mat.sum(N_axis)
        else:
            N_axis = 1
            return mat.sum(N_axis)
