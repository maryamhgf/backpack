from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from torch import matmul

class FisherBase(FirstOrderModuleExtension):
    def __init__(self, derivatives, params=None):
        self.derivatives = derivatives
        self.N_axis = 0
        super().__init__(params=params)
    # fix these for NGD
    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        grad_batch = self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        n = grad_batch.shape[0]
        grad_batch = n * grad_batch.reshape(grad_batch.shape[0], -1)

        return matmul(grad_batch, grad_batch.t())

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        grad_batch = self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False
        )
        n = grad_batch.shape[0]
        grad_batch = n * grad_batch.reshape(grad_batch.shape[0], -1)

        return matmul(grad_batch, grad_batch.t())
