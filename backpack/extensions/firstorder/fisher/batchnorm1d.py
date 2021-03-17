from torch import einsum
from torch import matmul
from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.firstorder.fisher.fisher_base import FisherBase


class FisherBatchNorm1d(FisherBase):
    def __init__(self):
        super().__init__(derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]

        input = module.input0
        mean = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)
        xhat = (input - mean) / (var + module.eps).sqrt()

        dw = g_out_sc * xhat
        return matmul(dw, dw.t())

    def bias(self, ext, module, g_inp, g_out, backproped):
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]
        return einsum("no,lo->nl", g_out_sc, g_out_sc)

