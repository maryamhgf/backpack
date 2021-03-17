from torch import einsum

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.fisher.fisher_base import FisherBase


class FisherLinear(FisherBase):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        """Compute second moments without expanding individual gradients.

        Overwrites the base class implementation that computes the gradient second
        moments from individual gradients. This approach is more memory-efficient.

        Note:
            For details, see page 12 (paragraph about "second moment") of the
            paper (https://arxiv.org/pdf/1912.10985.pdf).
        """
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]
        B =  einsum("ni,li->nl", (module.input0, module.input0))   
        A =  einsum("no,lo->nl", (g_out_sc, g_out_sc))
        return A * B

    def bias(self, ext, module, g_inp, g_out, backproped):
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]
        return einsum("no,lo->nl", g_out_sc, g_out_sc)
