from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.firstorder.fisher.fisher_base import FisherBase
from torch import einsum, matmul, sum, numel, sqrt
from torch.nn import Unfold
from torch.nn.functional import conv1d, conv2d, conv3d
from backpack.utils.ein import eingroup
from backpack.utils.conv import unfold_func

MODE = 0
class FisherConv2d(FisherBase):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        if MODE == 0: # my implementation
            n = g_out[0].shape[0]

            g_out_sc = n * g_out[0]
            input = unfold_func(module)(module.input0)
            grad_output_viewed = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)
            # print(sum(module.input0))
            # print('inp::::::::::::\n', (module.input0))
            # print('inp:', sum(module.input0))
            # print('mean:', sum(grad_output_viewed[0,...])/numel(grad_output_viewed[0,...]))
            # print('mean:', sum(module.input0[0,...])/numel(module.input0[0,...]))
            # print(sqrt(sum(grad_output_viewed**2))/numel(grad_output_viewed))
            # print(sqrt(sum(module.input0[0,...]**2))/numel(module.input0[0,...]))
            # print((grad_output_viewed))
            # print(sum(grad_output_viewed**2))
            AX = einsum("nkl,nml->nkm", (input, grad_output_viewed))
            AX = AX.reshape(n , -1)
            return matmul(AX, AX.t())
        elif MODE == 2:
            A = module.input0
            n = A.shape[0]
            p = 1
            M = g_out[0]

            M = M.reshape( M.shape[1] * M.shape[0], M.shape[2], M.shape[3]).unsqueeze(1)
            A = A.permute(1 ,0, 2, 3)
            output = conv2d(A, M, groups = n, padding = (p,p))
            output = output.permute(1, 0, 2, 3)
            output = output.reshape(n, -1)
            K_torch = matmul(output, output.t())
            return K_torch

        elif MODE ==4: # using backpack class
        	grad_batch = self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, g_out[0], sum_batch=False)
        	grad_batch = grad_batch.reshape(grad_batch.shape[0], -1)
        	return matmul(grad_batch, grad_batch.t())
        else:
            raise NotImplementedError(
                        "Extension SUSPENDED")
            return 0
    
    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        return einsum("nchw,lchw->nl", g_out[0], g_out[0])



