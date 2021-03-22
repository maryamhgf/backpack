from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.firstorder.fisher.fisher_base import FisherBase
from torch import einsum, matmul, sum, numel, sqrt, norm
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
            grad = module.weight.grad
            grad_reshape = grad.reshape(grad.shape[0], -1)
            # print(grad_reshape.shape)
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]
            input = unfold_func(module)(module.input0)
            grad_output_viewed = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)
            
            N = input.shape[0]
            K = input.shape[1]
            L = input.shape[2]
            M = grad_output_viewed.shape[1]

            # extra optimization for some networks such as VGG16
            if (L*L) * (K + M) < K * M :
                II = einsum("nkl,qkp->nqlp", (input, input))
                GG = einsum("nml,qmp->nqlp", (grad_output_viewed, grad_output_viewed))
                out = einsum('nqlp->nq', II * GG)                
                x1 = einsum("nkl,mk->nml", (input, grad_reshape))
                grad_prod = einsum("nml,nml->n", (x1, grad_output_viewed))
            else:
                AX = einsum("nkl,nml->nkm", (input, grad_output_viewed))
                # compute vector jacobian product in optimization method
                grad_prod = einsum("nkm,mk->n", (AX, grad_reshape))

                AX = AX.reshape(n , -1)
                out = matmul(AX, AX.t())

            return (out, grad_prod)
        elif MODE == 2:
            # st = time.time()

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
            # en = time.time()
            # print('Elapsed Time Conv2d Mode 2:', en - st)

            return K_torch

        elif MODE ==4: # using backpack class
            # st = time.time()

            grad_batch = self.derivatives.weight_jac_t_mat_prod(module, g_inp, g_out, g_out[0], sum_batch=False)
            grad_batch = grad_batch.reshape(grad_batch.shape[0], -1)
            out = matmul(grad_batch, grad_batch.t())
            # en = time.time()
            # print('Elapsed Time Conv2d Mode 4:', en - st)

            return out

        elif MODE == 6:
            return 0.
        else:
            raise NotImplementedError(
                        "Extension SUSPENDED")
            return 0
    
    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]

        # compute vector jacobian product in optimization method
        grad = module.bias.grad
        grad_prod = einsum("nchw,c->n", (g_out_sc, grad))

        out = einsum("nchw,lchw->nl", g_out_sc, g_out_sc)
        return (out, grad_prod)



