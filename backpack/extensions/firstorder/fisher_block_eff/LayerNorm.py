import torch
import torch.nn as nn
from fisher_block_eff_base import FisherBlockEffBase
from backpack.core.derivatives.basederivatives import BaseParameterDerivatives

__all__ = ['LayerNormalization']

class LayerNormalization(FisherBlockEffBase):

    def __init__(self,
                 params,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-10):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNormalization, self).__init__(derivatives=BaseParameterDerivatives(), 
                                        params=params)
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        grad = module.weight.grad
        # print(grad.shape)
        grad_reshape = grad.reshape(grad.shape[0], -1)
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]

        y = self.forward(module.input0)

        #################################
        J = g_out_sc * x_hat
        J = J.reshape(J.shape[0], -1)
        JJT = torch.matmul(J, J.t())
        grad_prod =	torch.matmul(J, grad)
        NGD_kernel = JJT / n
        NGD_inv = torch.linalg.inv(NGD_kernel + self.damping * torch.eye(n).to(grad.device))
        v = torch.matmul(NGD_inv, grad_prod)
        gv = torch.matmul(J.t(), v) / n
        update = (grad - gv) / self.damping
        update = update.reshape(module.weight.grad.shape)
        module.I = I
        module.G = G
        module.NGD_inv = NGD_inv

        return update
        

    def bias(self, ext, module, g_inp, g_out, backproped):
        
        return module.bias.grad