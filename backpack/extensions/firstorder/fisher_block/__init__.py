from torch.nn import (
    Conv1d,
    Conv2d,
    Linear,
    BatchNorm1d,
    BatchNorm2d
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    conv1d,
    conv2d,
    linear,
    batchnorm1d,
    batchnorm2d
)


class FisherBlock(BackpropExtension):
    

    def __init__(self, damping=1.0, alpha=0.95, low_rank=False, gamma=0.95):
        self.gamma = gamma
        self.damping = damping
        self.alpha =alpha
        self.low_rank = low_rank
        super().__init__(
            savefield="fisher_block",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.FisherBlockLinear(self.damping, self.alpha),
                Conv1d: conv1d.FisherBlockConv1d(self.damping),
                Conv2d: conv2d.FisherBlockConv2d(self.damping, self.low_rank, self.gamma),
                BatchNorm1d: batchnorm1d.FisherBlockBatchNorm1d(self.damping),
                BatchNorm2d: batchnorm2d.FisherBlockBatchNorm2d(self.damping),
            },
        )
