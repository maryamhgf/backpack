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


class Fisher(BackpropExtension):
    

    def __init__(self, silent=False):
        self.silent = silent
        super().__init__(
            savefield="fisher",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.FisherLinear(self.silent),
                Conv1d: conv1d.FisherConv1d(self.silent),
                Conv2d: conv2d.FisherConv2d(self.silent),
                BatchNorm1d: batchnorm1d.FisherBatchNorm1d(self.silent),
                BatchNorm2d: batchnorm2d.FisherBatchNorm2d(self.silent),
            },
        )
