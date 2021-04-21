from torch.nn import (
    AvgPool2d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
    BatchNorm1d
)

from backpack.extensions.backprop_extension import BackpropExtension
# from backpack.extensions.secondorder.hbp import LossHessianStrategy

# from . import activations, conv2d, dropout, flatten, linear, losses, padding, pooling
from . import activations, linear, losses, conv2d, flatten, pooling, dropout, batchnorm1d


class MNGD(BackpropExtension):
    def __init__(self, savefield=None):
        if savefield is None:
            savefield = "mngd"

        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                # MSELoss: losses.DiagGGNMSELoss(),
                CrossEntropyLoss: losses.MNGDLoss(),
                Linear: linear.MNGDLinear(),
                MaxPool2d: pooling.MNGDMaxPool2d(),
                AvgPool2d: pooling.MNGDAvgPool2d(),
                # ZeroPad2d: padding.DiagGGNZeroPad2d(),
                Conv2d: conv2d.MNGDConv2d(),
                Dropout: dropout.MNGDDropout(),
                Flatten: flatten.MNGDFlatten(),
                ReLU: activations.MNGDReLU(),
                Sigmoid: activations.MNGDSigmoid(),
                BatchNorm1d: batchnorm1d.MNGDBatchNorm1d()
                # Tanh: activations.DiagGGNTanh(),
            },
        )


