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


class TRIAL(BackpropExtension):
    def __init__(self, MODE, savefield=None):
        # print('MODE:', MODE)
        self.MODE = MODE
        if savefield is None:
            savefield = "trial"

        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                # MSELoss: losses.DiagGGNMSELoss(),
                CrossEntropyLoss: losses.TRIALCrossEntropyLoss(),
                Linear: linear.TRIALLinear(self.MODE),
                MaxPool2d: pooling.TRIALMaxPool2d(),
                AvgPool2d: pooling.TRIALAvgPool2d(),
                # ZeroPad2d: padding.DiagGGNZeroPad2d(),
                Conv2d: conv2d.TRIALConv2d(self.MODE),
                Dropout: dropout.TRIALDropout(),
                Flatten: flatten.TRIALFlatten(),
                ReLU: activations.TRIALReLU(),
                Sigmoid: activations.TRIALSigmoid(),
                BatchNorm1d: batchnorm1d.TRIALBatchNorm1d()
                # Tanh: activations.DiagGGNTanh(),
            },
        )


