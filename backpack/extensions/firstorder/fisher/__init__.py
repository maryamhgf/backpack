from torch.nn import (
    Conv1d,
    Conv2d,
    Linear,
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    conv1d,
    conv2d,
    linear,
)


class Fisher(BackpropExtension):
    

    def __init__(self):
        super().__init__(
            savefield="fisher",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.FisherLinear(),
                Conv1d: conv1d.FisherConv1d(),
                Conv2d: conv2d.FisherConv2d(),
            },
        )
