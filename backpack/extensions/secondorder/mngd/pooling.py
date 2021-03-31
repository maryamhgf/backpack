from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.extensions.secondorder.mngd.mngd_base import MNGDBaseModule


class MNGDMaxPool2d(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())


class MNGDAvgPool2d(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())
