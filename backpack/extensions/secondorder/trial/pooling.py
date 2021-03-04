from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.extensions.secondorder.trial.trial_base import TRIALBaseModule


class TRIALMaxPool2d(TRIALBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())


class TRIALAvgPool2d(TRIALBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())
