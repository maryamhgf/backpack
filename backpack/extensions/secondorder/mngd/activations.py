from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.extensions.secondorder.mngd.mngd_base import MNGDBaseModule


class MNGDReLU(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class MNGDSigmoid(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


# class DiagGGNTanh(DiagGGNBaseModule):
#     def __init__(self):
#         super().__init__(derivatives=TanhDerivatives())
