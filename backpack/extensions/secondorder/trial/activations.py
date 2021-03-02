from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
# from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.extensions.secondorder.trial.trial_base import TRIALBaseModule


class TRIALReLU(TRIALBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class TRIALSigmoid(TRIALBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


# class DiagGGNTanh(DiagGGNBaseModule):
#     def __init__(self):
#         super().__init__(derivatives=TanhDerivatives())
