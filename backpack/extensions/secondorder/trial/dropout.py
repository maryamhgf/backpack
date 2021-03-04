from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.extensions.secondorder.trial.trial_base import TRIALBaseModule


class TRIALDropout(TRIALBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
