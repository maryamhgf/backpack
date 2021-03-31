from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.extensions.secondorder.mngd.mngd_base import MNGDBaseModule


class MNGDDropout(MNGDBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
