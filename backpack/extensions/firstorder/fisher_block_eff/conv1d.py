from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.firstorder.fisher_block_eff.fisher_block_eff_base import FisherBlockEffBase


class FisherBlockEffConv1d(FisherBlockEffBase):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
