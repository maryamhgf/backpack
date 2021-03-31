from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.firstorder.fisher_block.fisher_block_base import FisherBlockBase


class FisherBlockConv1d(FisherBlockBase):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
