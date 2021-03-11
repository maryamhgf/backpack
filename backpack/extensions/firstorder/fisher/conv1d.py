from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.firstorder.fisher.fisher_base import FisherBase


class FisherConv1d(FisherBase):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
