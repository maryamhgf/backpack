from backpack.core.derivatives.batchnorm2d import BatchNorm2dDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradBatchNorm2d(BatchGradBase):
    def __init__(self):
        super().__init__(
            derivatives=BatchNorm2dDerivatives(), params=["bias", "weight"]
        )
