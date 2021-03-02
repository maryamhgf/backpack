"""
BackPACK Extensions
"""

from .curvmatprod import GGNMP, HMP, PCHMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, Variance
from .secondorder import (
    TRIAL,
    HBP,
    KFAC,
    KFLR,
    KFRA,
    DiagGGN,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
)

__all__ = [
    "TRIAL",
    "PCHMP",
    "GGNMP",
    "HMP",
    "BatchL2Grad",
    "BatchGrad",
    "SumGradSquared",
    "Variance",
    "KFAC",
    "KFLR",
    "KFRA",
    "HBP",
    "DiagGGNExact",
    "DiagGGNMC",
    "DiagGGN",
    "DiagHessian",
]
