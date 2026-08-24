"""Losses used by BIQA regression and ranking experiments."""

from vnct.losses.correlation import LoDaPLCCLoss, PearsonCorrelationLoss

__all__ = ["LoDaPLCCLoss", "PearsonCorrelationLoss"]
