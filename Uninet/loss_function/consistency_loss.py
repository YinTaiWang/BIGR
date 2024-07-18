from __future__ import annotations

from .common_imports import *
from torch.nn.modules.loss import _Loss
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ConsistencyLoss(torch.nn.Module):
    """
    Compute average Dice loss between four tensors. Only works for binary segmentation.
    The data `input` (BNHW[D] where N is number of channel), 
    each two channel is a pair and are compared with each other.

    Assume passing logits as input, must set either `sigmoid=True` or `softmax=True`.

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    Adapted from:
        MONAI (https://docs.monai.io/en/stable/_modules/monai/losses/dice.html#DiceLoss)
    """

    def __init__(
        self,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ) -> None:
        
        super().__init__()
        self.squared_pred = squared_pred
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        channel = int(input.shape[1])
        
        intersection = input[:, 0, ...].clone()
        for i in range(1, channel):
            intersection = intersection * input[:, i, ...]

        reduce_axis = torch.arange(1, len(intersection.shape)).tolist()
        intersection = torch.sum(intersection, dim=reduce_axis)

        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.squared_pred:
            pred_o = torch.sum(input**2, dim=reduce_axis)
            pred_o = torch.sum(pred_o, dim=1)
        else:
            pred_o = torch.sum(input, dim=reduce_axis)
            pred_o = torch.sum(pred_o, dim=1)

        denominator = pred_o
        f = 1.0 - (channel * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        f = torch.mean(f)  # batch and channel average
        return f