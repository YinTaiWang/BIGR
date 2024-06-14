from __future__ import annotations

from .common_imports import *
from torch.nn.modules.loss import _Loss
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ConsistencyLoss(_Loss):
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
        include_background: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.

        Raises:
            ValueError: When less or more than 1 of [``sigmoid=True``, ``softmax=True``].

        """
        
        super().__init__(reduction=LossReduction(reduction).value)
        if int(sigmoid) + int(softmax) < 1:
            raise ValueError("Must set either `sigmoid=True` or `softmax=True`.")
        if int(sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True].")
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D], where N is the number of channel.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        
        n_ch = input.shape[1] # BNHWD
        assert n_ch % 2 == 0, "the number of channel should be an even number. found: %d" % n_ch
        n_groups = int(input.shape[1]/2)
        
        # Apply activate function onto each pair
        # The new group dimension is [Batch, number of groups, 2, ...spatial dimensions...]
        if self.sigmoid:
            input = torch.sigmoid(input)
            input = input.view(1, n_groups, 2, *input.shape[2:])

        if self.softmax:
            input = input.view(1, n_groups, 2, *input.shape[2:])
            input = torch.softmax(input, dim=2)

        if not self.include_background:
            # if skipping background, removing first channel
            input = input[:, :, 1, ...]
        
        intersection = input[:, 0, ...].clone()
        for i in range(1, input.shape[1]):
            intersection = intersection * input[:, i, ...]
        # intersection shape B2HW[D]
        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(intersection.shape)).tolist()
        intersection = torch.sum(intersection, dim=reduce_axis)
        
        # input shape BG2HW[D]
        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(3, len(input.shape)).tolist()
        if self.squared_pred:
            pred_o = torch.sum(input**2, dim=reduce_axis)
            pred_o = torch.sum(pred_o, dim=1)
        else:
            pred_o = torch.sum(input, dim=reduce_axis)
            pred_o = torch.sum(pred_o, dim=1)

        denominator = pred_o

        f: torch.Tensor = 1.0 - (n_groups * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f