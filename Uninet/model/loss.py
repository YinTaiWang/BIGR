from __future__ import annotations
import warnings
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks import one_hot
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


################################
#####     Registration     #####
################################
    
class NCC(nn.Module):
    '''
    Calculate local normalized cross-correlation coefficient between tow images. 

    Parameters
    ----------
    dim : int
        Dimension of the input images. 
    windows_size : int
        Side length of the square window to calculate the local NCC. 
    '''
    def __init__(self, dim, windows_size = 11):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.num_stab_const = 1e-5 # numerical stability constant
        
        self.windows_size = windows_size
        
        self.pad = windows_size//2
        self.window_volume = windows_size**self.dim
        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d
    
    def forward(self, I, J):
        '''
        Parameters
        ----------
        I and J : (n, 1, h, w) or (n, 1, d, h, w)
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used. 
        windows_size : int
            Side length of the square window to calculate the local NCC. 
            
        Returns
        -------
        NCC : scalar
            Average local normalized cross-correlation coefficient. 
        '''
        I = I.permute(1,0,2,3,4)
        J = J.permute(1,0,2,3,4)
        
        try:
            I_sum = self.conv(I, self.sum_filter, padding = self.pad)
        except:
            self.sum_filter = torch.ones([1, 1] + [self.windows_size, ]*self.dim, dtype = I.dtype, device = I.device)
            I_sum = self.conv(I, self.sum_filter, padding = self.pad)

        J_sum = self.conv(J, self.sum_filter, padding = self.pad) # (n, 1, h, w) or (n, 1, d, h, w)
        I2_sum = self.conv(I*I, self.sum_filter, padding = self.pad)
        J2_sum = self.conv(J*J, self.sum_filter, padding = self.pad)
        IJ_sum = self.conv(I*J, self.sum_filter, padding = self.pad)

        cross = torch.clamp(IJ_sum - I_sum*J_sum/self.window_volume, min = self.num_stab_const)
        I_var = torch.clamp(I2_sum - I_sum**2/self.window_volume, min = self.num_stab_const)
        J_var = torch.clamp(J2_sum - J_sum**2/self.window_volume, min = self.num_stab_const)

        cc = cross/((I_var*J_var)**0.5)
        
        return -torch.mean(cc)

class Grad(nn.Module):
    """
    N-D gradient loss.
    
    Adapted from:
        https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def forward(self, y_pred):
        # Convert MetaTensor to Tensor if necessary
        if isinstance(y_pred, torch.Tensor) and hasattr(y_pred, 'as_tensor'):
            y_pred = y_pred.as_tensor()
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


################################
#####     Segmentation     #####
################################

class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.
    
    Copied from:
        MONAI (https://docs.monai.io/en/stable/_modules/monai/losses/dice.html#DiceLoss)

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.weight = weight
        self.register_buffer("class_weight", torch.ones(1))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.weight is not None and target.shape[1] != 1:
            # make sure the lengths of weights are equal to the number of classes
            num_of_classes = target.shape[1]
            if isinstance(self.weight, (float, int)):
                self.class_weight = torch.as_tensor([self.weight] * num_of_classes)
            else:
                self.class_weight = torch.as_tensor(self.weight)
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            f = f * self.class_weight.to(f)

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