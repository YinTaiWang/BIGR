from __future__ import annotations

from .common_imports import *
from torch.nn.modules.loss import _Loss
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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