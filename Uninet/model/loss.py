import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''
Adapt from: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py#L15
'''

class NCC(nn.Module):
    '''
    Local (over window) normalized cross correlation loss.

    Parameters:
        windows_size (int): Side length of the square window to calculate the local NCC.
    '''

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def forward(self, y_pred, y_true):
        '''
        Input:
            y_pred, y_true (tensor): Tensor of implicit template and warped images.
            
            Assume sized [batch(1), channel, *image_shape]
            The number of image in the first dimension can be different, in which broadcasting will be used. 
            
        Returns:
            NCC: Average local normalized cross-correlation coefficient. 
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        Ii = y_pred.permute(1, 0, 2, 3, 4)
        Ji = y_true.permute(1, 0, 2, 3, 4)

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [2, 3], "volumes should be 2 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win # [9,9,9]

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win) # calculate the product of array elements
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5) 
        # 1e-5 = ϵ, added for numerical stability to avoid division by zero

        return -torch.mean(cc)


class Grad(nn.Module):
    """
    N-D gradient loss.
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


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


    