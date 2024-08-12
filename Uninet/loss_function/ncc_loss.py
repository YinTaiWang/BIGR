from __future__ import annotations

from .common_imports import *
from torch.nn.modules.loss import _Loss
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
   
##################### VoxelMorph #####################
# class NCC(nn.Module):
#     """
#     Local (over window) normalized cross correlation loss.
#     Copied from the source code of VoxelMorph
#     """

#     def __init__(self, win=None):
#         self.win = win
#         super().__init__()

#     def forward(self, y_true, y_pred):
        
#         # Ii = y_true.permute(1,0,2,3,4)
#         # Ji = y_pred.permute(1,0,2,3,4)
        
#         Ii = y_true
#         Ji = y_pred

#         # get dimension of volume
#         # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
#         ndims = len(list(Ii.size())) - 2
#         assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

#         # set window size
#         win = [9] * ndims if self.win is None else self.win

#         # compute filters
#         sum_filt = torch.ones([1, 1, *win]).to("cuda")

#         pad_no = math.floor(win[0] / 2)

#         if ndims == 1:
#             stride = (1)
#             padding = (pad_no)
#         elif ndims == 2:
#             stride = (1, 1)
#             padding = (pad_no, pad_no)
#         else:
#             stride = (1, 1, 1)
#             padding = (pad_no, pad_no, pad_no)

#         # get convolution function
#         conv_fn = getattr(F, 'conv%dd' % ndims)

#         # compute CC squares
#         I2 = Ii * Ii
#         J2 = Ji * Ji
#         IJ = Ii * Ji

#         I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
#         J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
#         I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
#         J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
#         IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

#         win_size = np.prod(win)
#         u_I = I_sum / win_size
#         u_J = J_sum / win_size

#         cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
#         I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
#         J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

#         cc = cross * cross / (I_var * J_var + 1e-5)

#         return -torch.mean(cc)

##################### groupRegNet origin #####################
class NCC(nn.Module):
    '''
    Calculate local normalized cross-correlation coefficient between two images. 
    Copied from:
    https://github.com/vincentme/GroupRegNet/blob/master/model/loss.py

    Parameters
    ----------
    dim : int
        Dimension of the input images. 
    windows_size : int
        Side length of the square window to calculate the local NCC. 
    '''
    def __init__(self, win = 11):
        super().__init__()
        self.dim = 3
        self.num_stab_const = 1e-5 # numerical stability constant
        
        self.windows_size = win
        
        self.pad = win//2
        self.window_volume = win**self.dim
        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d
    
    def forward(self, I, J):
        '''
        Parameters
        ----------
        I and J : (1, c, d, h, w)
            Torch tensor of same shape. The number of image in the channel (c) can be different, in which broadcasting will be used. 
        windows_size : int
            Side length of the square window to calculate the local NCC. 
            
        Returns
        -------
        NCC : scalar
            Average local normalized cross-correlation coefficient. 
        '''
        I = I.permute(1,0,2,3,4) # change to (c, 1, d, h, w)
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
