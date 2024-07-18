## VOXEL MORPH ST
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import logging
    
# class SpatialTransformer(nn.Module):
#     """
#     N-D Spatial Transformer
#     """

#     def __init__(self, size, mode='bilinear'):
#         super().__init__()

#         self.mode = mode

#         # create sampling grid
#         vectors = [torch.arange(0, s) for s in size]
#         grids = torch.meshgrid(vectors)
#         grid = torch.stack(grids)
#         grid = torch.unsqueeze(grid, 0)
#         grid = grid.type(torch.FloatTensor)

#         # registering the grid as a buffer cleanly moves it to the GPU, but it also
#         # adds it to the state dict. this is annoying since everything in the state dict
#         # is included when saving weights to disk, so the model files are way bigger
#         # than they need to be. so far, there does not appear to be an elegant solution.
#         # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
#         self.register_buffer('grid', grid)

#     def forward(self, src, flow):
        
#         # src: b,n,x,y,z
#         src = src.permute(1, 0, 2, 3, 4)  #src: n,b,x,y,z
        
#         # new locations
#         new_locs = self.grid + flow
#         shape = flow.shape[2:]

#         # need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

#         # move channels dim to last position
#         # also not sure why, but the channels need to be reversed
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1, 0]]
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             new_locs = new_locs[..., [2, 1, 0]]
            
#         result = nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
#         result = result.permute(1, 0, 2, 3, 4)
#         return result
    

## GROUPREGNET ST
class SpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image
    
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # self.grid_dict = {}
        # self.norm_coeff_dict = {}

    def forward(self, input_image, flow):   
        '''
        input_image: (b, n, *image_shape)
        flow: (n, dim, *image_shape)
        
        return: 
            warped moving image, (b, n, *image_shape)
        '''
        input_image = input_image.permute(1, 0, 2, 3, 4) # (n, b, *image_shape)
        img_shape = input_image.shape[2:]
        # if img_shape in self.grid_dict:
        #     grid = self.grid_dict[img_shape]
        #     norm_coeff = self.norm_coeff_dict[img_shape]
        # else:
        grids = torch.meshgrid([torch.arange(0, s) for s in img_shape])
        # the data in second dimension is in the order of [w, h, d]
        grid = torch.stack(grids[::-1], dim=0) # (dim, *image_shape)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.to(dtype = flow.dtype, 
                        device = flow.device)
        # the coefficients to map image coordinates to [-1, 1]
        norm_coeff = 2. / (torch.tensor(img_shape[::-1],
                                        dtype = flow.dtype, 
                                        device = flow.device) - 1.) 
            # self.grid_dict[img_shape] = grid
            # self.norm_coeff_dict[img_shape] = norm_coeff
            # logging.info(f'\nAdd grid shape {tuple(img_shape)}')
        new_grid = grid + flow 
        if self.dim == 2:
            new_grid = new_grid.permute(0, 2, 3, 1) # (n, *image_shape, 2)
        elif self.dim == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1) # (n, *image_shape, 3)
            
        if len(input_image) != len(new_grid):
            # make the image shape compatable by broadcasting
            input_image += torch.zeros_like(new_grid)
            new_grid += torch.zeros_like(input_image)
            
        warped_input_img =  F.grid_sample(input_image, new_grid*norm_coeff - 1.,
                                          mode = 'bilinear',
                                          align_corners = True,
                                          padding_mode = 'border')
        warped_input_img = warped_input_img.permute(1, 0, 2, 3, 4)
        return warped_input_img