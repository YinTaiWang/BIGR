import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# def create_history_dict():
#     history = {'total_loss': [],
#                 'loss_dice': [],
#                 'loss_consis': [],
#                 'loss_simi': [],
#                 'loss_smooth': [],
#                 'metric_dice': [],
#                 'metric_ncc': []}
#     return history

def scale(image, seg, scale):
    if scale < 1:
        scaled_image = F.interpolate(image, scale_factor=scale, mode='trilinear', align_corners=True, recompute_scale_factor=False)
        scaled_seg = F.interpolate(seg.to(torch.float32), scale_factor=scale, mode='nearest', recompute_scale_factor=False)
    else:
        scaled_image, scaled_seg = image, seg
    return scaled_image, scaled_seg

def scale_to_origin(scaled_image, scaled_seg, scaled_prediction, original_shape):
    image = F.interpolate(scaled_image, size=original_shape, mode='trilinear', align_corners=True, recompute_scale_factor=False)
    seg = F.interpolate(scaled_seg, size=original_shape, mode='nearest', recompute_scale_factor=False)
    prediction = F.interpolate(scaled_prediction, size=original_shape, mode='nearest', recompute_scale_factor=False)
    return image, seg, prediction

def calculate_reduction_factors(strides):
    # Initialize reduction factors
    depth_reduction = 1
    height_reduction = 1
    width_reduction = 1

    # Calculate the cumulative reduction factor for each dimension
    for stride in strides:
        depth_reduction *= stride[0]
        height_reduction *= stride[1]
        width_reduction *= stride[2]

    return [depth_reduction, height_reduction, width_reduction]

def add_padding_to_divisible(image, divisors):
    """
    Pads the image so that its dimensions are divisible by the specified divisor.
    
    Args:
        image (torch.Tensor): The input image tensor of shape (N, C, D, H, W)
        divisor (int): The number to which the dimensions should be divisible.
    
    Returns:
        torch.Tensor: The padded image tensor.
    """
    # Current size of the dimensions
    depth, height, width = image.shape[2], image.shape[3], image.shape[4]

    # Calculate the padding needed to make each dimension divisible by `divisor`
    pad_depth = (divisors[0] - depth % divisors[0]) % divisors[0]
    pad_height = (divisors[1] - height % divisors[1]) % divisors[1]
    pad_width = (divisors[2] - width % divisors[2]) % divisors[2]

    # Define padding for depth, height, and width
    # Padding order in F.pad for 5D tensor is (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    padding = (0, pad_width, 0, pad_height, 0, pad_depth)

    # Apply padding
    padded_image = F.pad(image, padding, "constant", 0)
    # print(f"padding: {image.shape[2:]} -> {padded_image.shape[2:]}")

    return padded_image

# def compute_losses(loss_functions, outputs_seg, ori_seg, phase, warped_input_image, template, disp_t2i):
#     loss_dice = 0
#     for c_phase, c in enumerate(range(0, outputs_seg.shape[1], 2)):
#         phase_loss = loss_functions['dice'](outputs_seg[:,c:c+2,...], ori_seg)
#         if c_phase != phase:
#             phase_loss = phase_loss * 0.8 # give less weight on the phases without ground truth
#         loss_dice += phase_loss
#     pair = int(outputs_seg.shape[1]/2)
#     loss_dice /= pair
            
#     loss_consis = loss_functions['consis'](outputs_seg)
#     loss_simi = loss_functions['ncc'](warped_input_image, template)
#     loss_smooth = loss_functions['grad'](disp_t2i)
#     ncc_metric = -loss_simi
    
#     total_loss = (loss_dice + loss_simi) * 0.8 + (loss_consis + loss_smooth) * 0.2
#     losses = [total_loss, loss_dice, loss_consis, loss_simi, loss_smooth, ncc_metric]
#     return total_loss, losses

def update_epoch_stats(stats, losses):
    keys = list(stats.keys())
    for i, key in enumerate(keys): # skip metric_dice
        stats[key] += losses[i].item()
    return stats
