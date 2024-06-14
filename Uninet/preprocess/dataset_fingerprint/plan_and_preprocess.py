import os
import sys
sys.path.append(os.path.abspath('../'))
import numpy as np
import SimpleITK as sitk
from .calculation import *
from utils.file_and_folder_operations import save_json

def calculate_target_spacing(anisotropic, median_spacing, xyz_values):
    '''
    target spacing: If anisotropic, lowest resolution axis tenth percentile, other axes median.
    '''
    if anisotropic:
        # Identifying the lowest resolution axis based on maximum median value
        lowest_resolution_axis = median_spacing.index(max(median_spacing))  # Corrected variable usage from `xyz_values` to `median_xyz`

        # Calculate the 10th percentile for the lowest resolution axis
        if lowest_resolution_axis == 0:
            return (np.percentile(xyz_values[0], 10), median_spacing[1], median_spacing[2])
        elif lowest_resolution_axis == 1:
            return (median_spacing[0], np.percentile(xyz_values[1], 10), median_spacing[2])
        else:
            return (median_spacing[0], median_spacing[1], np.percentile(xyz_values[2], 10))
    else:
        return median_spacing
    
def get_kernels_strides(sizes, spacings):
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break

        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def get_supervision(strides):
        deep_supervision = len(strides) - 3
        weights = np.array([0.5**i for i in range(deep_supervision + 1)])
        weights = weights / np.sum(weights)
        return deep_supervision, weights

def generate_plans(output_path, data_properties):
    json_dict = {}
    
    # PREPROCESS
    preprocess = {}
    spacing_xyz_values = extract_xyz_values(data_properties['All_spacing'])
    target_spacing = calculate_target_spacing(anisotropic = data_properties['Anisotropic'],
                                            median_spacing = data_properties['Median_spacing'], 
                                            xyz_values = spacing_xyz_values)
    
    preprocess['Anisotropic'] = data_properties['Anisotropic']
    preprocess['Target_spacing'] = target_spacing
    
    # MODEL CONFIGS
    kernels, strides = get_kernels_strides(data_properties['Median_size'], preprocess['Target_spacing'])
    filters = [32, 64, 128, 256, 320, 320, 320, 320][:len(kernels)]
    deep_supervision, weights = get_supervision(strides)
    
    model_configs = {
        'max_epochs': 1000,
        'fold': 5,
        'batch_size': 1,
        'lr_initial': 0.01,
        'lr_scheduler': True,
        'accumulate_grad_batches': 4,
        # sanity check
        'check_train': True,
        'check_tr_interval': 1000,
        'check_val': True,
        'check_val_interval': 500
    }
    
    model_configs['filters'] = filters
    model_configs['kernels'] = kernels
    model_configs['strides'] = strides
    model_configs['deep_supervision'] = deep_supervision
    model_configs['deep_supervision_weights'] = weights
    
    json_dict['preprocess'] = preprocess
    json_dict['model_configs'] = model_configs
    
    save_json(json_dict, output_path)