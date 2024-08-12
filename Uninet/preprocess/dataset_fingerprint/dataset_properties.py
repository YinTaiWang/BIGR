import os
import sys
sys.path.append(os.path.abspath('../'))
import SimpleITK as sitk
import numpy as np
from .calculation import *
from utils.file_and_folder_operations import save_pickle

def check_anisotropic(spacing, threshold=3):
    return np.max(spacing) / np.min(spacing) >= threshold

def generate_dataset_properties(output_path, task_path):
    data_properties = {}
    all_sizes = list()
    all_spacings = list()
    
    # Get all size and spacing info
    imagesTr_dir = os.path.join(task_path, "imagesTr")
    for file in os.listdir(imagesTr_dir):
        if file.endswith(".nii.gz"):
            img_dir = os.path.join(imagesTr_dir, file)
            img = sitk.ReadImage(img_dir)
            all_sizes.append(img.GetSize())
            all_spacings.append(img.GetSpacing())
    
    # Summary
    data_properties['All_size'] = all_sizes
    data_properties['All_spacing'] = all_spacings
    data_properties['Mean_size'], data_properties['Median_size'] = calculate_mean_and_median(all_sizes)
    data_properties['Mean_spacing'], data_properties['Median_spacing'] = calculate_mean_and_median(all_spacings)
    data_properties['Anisotropic'] = check_anisotropic(data_properties['Median_spacing'])
    
    save_pickle(data_properties, output_path)
