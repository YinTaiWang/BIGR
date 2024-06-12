import os
import sys
sys.path.append(os.path.abspath('../'))
import numpy as np
import SimpleITK as sitk
from utils.file_and_folder_operations import save_pickle

def calculate_mean_and_median(list_of_size_or_spacing):
    x_values = [dim[0] for dim in list_of_size_or_spacing]
    y_values = [dim[1] for dim in list_of_size_or_spacing]
    z_values = [dim[2] for dim in list_of_size_or_spacing]

    # Calculate mean and median for each dimension
    mean_x = np.mean(x_values)
    median_x = np.median(x_values)

    mean_y = np.mean(y_values)
    median_y = np.median(y_values)

    mean_z = np.mean(z_values)
    median_z = np.median(z_values)
    return (mean_x, mean_y, mean_z), (median_x, median_y, median_z)

def generate_dataset_properties(task_path):
    dataset_properties = dict()
    all_sizes = list()
    all_spacings = list()
    
    imagesTr_dir = os.path.join(task_path, "imagesTr")
    for file in os.listdir(imagesTr_dir):
        if file.endswith(".nii.gz"):
            img_dir = os.path.join(imagesTr_dir, file)
            img = sitk.ReadImage(img_dir)
            all_sizes.append(img.GetSize())
            all_spacings.append(img.GetSpacing())
    dataset_properties['all_sizes'] = all_sizes
    dataset_properties['all_spacing'] = all_spacings
    dataset_properties['mean_size'], dataset_properties['median_size'] = calculate_mean_and_median(all_sizes)
    dataset_properties['mean_spacing'], dataset_properties['median_spacing'] = calculate_mean_and_median(all_spacings)
    
    save_pickle(dataset_properties, os.path.join(task_path, "dataset_properties.pkl"))
