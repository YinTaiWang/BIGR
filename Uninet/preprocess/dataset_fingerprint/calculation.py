import numpy as np

def extract_xyz_values(sizes_or_spacings):
    x_values = [dim[0] for dim in sizes_or_spacings]
    y_values = [dim[1] for dim in sizes_or_spacings]
    z_values = [dim[2] for dim in sizes_or_spacings]
    return (x_values, y_values, z_values)

def calculate_mean_and_median(sizes_or_spacings):
    
    xyz_values = extract_xyz_values(sizes_or_spacings)
    
    # Calculate mean and median for each dimension
    mean_x = np.mean(xyz_values[0])
    median_x = np.median(xyz_values[0])

    mean_y = np.mean(xyz_values[1])
    median_y = np.median(xyz_values[1])

    mean_z = np.mean(xyz_values[2])
    median_z = np.median(xyz_values[2])
    return (mean_x, mean_y, mean_z), (median_x, median_y, median_z)