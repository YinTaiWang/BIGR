## Import libraries ##
import os
import csv
from tqdm import tqdm
from collections import Counter
import SimpleITK as sitk

## Functions ##
def chkList(info_list, with_title=False):
    '''
    check if all the image shapes are the same.
    
    Args:
        shape_list: a list includes image shapes from a series of data
    Returns:
        a boolean value of whether all the image shapes are the same
    '''
    if with_title:
        return len(set(info_list)) == 2
    else:
        return len(set(info_list)) == 1

def get_quantities(data_path):
    '''
    check how many image files for a patient.
    
    Args:
        data_path: data directory
    Returns:
        a dictionary contains the patient ID and corresponding number of image files
    '''
    patient_IDs = dict()
    for file in os.listdir(data_path):
        if file.endswith(".nii.gz"):
            # assume the file format: Task_xxx_xxxx
            ID = file.split("_")[1]
            if ID not in patient_IDs:
                patient_IDs[ID] = 1
            else:
                patient_IDs[ID] += 1
    return patient_IDs

if __name__ == "__main__":
    ## Data path ##
    # Change if needed
    script_path = os.getcwd()
    data_path = os.path.join(script_path, "data", "Segmentation_data", "raw", "Task402_BLT_MRI", "imagesTr")
    out_filename = "./data/Images_info.csv"
    # data_path = "C:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/elastix-test-data/resample/"
    # out_filename = "C:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/elastix-test-data/resample/Images_shape.csv"
    print(f"Data path: {data_path}")

    ## Get a list of all the patient IDs and their amount of modalities ##
    patient_IDs = get_quantities(data_path)
    
    
    ## Initiate a CSV file with title row ##
    max_modalities = max(patient_IDs.values())
    columns = ["Patient"]
    for i in range(max_modalities):
        i += 1
        columns.append(f"shape_{i}")
    columns.append("shape equivalent")
    columns.append("Most common shape")
    for i in range(max_modalities):
        i += 1
        columns.append(f"spacing_{i}")
    columns.append("spacing equivalent")
    columns.append("Most common spacing")
    # write the CSV file
    with open(out_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
    
    
    ## Iterate over each patient and write the file ##
    for ID in tqdm(patient_IDs):
        ## check the shape ##
        shape_list = [f"BLT_{ID}"]
        spacing_list = list()
            
        n = patient_IDs[ID]
        # obtain image info
        for i in range(n):
            img_path = os.path.join(data_path, f"BLT_{ID}_000{i}.nii.gz")
            img = sitk.ReadImage(img_path)
            shape = img.GetSize()
            spacing = img.GetSpacing()
            
            shape_list.append(shape)
            spacing_list.append(spacing)
        # check if all the image shapes are the same
        shape_list.append("1") if chkList(shape_list, with_title=True) else shape_list.append("0")
        spacing_list.append("1") if chkList(spacing_list) else spacing_list.append("0")
        # find the most common image shape
        most_common_shape = Counter(shape_list).most_common(1)[0][0]
        most_common_sapcing = Counter(spacing_list).most_common(1)[0][0]
        shape_list.append(most_common_shape)
        spacing_list.append(most_common_sapcing)
        # write the CSV file
        with open(out_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            info = shape_list + spacing_list
            writer.writerow(info)
            
