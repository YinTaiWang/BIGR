## Import libraries ##
import os
import csv
import statistics
from tqdm import tqdm
from collections import Counter
import argparse
import SimpleITK as sitk

def parse_args():
    """
        Parses inputs from the commandline.
        :return: inputs as a Namespace object
    """
    parser = argparse.ArgumentParser(description="Write a csv file for checking the shapes and spacing of the images")
    parser.add_argument("-i", "--image_files_path", help="a path of where the images. This arguments is required!",
                        required=True)
    parser.add_argument("-o", "--output_path", help="a path of where to save the file. This arguments is required!",
                        required=True)
    parser.add_argument("-n", "--output_filename", help="output .csv file name. If it's not specified, then used the default name.", 
                        required=False)

    return parser.parse_args()


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

def get_stat(alist, stat:str):
    stat_list = list()
    if stat == "max":
        for l in range(len(alist)):
            m = max(alist[l])
            stat_list.append(m)
    elif stat == "median":
        for l in range(len(alist)):
            m = statistics.median(alist[l])
            stat_list.append(m)
    elif stat == "min":
        for l in range(len(alist)):
            m = min(alist[l])
            stat_list.append(m)    
    return stat_list


def main():
    stat = input("Need statistic info (Y/N)? ")
    q_shape_stat = input("For shape (max/median/min): ")
    q_spacing_stat = input("For spacing (max/median/min): ")
    # Process arguments
    args = parse_args()
    data_path = args.image_files_path
    out_path = args.output_path
    out_filename = args.output_filename
    if out_filename is None:
        out_filename = "images_info.csv"
    elif ".csv" not in out_filename:
        out_filename = out_filename + ".csv"
    
    out_file = os.path.join(out_path, out_filename)
    print(f"Data: {data_path}")
    print(f"Output: {out_file}")
    
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
    with open(out_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
    
    
    ## Iterate over each patient and write the file ##
    all_spacing = [[], [], []]
    all_shape = [[], [], []]
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
            
            # get the shape and spacing in the list
            shape_list.append(shape)
            spacing_list.append(spacing)
            
            # store the info for furthur median calculating
            all_shape[0].append(shape[0])
            all_shape[1].append(shape[1])
            all_shape[2].append(shape[2])
            all_spacing[0].append(spacing[0])
            all_spacing[1].append(spacing[1])
            all_spacing[2].append(spacing[2])
            
        # check if all the image shapes are the same
        shape_list.append("1") if chkList(shape_list, with_title=True) else shape_list.append("0")
        spacing_list.append("1") if chkList(spacing_list) else spacing_list.append("0")
        # find the most common image shape
        most_common_shape = Counter(shape_list).most_common(1)[0][0]
        most_common_sapcing = Counter(spacing_list).most_common(1)[0][0]
        shape_list.append(most_common_shape)
        spacing_list.append(most_common_sapcing)
        # write the CSV file
        with open(out_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            info = shape_list + spacing_list
            writer.writerow(info)
    
    if stat == "Y" or stat == "y":
        shape_stat = get_stat(all_shape, stat=q_shape_stat)
        spacing_stat = get_stat(all_spacing, stat=q_spacing_stat)
        print(f"Max shape:\t{shape_stat}\nMedian spacing:\t{spacing_stat}")
    else:
        pass
    
if __name__ == "__main__":
    main()