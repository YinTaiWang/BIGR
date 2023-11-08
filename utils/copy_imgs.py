# import libraries
import os
import glob
import shutil
from tqdm import tqdm

def get_file_len(path):
    all_items = os.listdir(path)
    files_in_dir = [i for i in all_items if os.path.isfile(os.path.join(path, i))]
    return len(files_in_dir)

if __name__ == "__main__":
    # Get the paths
    root_dir = os.getcwd()
    segs_dir = os.path.join(root_dir, "data/Segmentation_data/raw/Task403_BLT_arterial/imagesTr/labels/final/")
    raw_img_dir = os.path.join(root_dir, "data/Segmentation_data/raw/Task402_BLT_MRI/imagesTr/")
    dest_img_dir = os.path.join(root_dir, "data/elastix-test-data")
    dest_seg_dir = os.path.join(root_dir, "data/elastix-test-data/mask/")
    
    ## First, copy all the done segmentations to another segmentation folder ##
    print("Copy segmentation files.")
    for segmentations in tqdm(os.listdir(segs_dir)):
        seg_path = os.path.join(segs_dir, segmentations)
        shutil.copy(seg_path, dest_seg_dir)

    ## Second, copy all the corresponding images from the patient to image folder ##
    # check what are the patient's ID
    print("Copy all phases of image files.")
    for file in tqdm(os.listdir(segs_dir)):
        if file.endswith(".nii.gz"):
            # assume file names are Task_xxx_xxxx
            ID = file.split("_")[1]
            # use glob to list all the images and copy it to the folder
            img_filename = f"BLT_{ID}_*.nii.gz"
            img_path = glob.glob(os.path.join(raw_img_dir, img_filename))
            for file in img_path:
                shutil.copy(file, dest_img_dir)

    # check the length of data is correct
    seg_count = get_file_len(dest_seg_dir)
    img_count =  get_file_len(dest_img_dir)
    print(f"Number of segmentation files: {seg_count}")
    print(f"Number of image files: {img_count}")
    print(f"The number of files are correct: {seg_count*4==img_count}")