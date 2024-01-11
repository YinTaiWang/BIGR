import os
import shutil

# copy the raw data to another folder with new names
script_dir = os.getcwd()
rawdata_dir = os.path.join(script_dir, "data", "raw", "Bigger_dataset")
new_rawdata_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files")

for folder in os.listdir(rawdata_dir):
    if folder.startswith('BLTRadiomics-'):
        folder_path = os.path.join(rawdata_dir, folder)
        
        # Loop through each file in the folder
        for file in os.listdir(folder_path):
            if file.startswith('image_phase_'):
                old_img_path = os.path.join(folder_path, file)
                
                id = folder_path.split("-")[1]
                phase = file.split("_")[2].split(".")[0]
                
                # Rename the file as needed
                new_img_name = f"BLT_{id}_000{phase}.nii.gz"
                new_img_path = os.path.join(new_rawdata_dir, "image", new_img_name)
                # Move the file
                shutil.copy(old_img_path, new_img_path)
            elif file.startswith('seg_phase_'):
                old_seg_path = os.path.join(folder_path, file)
                
                id = folder_path.split("-")[1]
                phase = file.split("_")[2].split(".")[0]
                
                # Rename the file as needed
                new_seg_name = f"BLT_{id}_000{phase}.nii.gz"
                new_seg_path = os.path.join(new_rawdata_dir, "mask", new_seg_name)
                # Move the file
                shutil.copy(old_seg_path, new_seg_path)