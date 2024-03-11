import os
import sys
import glob
import logging
from tqdm import tqdm
from collections import Counter
import numpy as np

import itk
import SimpleITK as sitk

def create_4Dimage_array(patient_IDs, ID, resampling_img_dir, resampling_seg_dir,is_seg=False):
    '''
    Update the created image arrary with the loaded image data

    Args:
        ID: Patient ID
        is_seg: Set True if the file is segmentation (mask)
    Returns:
        A 4D numpy array contains 4 images or 4 duplicated segmentations.
    '''
    n_channel = patient_IDs[ID]
    if is_seg:
        file_pattern = os.path.join(resampling_seg_dir, f"BLT_{ID}_000*.nii.gz")
        for file_path in glob.glob(file_pattern):
            img_dir = file_path
            img = sitk.ReadImage(img_dir)
            ref_shape = img.GetSize()
            x, y, z = ref_shape[2], ref_shape[1], ref_shape[0]
            combined_img = np.zeros([n_channel, x, y, z], np.float32)
    else:
        img_dir = os.path.join(resampling_img_dir, f"BLT_{ID}_0000.nii.gz")
        img = sitk.ReadImage(img_dir)
        ref_shape = img.GetSize()
        x, y, z = ref_shape[2], ref_shape[1], ref_shape[0]
        combined_img = np.zeros([n_channel, x, y, z], np.float32)

    for i in range(n_channel):
        if is_seg:
            file_pattern = os.path.join(resampling_seg_dir, f"BLT_{ID}_000*.nii.gz")
            for file_path in glob.glob(file_pattern):
                img_dir = file_path
        else:
            img_dir = os.path.join(resampling_img_dir, f"BLT_{ID}_000{i}.nii.gz")
        
        print(f"Load image: {img_dir}")
        img = sitk.ReadImage(img_dir)
        img_array = sitk.GetArrayFromImage(img)
        combined_img[i] = img_array
        if i == patient_IDs[ID]-1:
            print(f"Combined image size: {combined_img.shape}")
    return combined_img

def set_image_orientation(image, image_4D):
    """Set the 4D image orientation based on the original image."""
    
    direction_img = image.GetDirection()
    direction_4Dimg = image_4D.GetDirection()
    direction_4Dimg_array = itk.array_from_matrix(image_4D.GetDirection())
    
    ndim_img = len(direction_img)
    
    assert ndim_img in [9], "input image should be 3D. found: %d" % ndim_img/3
    
    X_vector = direction_img[0:3]
    Y_vector = direction_img[3:6]
    Z_vextor = direction_img[6:9]
    
    direction_4Dimg_array[0][0:3] = X_vector
    direction_4Dimg_array[1][0:3] = Y_vector
    direction_4Dimg_array[2][0:3] = Z_vextor
    
    direction_4Dimg = itk.matrix_from_array(direction_4Dimg_array)
    
    return direction_4Dimg

def main():
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    create_4Dimg_bool = True
    create_4Dseg_bool = True
    
    ####################
    ## Set path
    script_dir = os.getcwd()
    resampling_img_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "images_resampled_oldseg")
    resampling_seg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_old_resampled")
    images_4D_registered_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "4D_old_NP", "images")
    segs_4D_registered_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "4D_old_NP", "segs")
    
    ####################
    ## Get data info
    patient_IDs = dict()
    for file in os.listdir(resampling_img_dir):
        if file.endswith(".nii.gz"):
            ID = file.split("_")[1]
            if ID not in patient_IDs:
                patient_IDs[ID] = 1
            else:
                patient_IDs[ID] += 1
        else:
            continue
    print("\n","-"*30)
    print(f"Patient counts: {len(patient_IDs)}")
    print(f"File counts: {sum(patient_IDs.values())}")
    
    
    ####################
    ## Combine images
    # Note that images are already resampled and modified to the same size within patient
    # Therefore here is only a siimple combining without 
    print("\n","-"*30)
    if create_4Dimg_bool:
        for ID in tqdm(patient_IDs):
        # for ID in ["004"]: # for testing
            
            print(f"Load patient: Patient_{ID}")
            img_dir = os.path.join(resampling_img_dir, f"BLT_{ID}_0000.nii.gz")
            img = sitk.ReadImage(img_dir)
            
            ## Get metadata
            img_spacing = img.GetSpacing()
            img_origin = img.GetOrigin()
            
            ## Load image ##
            img_4D = create_4Dimage_array(patient_IDs, ID, resampling_img_dir, resampling_img_dir, is_seg=False)
            image_itk_4D = itk.image_view_from_array(img_4D)
            
            ## Set output metadata
            image_itk_4D.SetSpacing(img_spacing + (1,))
            image_itk_4D.SetOrigin(img_origin + (0,))
            image_itk_4D.SetDirection(set_image_orientation(img, image_itk_4D))
            
            ## Write image
            img4D_out_dir = os.path.join(images_4D_registered_dir, f"BLT_{ID}.nii.gz")
            itk.imwrite(image_itk_4D, img4D_out_dir)
    if create_4Dseg_bool:
        for ID in tqdm(patient_IDs):
        # for ID in ["004"]: # for testing
            
            print(f"Load patient: Patient_{ID}")
            img_dir = os.path.join(resampling_seg_dir, f"BLT_{ID}_0000.nii.gz")
            img = sitk.ReadImage(img_dir)
            
            ## Get metadata
            img_spacing = img.GetSpacing()
            img_origin = img.GetOrigin()
            
            ## Load image ##
            img_4D = create_4Dimage_array(patient_IDs, ID, resampling_seg_dir, resampling_seg_dir, is_seg=True)
            image_itk_4D = itk.image_view_from_array(img_4D)
            
            ## Set output metadata
            image_itk_4D.SetSpacing(img_spacing + (1,))
            image_itk_4D.SetOrigin(img_origin + (0,))
            image_itk_4D.SetDirection(set_image_orientation(img, image_itk_4D))
            
            ## Write image
            img4D_out_dir = os.path.join(segs_4D_registered_dir, f"BLT_{ID}.nii.gz")
            itk.imwrite(image_itk_4D, img4D_out_dir)
        
if __name__ == "__main__":
    main()