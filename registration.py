import os
import glob
import numpy as np
import itk
import SimpleITK as sitk
from tqdm import tqdm

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
    
    ########################################
    ### Set path
    script_dir = os.getcwd()
    data_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files")
    resampling_img_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "images_resampled")
    resampling_seg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_new_resampled")
    
    ########################################
    ### Get a dictionary of all the patient IDs and number of phases
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

    
    ########################################
    ### Get a dictionary of all the patient IDs and their segmentation phase
    seg_file_num = dict()
    for file in os.listdir(resampling_seg_dir):
        patient = file.split(".")[0].split("_")[1]
        phase = int(file.split(".")[0].split("_")[2][3])
        seg_file_num[patient] = phase
    
    ########################################
    ### Registration
    for ID in tqdm(patient_IDs):
    # for ID in ["006"]: # for test
        
        print(f"Load patient: Patient_{ID}")
        img_dir = os.path.join(resampling_img_dir, f"BLT_{ID}_0000.nii.gz")
        img = sitk.ReadImage(img_dir)
        
        ## Get metadata
        img_spacing = img.GetSpacing()
        img_origin = img.GetOrigin()
        
        ## Load image ##
        img_4D = create_4Dimage_array(patient_IDs, ID, resampling_img_dir, resampling_seg_dir, is_seg=False)
        image_itk_4D = itk.image_view_from_array(img_4D)
        
        # Setting groupwise parameter object
        print("\n\n")
        print("SETTING PARAMETERS")
        print("-"*30)
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile("./Par0047Groupwise.txt")
        
        print("\n\n")
        print("START REGISTRATION")
        print("-"*30)
        result_image, result_transform_parameters = itk.elastix_registration_method(
            image_itk_4D,  image_itk_4D,
            parameter_object=parameter_object,
            log_to_console=True)
        result_image.SetSpacing(img_spacing + (1,))
        result_image.SetOrigin(img_origin + (0,))
        
        # check orientation
        re_orientation = set_image_orientation(img, result_image)
        print(f"Original image orientation: \n{img.GetDirection()}")
        print(f"Result image orientation: \n{result_image.GetDirection()}")
        print(f"Re-oriented data: \n{re_orientation}")
        
        result_image.SetDirection(re_orientation)
        
        # Load segmentation ##
        mask_4D = create_4Dimage_array(patient_IDs, ID, resampling_img_dir, resampling_seg_dir, is_seg=True)
        mask_itk_4D = itk.image_view_from_array(mask_4D)
        
        print("\n\n")
        print("START TRANSFORMIX")
        print("-"*30)
        result_transform_parameters.SetParameter("FinalBSplineInterpolationOrder", "0")
        result_mask_transformix = itk.transformix_filter(
            mask_itk_4D,
            result_transform_parameters)
        
        # cover other channels with the ref_phase in segmentation
        ref_phase = seg_file_num[ID]
        seg_dir = os.path.join(resampling_seg_dir, f"BLT_{ID}_000{ref_phase}.nii.gz")
        seg = sitk.ReadImage(seg_dir)
        
        for i in range(result_mask_transformix.shape[0]):
            result_mask_transformix[i,:,:,:] = result_mask_transformix[ref_phase,:,:,:]
        result_mask_transformix = itk.image_view_from_array(result_mask_transformix)
        result_mask_transformix.SetSpacing(img_spacing + (1,))
        result_mask_transformix.SetOrigin(img_origin + (0,))
        
        # check orientation
        re_orientation_seg = set_image_orientation(seg, result_mask_transformix)
        print(f"Original seg orientation:\n {seg.GetDirection()}")
        print(f"Result seg orientation:\n {result_mask_transformix.GetDirection()}")
        print(f"Re-oriented data:\n {re_orientation_seg}")
        
        result_mask_transformix.SetDirection(re_orientation_seg)
        
        print("\nSAVE RESULTS!\n")
        # Set the path for saving 4D images
        img4D_out_dir = os.path.join(data_dir, "4D_new_registered", "images", f"BLT_{ID}.nii.gz")
        seg4D_out_dir = os.path.join(data_dir, "4D_new_registered", "segs", f"BLT_{ID}.nii.gz")
        itk.imwrite(result_image, img4D_out_dir)
        itk.imwrite(result_mask_transformix, seg4D_out_dir)
        
        param_out_dir = os.path.join(data_dir, "4D_new_registered", "parameters", f"BLT_{ID}.txt")
        parameter_object.WriteParameterFile(result_transform_parameters, param_out_dir)

if __name__ == "__main__":
    main()