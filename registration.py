import os
import glob
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
            print(f"shape: {combined_img.shape}")
    return combined_img

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
    print(f"patient counts: {len(patient_IDs)}")
    print(f"file counts: {sum(patient_IDs.values())}")

    
    ########################################
    ### Get a dictionary of all the patient IDs and their segmentation phase
    seg_file_num = dict()
    for file in os.listdir(resampling_seg_dir):
        patient = file.split(".")[0].split("_")[1]
        phase = int(file.split(".")[0].split("_")[2][3])
        seg_file_num[patient] = phase
    
    ########################################
    ### Registration
    for ID in patient_IDs:
        
        spacing_4D = (1.4062749743461609, 1.4062749743461609, 2.499995708465576, 1)
        print(f"Load image data: Patient_{ID}")
        
        ## Load image ##
        img_4D = create_4Dimage_array(patient_IDs, ID, resampling_img_dir, resampling_seg_dir,is_seg=False)
        image_itk_4D = itk.image_view_from_array(img_4D)
        
        # Setting groupwise parameter object
        print("Setting registration parameters.")
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile("./Par0047Groupwise.txt")
        
        print("Start registration")  
        result_image, result_transform_parameters = itk.elastix_registration_method(
            image_itk_4D,  image_itk_4D,
            parameter_object=parameter_object,
            log_to_console=True)
        result_image.SetSpacing(spacing_4D)
        
        ## Load segmentation ##
        mask_4D = create_4Dimage_array(patient_IDs, ID, resampling_img_dir, resampling_seg_dir,is_seg=True)
        mask_itk_4D = itk.image_view_from_array(mask_4D)
        
        print("Start transformix")
        result_transform_parameters.SetParameter("FinalBSplineInterpolationOrder", "0")
        result_mask_transformix = itk.transformix_filter(
            mask_itk_4D,
            result_transform_parameters)
        
        ref_phase = seg_file_num[ID]
        for i in range(result_mask_transformix.shape[0]):
            result_mask_transformix[i,:,:,:] = result_mask_transformix[ref_phase,:,:,:]
        result_mask_transformix = itk.image_view_from_array(result_mask_transformix)
        result_mask_transformix.SetSpacing(spacing_4D)
        
        print("Save results")
        # Set the path for saving 4D images
        img4D_out_dir = os.path.join(data_dir, "4D_new_registered", "images", f"BLT_{ID}.nii.gz")
        seg4D_out_dir = os.path.join(data_dir, "4D_new_registered", "segs", f"BLT_{ID}.nii.gz")
        itk.imwrite(result_image, img4D_out_dir)
        itk.imwrite(result_mask_transformix, seg4D_out_dir)
        
        param_out_dir = os.path.join(data_dir, "4D_new_registered", "parameters", f"BLT_{ID}.txt")
        parameter_object.WriteParameterFile(result_transform_parameters, param_out_dir)

if __name__ == "__main__":
    main()