import os
import SimpleITK as sitk
from set_preprocess import *

RESAMPLING = True
BIAS_CORRECTION = False
RIGID_TRANSFORM = True

################################## MAIN ##################################
def main():
    print(f"RESAMPLING: {RESAMPLING}; BIAS_CORRECTION: {BIAS_CORRECTION}; RIGID_TRANSFORM: {RIGID_TRANSFORM}")
    
    script_dir = os.getcwd()
    if 'r098906' in script_dir:
        GPU_cluster = True
    else:
        GPU_cluster = False
        
    if GPU_cluster:
        RawIMG_DIR = "/data/scratch/r098906/BLT_radiomics/images_raw"
        RawSEG_DIR = "/data/scratch/r098906/BLT_radiomics/segs_new_raw"
        PreprocessedIMG_DIR = "/data/scratch/r098906/BLT_radiomics/images_preprocessed_nobc"
        PreprocessedSEG_DIR = "/data/scratch/r098906/BLT_radiomics/segs_new_preprocessed"
        
    else:
        RawIMG_DIR = "C:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/images_raw"
        RawSEG_DIR = "C:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/segs_new_raw"
        PreprocessedIMG_DIR = "C:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/images_preprocessed_nobc"
        PreprocessedSEG_DIR = "C:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/segs_new_preprocessed"
    
    print(f"The processed files would be save at:"
          f"\nImages: {PreprocessedIMG_DIR}"
          f"\nSegmentations: {PreprocessedSEG_DIR}")
    
    
    patient_IDs = dict()
    for file in os.listdir(RawSEG_DIR):
        if file.endswith(".nii.gz"):
            ID = file.split("_")[1]
            phase = int(file.split("_")[2].split(".")[0])
            if ID not in patient_IDs:
                patient_IDs[ID] = phase
    print(f"patient counts: {len(patient_IDs)}")
    
    #########################
    ##  Preprocess images  ##
    #########################
    print("Preprocessing image files:")
    for ID in patient_IDs:
        # Create a list to save the images from the same patient
        processed_imgs_list = list()
        for img_file in os.listdir(RawIMG_DIR):
            if ID in img_file:
                ref_metadata = get_reference_metadata(ID, RawIMG_DIR, RawSEG_DIR)
                img_dir = os.path.join(RawIMG_DIR, img_file)
                img = sitk.ReadImage(img_dir) # read with sitk
                if RESAMPLING:
                    print("Resampling...")
                    processed_img = resampling(img, ref_metadata, is_seg=False)
                if BIAS_CORRECTION:
                    print("Bias correction...")
                    processed_img = n4_bias_correction(processed_img)
                processed_imgs_list.append(processed_img)
        if RIGID_TRANSFORM:
            print("Rigid transforming...")
            processed_imgs_list = rigid_transformation(processed_imgs_list, patient_IDs[ID])
        print("Creating 4D image...")
        image_itk_4D = create_4Dimage_array(processed_imgs_list) # combine img with itk
        # there was some issue with setting metadata (e.g. spacing) using SimpleITK
        # and thus here use itk to convert array to itk_image for further process 
        
        ## Set output metadata
        image_itk_4D.SetSpacing(processed_imgs_list[0].GetSpacing() + (1,))
        image_itk_4D.SetOrigin(processed_imgs_list[0].GetOrigin() + (0,))
        image_itk_4D.SetDirection(set_image_orientation(processed_imgs_list[0], image_itk_4D))
        
        img4D_out_dir = os.path.join(PreprocessedIMG_DIR, f"BLT_{ID}.nii.gz")
        itk.imwrite(image_itk_4D, img4D_out_dir)
        print(f"BLT_{ID}.nii.gz is created!")
    
    ################################
    ##  Preprocess segmentations  ##
    ################################
    # print("Preprocessing segmentation files:")
    # for seg_file in os.listdir(RawSEG_DIR):
    #     if seg_file.endswith(".nii.gz"):
    #         ID = seg_file.split("_")[1]
    #         ref_metadata = get_reference_metadata(ID, RawIMG_DIR, RawSEG_DIR)
    #         seg_dir = os.path.join(RawSEG_DIR, seg_file)
    #         seg = sitk.ReadImage(seg_dir)
    #         preprocessed_seg = resampling(seg, ref_metadata, is_seg=True)
    #         sitk.WriteImage(preprocessed_seg, os.path.join(PreprocessedSEG_DIR, seg_file))
    
if __name__ == "__main__":
    main()