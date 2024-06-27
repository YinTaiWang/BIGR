import os
import sys
sys.path.append(os.path.abspath('../'))
import warnings
from tqdm import tqdm
import SimpleITK as sitk
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from set_preprocess import *
from utils.file_and_folder_operations import *
from dataset_fingerprint import generate_dataset_json, generate_dataset_properties, generate_plans
from utils.tasks_by_id import set_preprocess_path_by_id

################################## MAIN ##################################
def preprocess(args):

    root_path = args.root_path
    task_id = str(args.task_id)
    BIAS_CORRECTION = args.bias_correction
    RIGID_TRANSFORMATION = args.rigid_transformation
    
    print("== SETUP ==")
    print(f"BIAS_CORRECTION: {BIAS_CORRECTION}; "
          f"RIGID_TRANSFORM: {RIGID_TRANSFORMATION}")
    
    (task_name, 
     RawTask_dir, PreprocessTask_dir, 
     RawIMG_dir, RawSEG_dir, 
     PreprocessIMG_dir, PreprocessSEG_dir) = set_preprocess_path_by_id(root_path, task_id)
    
    print(f"\nThe processed files would be save at:"
          f"\nImages: {PreprocessIMG_dir}"
          f"\nSegmentations: {PreprocessSEG_dir}")
    
    generate_dataset_json(output_path = os.path.join(RawTask_dir, "dataset.json"), 
                          imagesTr_dir = RawIMG_dir,
                          labelsTr_dir = RawSEG_dir, 
                          dataset_name = task_name)
    generate_dataset_properties(output_path = os.path.join(RawTask_dir, "dataset_properties.pkl"),
                                task_path = RawTask_dir)
    
    #############################  Plans  #############################
    dataset_json = load_json(os.path.join(RawTask_dir, "dataset.json"))
    dataset_properties = load_pickle(os.path.join(RawTask_dir, "dataset_properties.pkl"))
    generate_plans(output_path = os.path.join(PreprocessTask_dir, "plans.json"),
                   dataset_json = dataset_json,
                   dataset_properties = dataset_properties,
                   bias_correction = BIAS_CORRECTION,
                   rigid_transformation = RIGID_TRANSFORMATION)
    
    ###########################  Preprocess  ###########################
    plans = load_json(os.path.join(PreprocessTask_dir, "plans.json"))
    target_spacing = plans['preprocess']['Target_spacing']
    
    for training_data in tqdm(dataset_json['Training']):
        processed_imgs_list = []
        
        image_list = training_data['image']
        seg_file = training_data['label'] # assume only has one
        reference_metadata = get_reference_metadata(training_data, RawIMG_dir)
        
        for img_file in image_list:
            img_dir = os.path.join(RawIMG_dir, img_file)
            img = sitk.ReadImage(img_dir)
            # target_spacing = [1.5, 1.5, 2.5] # testing
            processed_img = resampling(img, target_spacing, reference_metadata, is_seg=False)
            
            if BIAS_CORRECTION:
                processed_img = n4_bias_correction(processed_img)
                
            processed_imgs_list.append(processed_img)
        
        # Segmentation only needs resampling
        # We assume only has one segmentation!
        seg_dir = os.path.join(RawSEG_dir, seg_file)
        seg = sitk.ReadImage(seg_dir)
        processed_seg = resampling(seg, target_spacing, reference_metadata, is_seg=True)
        sitk.WriteImage(processed_seg, os.path.join(PreprocessSEG_dir, seg_file))
        
        if len(processed_imgs_list) < 1:
            warnings.warn("Single image found, skipping 4D processing and rigid transformation.")
            img3D_out_dir = os.path.join(PreprocessIMG_dir, f"{training_data['id']}.nii.gz")
            sitk.WriteImage(processed_imgs_list[0], img3D_out_dir)
        else:
            if RIGID_TRANSFORMATION:
                processed_imgs_list = rigid_transformation(processed_imgs_list, training_data['gt_phase'])
            image_itk_4D = create_4Dimage_array(processed_imgs_list)
            
            ## Set output metadata
            image_itk_4D.SetSpacing(processed_imgs_list[0].GetSpacing() + (1,))
            image_itk_4D.SetOrigin(processed_imgs_list[0].GetOrigin() + (0,))
            image_itk_4D.SetDirection(set_image_orientation(processed_imgs_list[0], image_itk_4D))
            
            img4D_out_dir = os.path.join(PreprocessIMG_dir, f"{training_data['id']}.nii.gz")
            itk.imwrite(image_itk_4D, img4D_out_dir)
        
        
        if RIGID_TRANSFORMATION:
            if len(processed_imgs_list) > 1:
                processed_imgs_list = rigid_transformation(processed_imgs_list, training_data['gt_phase'])
                image_itk_4D = create_4Dimage_array(processed_imgs_list)
                
                ## Set output metadata
                image_itk_4D.SetSpacing(processed_imgs_list[0].GetSpacing() + (1,))
                image_itk_4D.SetOrigin(processed_imgs_list[0].GetOrigin() + (0,))
                image_itk_4D.SetDirection(set_image_orientation(processed_imgs_list[0], image_itk_4D))
                
                img4D_out_dir = os.path.join(PreprocessIMG_dir, f"{training_data['id']}.nii.gz")
                itk.imwrite(image_itk_4D, img4D_out_dir)
            else:
                warnings.warn("Single image found, skipping 4D processing and rigid transformation.")
                
                img3D_out_dir = os.path.join(PreprocessIMG_dir, f"{training_data['id']}.nii.gz")
                sitk.WriteImage(processed_imgs_list[0], img3D_out_dir)
        else:
            if len(processed_imgs_list) > 1:
                warnings.warn("Make sure you really don't want to perform rigid transformation...")
                image_itk_4D = create_4Dimage_array(processed_imgs_list)
                
                ## Set output metadata
                image_itk_4D.SetSpacing(processed_imgs_list[0].GetSpacing() + (1,))
                image_itk_4D.SetOrigin(processed_imgs_list[0].GetOrigin() + (0,))
                image_itk_4D.SetDirection(set_image_orientation(processed_imgs_list[0], image_itk_4D))
                
                img4D_out_dir = os.path.join(PreprocessIMG_dir, f"{training_data['id']}.nii.gz")
                itk.imwrite(image_itk_4D, img4D_out_dir)
            else:
                img3D_out_dir = os.path.join(PreprocessIMG_dir, f"{training_data['id']}.nii.gz")
                sitk.WriteImage(processed_imgs_list[0], img3D_out_dir)
    
    generate_dataset_properties(output_path = os.path.join(PreprocessTask_dir, "new_dataset_properties.pkl"),
                                task_path = PreprocessTask_dir)    
    
if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-root", "--root_path", type=str, help="path to root folder, where contains raw_data, preprocessed, and results folders")
    parser.add_argument("-t", "--task_id", type=int, help="task ID")
    parser.add_argument("--bias_correction", action='store_true', help="Apply bias correction")
    parser.add_argument("--rigid_transformation", action='store_true', help="Apply rigid transformation")

    args = parser.parse_args()
    preprocess(args)