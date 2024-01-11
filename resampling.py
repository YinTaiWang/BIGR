import os
import glob
import argparse
from tqdm import tqdm
from collections import Counter
import SimpleITK as sitk


def get_ref_info(patient_IDs:dict, ID, rawimg_dir, rawseg_dir, one:bool, is_seg=False):
    '''
    Obtain the metadata of reference image.
    
    Args:
        patient_IDs: a dictionary of patient IDs with number of phases
        ID: patient ID
        rawimg_dir: a path to raw images
        rawseg_dir: a path to raw segmentations
        is_seg: set True if the file is segmentation (mask)
        one: set True if only has one segmentation among the phases
    Returns:
        A dictionary contains image shape, spacing, direction, origin, and pixel value.
    '''
    
    # a dictionary to store metadata
    ref_info = dict()
    
    if is_seg:
        if one:
            file_pattern = os.path.join(rawseg_dir, f"BLT_{ID}_000*.nii.gz")
            for file_path in glob.glob(file_pattern):
                ref_dir = file_path
                ref_seg = sitk.ReadImage(ref_dir)
                ref_info["shape"] = ref_seg.GetSize()
        else:
            shape_list_seg = list()
            for i in range(patient_IDs[ID]):
                ref_dir = os.path.join(rawseg_dir, f"BLT_{ID}_000{i}.nii.gz")
                ref_seg = sitk.ReadImage(ref_dir)
                shape_list_seg.append(ref_seg.GetSize())
            ref_seg_shape = Counter(shape_list_seg).most_common(1)[0][0]
            ref_info["shape"] = ref_seg_shape
    else:
        shape_list = list()
        for i in range(patient_IDs[ID]):
            ref_dir = os.path.join(rawimg_dir, f"BLT_{ID}_000{i}.nii.gz")
            ref_img = sitk.ReadImage(ref_dir)
            shape_list.append(ref_img.GetSize())
        ref_img_shape = Counter(shape_list).most_common(1)[0][0]
        ref_info["shape"] = ref_img_shape
        
    ref_img = sitk.ReadImage(ref_dir)
    
    # ref_info["shape"] = most_common_shape
    ref_info["spacing"] = ref_img.GetSpacing()
    ref_info["direction"] = ref_img.GetDirection()
    ref_info["origin"] = ref_img.GetOrigin()
    ref_info["pixelvalue"] = ref_img.GetPixelIDValue()
    return ref_info

def resampling(image, ref_info, is_seg=False):
    '''
    Implement the image resampling.
    
    Args:
        image: the targeted image
        ref_info: a dictionary contains image shape, spacing, direction, origin, and pixel value.
        is_seg: Set True if the file is segmentation (mask)
    Returns:
        Resampled image.
    '''
    ref_spacing = (1.4062749743461609, 1.4062749743461609, 2.499995708465576)
    ref_shape = list(ref_info["shape"])
    scales = [ref_info["spacing"][i]/ref_spacing[i] for i in range(len(ref_spacing))]
    scaled_shape = [round(ref_shape[i]*scales[i]) for i in range(len(ref_shape))]
    scaled_shape = tuple(scaled_shape)
    
    resample = sitk.ResampleImageFilter()
    
    resample.SetOutputSpacing(ref_spacing)
    resample.SetSize(scaled_shape)
    resample.SetOutputDirection(ref_info["direction"])
    resample.SetOutputOrigin(ref_info["origin"])
    resample.SetDefaultPixelValue(ref_info["pixelvalue"])
   
    resample.SetTransform(sitk.Transform())

    if is_seg:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)

def main():
    one_seg = input("Only has one segmentation among the phases (Y/N)? ").lower()
    resample_img = input("Resample the image files (Y/N)? ").lower()
    resample_seg = input("Resample the segmentation files files (Y/N)? ").lower()
    
    if not all(answer in ['y', 'n'] for answer in [one_seg, resample_img, resample_seg]):
        raise ValueError("Input must be 'Y' or 'N'")
    
    one_seg_bool = True if one_seg == 'y' else False
    resample_img_bool = True if resample_img == 'y' else False
    resample_seg_bool = True if resample_seg == 'y' else False
    
    script_dir = os.getcwd()
    rawimg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "images_raw")
    resampling_img_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "images_resampled")
    
    # assuming Y means it's the new files with the default path
    # assuning new files only has one segmentation file
    if one_seg_bool == True:
        rawseg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_new_raw")
        resampling_seg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_new_resampled")
    elif one_seg_bool == False:
        rawseg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_old_raw")
        resampling_seg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_old_resampled")

    # get a list of all the patient IDs and number of phases
    patient_IDs = dict()
    for file in os.listdir(rawimg_dir):
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
    
    if resample_img_bool:
        print("Resampling image files:")
        for img_file in tqdm(os.listdir(rawimg_dir)):
            if img_file.endswith(".nii.gz"):
                ID = img_file.split("_")[1]
                ref_info = get_ref_info(patient_IDs, ID, rawimg_dir, rawseg_dir, one_seg_bool, is_seg=False)
                img_dir = os.path.join(rawimg_dir, img_file)
                img = sitk.ReadImage(img_dir)
                resampled_img = resampling(img, ref_info, is_seg=False)
                sitk.WriteImage(resampled_img, os.path.join(resampling_img_dir, img_file))
    
    if resample_seg_bool:    
        print("Resampling segmentation files:")
        for seg_file in tqdm(os.listdir(rawseg_dir)):
            if seg_file.endswith(".nii.gz"):
                ID = seg_file.split("_")[1]
                ref_info = get_ref_info(patient_IDs, ID, rawimg_dir, rawseg_dir, one_seg_bool, is_seg=True)
                seg_dir = os.path.join(rawseg_dir, seg_file)
                seg = sitk.ReadImage(seg_dir)
                resampled_seg = resampling(seg, ref_info, is_seg=True)
                sitk.WriteImage(resampled_seg, os.path.join(resampling_seg_dir, seg_file))

    
if __name__ == "__main__":
    main()