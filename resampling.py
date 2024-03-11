import os
import glob
from tqdm import tqdm
from collections import Counter
import SimpleITK as sitk

####################
## Store the data for resampling
def get_ref_metadata(patient_IDs:dict, ID, rawimg_dir, rawseg_dir, one:bool, is_seg=False):
    '''
    Obtain the metadata from a reference image.
    
    Args:
        patient_IDs: a dictionary of patient IDs with number of phases
        ID: patient ID
        rawimg_dir: a path to raw images
        rawseg_dir: a path to raw segmentations
        one: set True if only has one segmentation among the phases
        is_seg: set True if the file is segmentation (mask)

    Returns:
        A dictionary contains image shape, spacing, direction, origin, and pixel value.
    '''
    
    # a dictionary to store metadata
    ref_metadata = dict()
    
    ### For Segmentations ###
    if is_seg:
        if one:
            # Get the metadata when only have one segmentation
            file_pattern = os.path.join(rawseg_dir, f"BLT_{ID}_000*.nii.gz")
            for file_path in glob.glob(file_pattern):
                ref_dir = file_path
                ref_seg = sitk.ReadImage(ref_dir)
                ref_metadata["shape"] = ref_seg.GetSize()
                ref_metadata["spacing"] = ref_seg.GetSpacing()
                ref_metadata["direction"] = ref_seg.GetDirection()
                ref_metadata["origin"] = ref_seg.GetOrigin()
                ref_metadata["pixelvalue"] = ref_seg.GetPixelIDValue()
        else:
            seg_metadata = {
                "shape": [],
                "spacing": [],
                "direction": [],
                "origin": [],
                "pixelvalue": []}
            
            for i in range(patient_IDs[ID]):
                ref_dir = os.path.join(rawseg_dir, f"BLT_{ID}_000{i}.nii.gz")
                ref_seg = sitk.ReadImage(ref_dir)
                seg_metadata["shape"].append(ref_seg.GetSize())
                seg_metadata["spacing"].append(ref_seg.GetSpacing())
                seg_metadata["direction"].append(ref_seg.GetDirection())
                seg_metadata["origin"].append(ref_seg.GetOrigin())
                seg_metadata["pixelvalue"].append(ref_seg.GetPixelIDValue())
            ref_metadata["shape"] = Counter(seg_metadata["shape"]).most_common(1)[0][0]
            ref_metadata["spacing"] = Counter(seg_metadata["spacing"]).most_common(1)[0][0]
            ref_metadata["direction"] = Counter(seg_metadata["direction"]).most_common(1)[0][0]
            ref_metadata["origin"] = Counter(seg_metadata["origin"]).most_common(1)[0][0]
            ref_metadata["pixelvalue"] = Counter(seg_metadata["pixelvalue"]).most_common(1)[0][0]
    
    ### For Images ###
    else:
        if one:
            # All images aligned to the phase that has segmentation
            file_pattern = os.path.join(rawseg_dir, f"BLT_{ID}_000*.nii.gz")
            for seg_file_path in glob.glob(file_pattern):
                phase = os.path.basename(seg_file_path).split('_')[2].split(".")[0]
                ref_dir = os.path.join(rawimg_dir, f"BLT_{ID}_{phase}.nii.gz")
                if os.path.exists(ref_dir):
                    ref_img = sitk.ReadImage(ref_dir)
                    ref_metadata["shape"] = ref_img.GetSize()
                    ref_metadata["spacing"] = ref_img.GetSpacing()
                    ref_metadata["direction"] = ref_img.GetDirection()
                    ref_metadata["origin"] = ref_img.GetOrigin()
                    ref_metadata["pixelvalue"] = ref_img.GetPixelIDValue()
                else:
                    raise ValueError("The image file is not exist.")
        else:
            metadata = {
                "shape": [],
                "spacing": [],
                "direction": [],
                "origin": [],
                "pixelvalue": []}
            
            for i in range(patient_IDs[ID]):
                ref_dir = os.path.join(rawimg_dir, f"BLT_{ID}_000{i}.nii.gz")
                ref_img = sitk.ReadImage(ref_dir)
                metadata["shape"].append(ref_img.GetSize())
                metadata["spacing"].append(ref_img.GetSpacing())
                metadata["direction"].append(ref_img.GetDirection())
                metadata["origin"].append(ref_img.GetOrigin())
                metadata["pixelvalue"].append(ref_img.GetPixelIDValue())
            ref_metadata["shape"] = Counter(metadata["shape"]).most_common(1)[0][0]
            ref_metadata["spacing"] = Counter(metadata["spacing"]).most_common(1)[0][0]
            ref_metadata["direction"] = Counter(metadata["direction"]).most_common(1)[0][0]
            ref_metadata["origin"] = Counter(metadata["origin"]).most_common(1)[0][0]
            ref_metadata["pixelvalue"] = Counter(metadata["pixelvalue"]).most_common(1)[0][0]
    return ref_metadata


def resampling(image, ref_metadata, is_seg=False):
    '''
    Implement the image resampling.
    
    Args:
        image: the targeted image
        ref_metadata: a dictionary contains image shape, spacing, direction, origin, and pixel value.
        is_seg: Set True if the file is segmentation (mask)
    Returns:
        Resampled image.
    '''
    ref_spacing = (1.4, 1.4, 2.5)
    ref_shape = list(ref_metadata["shape"])
    scales = [ref_metadata["spacing"][i]/ref_spacing[i] for i in range(len(ref_spacing))]
    scaled_shape = [round(ref_shape[i]*scales[i]) for i in range(len(ref_shape))]
    scaled_shape = tuple(scaled_shape)
    
    resample = sitk.ResampleImageFilter()
    
    resample.SetOutputSpacing(ref_spacing)
    resample.SetSize(scaled_shape)
    resample.SetOutputDirection(ref_metadata["direction"])
    resample.SetOutputOrigin(ref_metadata["origin"])
   
    resample.SetTransform(sitk.Transform())

    if is_seg:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)

def main():
    
    ####################
    ## Setting, change if needed
    resample_img_bool = True
    resample_seg_bool = True
    one_seg_bool = False
    
    ####################
    ## Set path
    script_dir = os.getcwd()
    rawimg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "images_raw")
    resampling_img_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "images_resampled_FOROLD")
    # assuming new files only has one segmentation file -> one_seg == True
    if one_seg_bool:
        rawseg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_new_raw")
        resampling_seg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_new_resampled")
    else:
        rawseg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_old_raw")
        resampling_seg_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "segs_old_resampled")

    ####################
    ## Get a dictionary of all the patient IDs and number of phases
    patient_IDs = dict()
    for file in os.listdir(rawimg_dir):
        if file.endswith(".nii.gz"):
            ID = file.split("_")[1]
            if ID not in patient_IDs:
                patient_IDs[ID] = 1
            else:
                patient_IDs[ID] += 1
    print(f"patient counts: {len(patient_IDs)}")
    print(f"file counts: {sum(patient_IDs.values())}")
    
    ####################
    ## Execution
    if resample_img_bool:
        print("Resampling image files:")
        for img_file in tqdm(os.listdir(rawimg_dir)):
            # if "017" in img_file:
            if img_file.endswith(".nii.gz"):
                ID = img_file.split("_")[1]
                ref_metadata = get_ref_metadata(patient_IDs, ID, rawimg_dir, rawseg_dir, one_seg_bool, is_seg=False)
                img_dir = os.path.join(rawimg_dir, img_file)
                img = sitk.ReadImage(img_dir)
                resampled_img = resampling(img, ref_metadata, is_seg=False)
                sitk.WriteImage(resampled_img, os.path.join(resampling_img_dir, img_file))
    
    if resample_seg_bool:    
        print("Resampling segmentation files:")
        for seg_file in tqdm(os.listdir(rawseg_dir)):
            # if "008" in seg_file:
            if seg_file.endswith(".nii.gz"):
                ID = seg_file.split("_")[1]
                ref_metadata = get_ref_metadata(patient_IDs, ID, rawimg_dir, rawseg_dir, one_seg_bool, is_seg=True)
                seg_dir = os.path.join(rawseg_dir, seg_file)
                seg = sitk.ReadImage(seg_dir)
                resampled_seg = resampling(seg, ref_metadata, is_seg=True)
                sitk.WriteImage(resampled_seg, os.path.join(resampling_seg_dir, seg_file))

    
if __name__ == "__main__":
    main()