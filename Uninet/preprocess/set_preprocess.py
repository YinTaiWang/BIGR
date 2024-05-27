import os
import glob
import numpy as np
import itk
import SimpleITK as sitk

##################
##  Resampling  ##
##################

def get_reference_metadata(ID, rawimg_dir, rawseg_dir):
    '''
    Given the patient ID. Obtain the metadata from the reference image.
    
    Args:
        ID: patient ID
        rawimg_dir: path to raw image
        rawseg_dir: path to raw segmentation

    Returns:
        A dictionary contains image metadata,
        including image shape, spacing, direction, origin, and pixel value.
    '''
    
    # dictionary to store metadata
    ref_metadata = dict()
    
    # assume only has one segmentation file for one of the phase
    # the file format is in XXX_ID_YYYY, where XXX is the experiment and YYYY represent the phase
    # first, with this pattern, we find the phase that has the segmentation file
    file_pattern = os.path.join(rawseg_dir, f"BLT_{ID}_000*.nii.gz")
    for file in glob.glob(file_pattern):
        phase = os.path.basename(file).split('_')[2].split(".")[0] # get the YYYY
        # with the phase, get the metadata from the corresponding image
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
    return ref_metadata


def resampling(image, metadata, is_seg=False):
    '''
    Implement the image resampling.
    
    Args:
        image: the targeted image
        metadata: a dictionary contains image shape, spacing, direction, origin, and pixel value.
        is_seg: Set True if the file is segmentation (mask)
    Returns:
        Resampled image.
    '''
    ref_spacing = (1.4, 1.4, 2.5)
    ref_shape = list(metadata["shape"])
    scales = [metadata["spacing"][i]/ref_spacing[i] for i in range(len(ref_spacing))]
    scaled_shape = [round(ref_shape[i]*scales[i]) for i in range(len(ref_shape))]
    scaled_shape = tuple(scaled_shape)
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(ref_spacing)
    resample.SetSize(scaled_shape)
    resample.SetOutputDirection(metadata["direction"])
    resample.SetOutputOrigin(metadata["origin"])
   
    resample.SetTransform(sitk.Transform())

    if is_seg:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)


def n4_bias_correction(image):
    '''
    Bias correction.
    
    Args:
        image: the targeted image
    Returns:
        Bias corrected image.
    '''
    # Convert the image to a floating point type
    image = sitk.Cast(image, sitk.sitkFloat32)
    # Setting up N4 Bias field correction filter
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    return corrector.Execute(image)

def rigid_transformation(images, f_phase):
    '''
    Rigid transformation.
    
    Args:
        images: a list of targeted images
        f_phase: int, the phase that is set to the fixed image
    Returns:
        A list of rigid transformed images.
    '''
    fixed_image = images[f_phase]
    for phase in range(len(images)):
        if phase != f_phase:
            moving_image = images[phase]
            elastixImageFilter = sitk.ElastixImageFilter()
            elastixImageFilter.SetFixedImage(fixed_image)
            elastixImageFilter.SetMovingImage(moving_image)
            elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
            registed_img = elastixImageFilter.Execute()
            images[phase] = registed_img

    return images

#######################
##  Create 4D image  ##
#######################

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

def create_4Dimage_array(images_list):
    '''
    Stack images in the channel dimention.

    Args:
        images_list: a list that contains read images by SimpleITK.
    Returns:
        A 4D image in itk format
    '''
    for i in range(len(images_list)):
        if images_list[0].GetSize() != images_list[i].GetSize():
            raise ValueError("The size of images in the list is not matched.")
        
    ref_shape = images_list[0].GetSize()
    x, y, z = ref_shape[2], ref_shape[1], ref_shape[0]
    combined_img = np.zeros([len(images_list), x, y, z], np.float32)

    for i in range(len(images_list)):
        img_array = sitk.GetArrayFromImage(images_list[i])
        combined_img[i] = img_array
    image_itk_4D = itk.image_view_from_array(combined_img)
    return image_itk_4D