import numpy as np
import torch
from typing import Union, List
import SimpleITK as sitk
from monai.transforms import Transform

class N4BiasFieldCorrectiond(Transform):
    """
    Applies N4 bias field correction to each 3D volume in a 4D image (3D + time).
    Assumes input is a dictionary with keys containing the 4D MRI volume(s).
    """
    def __init__(
        self,
        keys: Union[str, List[str]],
    ) -> None:
        super().__init__()  # Corrected this line
        self.keys = [keys] if isinstance(keys, str) else keys  # Ensure self.keys is always a list

    def __call__(self, data):
        for key in self.keys:  # Iterate over each key if there are multiple
            # Extract the 4D image tensor from the input dictionary
            mri_volume = data[key]
            # Ensure the input is a PyTorch tensor and convert it to a numpy array
            if isinstance(mri_volume, torch.Tensor):
                mri_volume_numpy = mri_volume.numpy()
            else:
                raise TypeError("Image data is not a PyTorch tensor")

            # Initialize an empty list to hold the corrected 3D volumes
            corrected_volumes = []
            # Iterate over the time dimension (assumed to be the first dimension)
            for t in range(mri_volume_numpy.shape[0]):
                # Extract the 3D volume at the current time point
                volume_3d = mri_volume_numpy[t, :, :, :]

                # Convert the 3D numpy array to a SimpleITK Image
                volume_3d_itk = sitk.GetImageFromArray(volume_3d)

                # Apply N4 bias correction
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrected_volume_itk = corrector.Execute(volume_3d_itk)

                # Convert the corrected SimpleITK Image back to a numpy array
                corrected_volume_numpy = sitk.GetArrayFromImage(corrected_volume_itk)

                # Add the corrected 3D volume to the list
                corrected_volumes.append(corrected_volume_numpy)

            # Stack the corrected volumes along the time dimension to form a 4D volume
            corrected_image_4d = np.stack(corrected_volumes, axis=0)

            # Convert the corrected 4D numpy array back to a PyTorch tensor
            corrected_image_tensor = torch.tensor(corrected_image_4d, dtype=torch.float32)

            # Update the input dictionary with the corrected 4D image
            data[key] = corrected_image_tensor

        return data
