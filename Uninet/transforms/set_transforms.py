from typing import List, Tuple, Optional, Union
from monai.utils import set_determinism

from monai.transforms import (
    Compose, EnsureChannelFirstd, CropForegroundd, LoadImaged,
    NormalizeIntensityd, Orientationd, AsDiscrete,
    # augmentaion
    RandRotated, RandZoomd, RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandAdjustContrastd, RandFlipd, ToTensord)

from Uninet.transforms.transforms import N4BiasFieldCorrectiond


def training_transforms(seed: Optional[int] = None, validation: bool = False):
    if seed:
        set_determinism(seed=seed)

    transforms = [
        LoadImaged(keys=["image", "seg"]),
        EnsureChannelFirstd(keys=["image", "seg"]),
        NormalizeIntensityd(keys=["image"], channel_wise=True),
        CropForegroundd(keys=["image", "seg"], source_key="image"),
        Orientationd(keys=["image", "seg"], axcodes="RAS"),
        ToTensord(keys=["image", "seg"]),
    ]

    if not validation:
        ## augmentation
        transforms += [
            # RandRotated(
            #     keys=["image", "seg"],
            #     range_x=180,
            #     range_y=180,
            #     mode=("bilinear", "nearest"),
            #     align_corners=(True, None),
            #     prob=0.2,
            # ),
            RandZoomd(
                keys=["image", "seg"],
                min_zoom=0.7,
                max_zoom=1.4,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.2,
            ),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.2,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandAdjustContrastd(keys=["image"], gamma=(0.65, 1.5), prob=0.15),
            RandFlipd(
                keys=["image", "seg"], spatial_axis=[0], prob=0.5
            ),
            RandFlipd(
                keys=["image", "seg"], spatial_axis=[1], prob=0.5
            ),
            RandFlipd(
                keys=["image", "seg"], spatial_axis=[2], prob=0.5
            ),
        ]


    return Compose(transforms)

def post_transfroms(label: bool = False):
    if label:
        transforms = [AsDiscrete(to_onehot=2)]
    else:
        transforms = [AsDiscrete(argmax=True, to_onehot=2)]
    return Compose(transforms)
    