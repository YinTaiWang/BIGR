from typing import List, Tuple, Optional, Union
from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.data import decollate_batch

from monai.transforms import (
    Compose, LoadImaged, ToTensord,
    EnsureChannelFirstd, CropForegroundd, NormalizeIntensityd, Orientationd,
    # augmentaion
    RandZoomd, RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandAdjustContrastd, RandFlipd,
    # post process
    AsDiscrete, KeepLargestConnectedComponent, FillHoles)

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

def post_transforms(ground_truth: bool = False, method: str = None):
    if ground_truth:
        transforms = [AsDiscrete(to_onehot=2)]
    
    else:
        transforms = [AsDiscrete(argmax=True, to_onehot=2)]
        if method == 'fillholes':
            transforms += [FillHoles(applied_labels=None, connectivity=2)]
        elif method == 'largestcomponent': 
            transforms += [KeepLargestConnectedComponent(applied_labels=None, connectivity=2)]
        elif method == 'fillholes_and_largestcomponent':
            transforms += [
                    FillHoles(applied_labels=None, connectivity=2),
                    KeepLargestConnectedComponent(applied_labels=None, connectivity=2),
                ]
    return Compose(transforms)

def apply_transforms(output, method=None):
    # Define transform based on method
    if method == 'fillholes':
        transform = post_transforms(method='fillholes')
    elif method == 'largestcomponent':
        transform = post_transforms(method='largestcomponent')
    elif method == 'fillholes_and_largestcomponent':
        transform = post_transforms(method='fillholes_and_largestcomponent')
    else:
        transform = post_transforms()

    # Apply transform on each slice of the output
    if output.shape[1] > 2:
        transformed_outputs = [
            transform(image)
            for x in range(0, output.shape[1], 2)
            for image in decollate_batch(output[:, x:x+2, :, :, :])
        ]
    else:
        transformed_outputs = [transform(image) for image in decollate_batch(output)]
    return transformed_outputs

def best_post_processing_finder(output, seg):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    methods = [None, 'fillholes', 'largestcomponent', 'fillholes_and_largestcomponent']
    metrics = []
    transformed_outputs = []

    # Prepare segmentation labels for comparison
    post_label = post_transforms(ground_truth=True)
    post_seg = [post_label(image) for image in decollate_batch(seg)]
    if output.shape[1] > 2:
        post_seg = post_seg * (output.shape[1]/2)
        
    for method in methods:
        transformed_output = apply_transforms(output, method=method)
        dice_metric(y_pred=transformed_output, y=post_seg)
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        metrics.append(metric)
        transformed_outputs.append(transformed_output)
    
    max_metric = max(metrics)
    max_index = metrics.index(max_metric)

    return max_metric, methods[max_index], transformed_outputs[max_index]
    