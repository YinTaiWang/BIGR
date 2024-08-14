# Joint segmentation and groupwise registration network.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Format](#dataset-format)
- [Preprocessing](#preprocess)
- [Training](#training)

## Introduction
The primary objective of this study is to develop an integrated model that performs joint segmentation and groupwise registration simultaneously to efficiently process DCE-MRI scans. We propose utilizing a SEDD architecture for multi-task learning that integrates these tasks at the architectural level, coupled with a novel loss function that facilitates joint learning during algorithm optimization. By implementing this unified model, we aim to streamline operational workflows while simultaneously enhancing the model efficiency and accuracy. To the best of our knowledge, this is the first attempt to integrate joint segmentation and groupwise registration into one single network framework.


## Dataset Format

### Supported File Formats

Currently, only SimpleITKIO: `.nii.gz` is supported.

### Dataset Folder Structure

The folder setup should be similar to `nnUNet`:

```
root/
├── raw_data
├── preprocessed
└── results
```

```
root/raw_data/
├── Task001_BrainTumour
├── Task002_Heart
├── Task003_Liver
├── ...
```

All files in the folder must be named using the format `Task_XXX_YYYY`. Here, `XXX` is the patient ID, and `YYYY` is a 4-digit modality or channel identifier. If there is only one modality or channel, use `_0000`.

Labels for segmentation are stored in the labelsTr folder and should follow the same naming format as the image files. The label file names include the _YYYY suffix to indicate the modality or channel used for creating the label. For example, if you create a label for patient 001 using modality 0001, the file should be named `Task_001_0001`. Similarly, for patient 002 using modality 0002, the file should be named `Task_002_0002`.

Currently, only training data setup is required. There is no need for `ImagesTs` and `labelTs` folders as there is no testing data preparation code at this stage.

```
root/raw_data/Task003_Liver/
├── imagesTr
│   ├── BLT_001_0000.nii.gz
│   ├── BLT_001_0001.nii.gz
│   ├── BLT_001_0002.nii.gz
│   ├── BLT_001_0003.nii.gz
│   ├── BLT_002_0000.nii.gz
│   ├── BLT_002_0001.nii.gz
│   ├── BLT_002_0002.nii.gz
│   ├── BLT_002_0003.nii.gz
│   ├── ...
└── labelsTr
    ├── BLT_001_0001.nii.gz # label was created using image "imagesTr/BLT_001_0001.nii.gz"
    ├── BLT_002_0003.nii.gz # label was created using image "imagesTr/BLT_002_0003.nii.gz"
    ├── ...
```

## Preprocess

The preprocessing steps are performed independently prior to training. To preprocess the dataset, run the following Python script:

```
python ./preprocess/RUN.py -root ROOT_DIRECTORY -t TASK_ID --bias_correction --rigid_transformation
```

- `-root`: The root directory where the dataset is stored.
- `-t`: The task ID to preprocess.
- `--bias_correction`: (Optional) Apply bias correction.
- `--rigid_transformation`: (Optional) Apply rigid transformation.

The script will generate `dataset.json` and `dataset_properties.pkl` in the `raw_data` folder. `plans.json` along with all preprocessed images and segmentations will be stored in the `preprocessed` folder.

## Training

To initiate training, you can choose between two Python scripts depending on your specific requirements:

- For single segmentation tasks using a 3D model, execute:

```
python ./SegNet.py -root ROOT_DIRECTORY -t TASK_ID -f FOLD
```
This script employs the DynUnet model and processes single-image inputs.

- For joint segmentation and groupwise registration using a 4D model, execute:
```
python ./UniNet.py -root ROOT_DIRECTORY -t TASK_ID -f FOLD
```
This script utilizes the DynUnet_uni model and handles multi-channel image inputs.

In both cases, ensure to replace `ROOT_DIRECTORY` with the path to your dataset, `TASK_ID` with the task identifier, and `FOLD` with the specific fold number for cross-validation. The results from either script will be stored in the `results` folder.