# Joint segmentation and groupwise registration network.

## Usage

### Dataset Format

#### Supported File Formats

Currently, only SimpleITKIO: `.nii.gz` is supported.

#### Dataset Folder Structure

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

Each file name should follow the format `Task_XXX_YYYY`, where `XXX` is the task ID and `YYYY` is the 4-digit modality/channel identifier. If only one modality/channel is available, it should still be named using the format `_0000`.

For each ID, it is assumed that there is only one label available. Therefore, the file name in `labelsTr` should use the same format, with `_YYYY` indicating the modality/channel from which the label was created. 

For example, if you create labels (segmentations) using specific modality/channel images for IDs `001` and `002`, then the label file names in the `labelsTr` folder should correspond to those specific modality/channel images.


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
├── imagesTs
│   ├── BLT_485_0000.nii.gz
│   ├── BLT_485_0001.nii.gz
│   ├── BLT_485_0002.nii.gz
│   ├── BLT_485_0003.nii.gz
│   ├── BLT_486_0000.nii.gz
│   ├── BLT_486_0001.nii.gz
│   ├── BLT_486_0002.nii.gz
│   ├── BLT_486_0003.nii.gz
│   ├── ...
└── labelsTr
    ├── BLT_001_0001.nii.gz # label was created using image "imagesTr/BLT_001_0001.nii.gz"
    ├── BLT_002_0003.nii.gz # label was created using image "imagesTr/BLT_002_0003.nii.gz"
    ├── ...
```

### Preprocess

To preprocess the dataset, run the following Python script:

```
python ./preprocess/RUN.py -root ROOT_DIRECTORY -t TASK_ID --bias_correction --rigid_transformation
```

- `-root`: The root directory where the dataset is stored.
- `-t`: The task ID to preprocess.
- `--bias_correction`: (Optional) Apply bias correction.
- `--rigid_transformation`: (Optional) Apply rigid transformation.

The script will generate `dataset.json` and `dataset_properties.pkl` in the `raw_data` folder. The preprocessed images and segmentations will be in the `preprocessed` folder.

### Training

To run the training, run the following Python script:

```
python ./SegNet.py -root ROOT_DIRECTORY -t TASK_ID -f FOLD # 3D model
python ./UniNet.py -root ROOT_DIRECTORY -t TASK_ID -f FOLD # 4D model
```

The results would be in the 'results' folder.