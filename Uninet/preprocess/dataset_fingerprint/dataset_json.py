import os
import sys
sys.path.append(os.path.abspath('../'))
from collections import defaultdict
from utils.file_and_folder_operations import save_json

def list_files(directory, pattern):
    """ List all files in the directory that match the pattern """
    return [f for f in os.listdir(directory) if f.endswith(pattern)]

def group_files_by_id(file_list):
    """ Group files based on the ID part of their filenames """
    files_dict = defaultdict(list)
    for filename in file_list:
        id_part = '_'.join(filename.split('.')[0].split("_")[:2])  # Adjust based on your file naming
        files_dict[id_part].append(filename)
    return files_dict

def create_image_label_list(images_dir, labels_dir=None):
    image_files = list_files(images_dir, '.nii.gz')
    grouped_images = group_files_by_id(image_files)

    image_label_list = []
    if labels_dir:
        label_files = list_files(labels_dir, '.nii.gz')
        grouped_labels = group_files_by_id(label_files)
    
    for id_part in grouped_images:
        batch = {}
        images_list = sorted(grouped_images[id_part])  # Sort to ensure correct order

        batch['id'] = id_part
        batch['image'] = images_list

        if labels_dir:
            label_list = sorted(grouped_labels[id_part])
            batch['label'] = label_list[0] # assume only has one segmentation
            batch['gt_phase'] = label_list[0].split(".")[0].split("_")[2] # assume only has one segmentation

        image_label_list.append(batch)

    return image_label_list

def generate_dataset_json(output_file: str, imagesTr_dir: str, labelsTr_dir: str, dataset_name: str, imagesTs_dir: str = None):
    
    json_dict = {}
    json_dict['name'] = dataset_name
    if imagesTs_dir:
        ts_image_label_list = create_image_label_list(imagesTs_dir, None)
        json_dict['num_of_test'] = len(ts_image_label_list)
        json_dict['test'] = ts_image_label_list
    tr_image_label_list = create_image_label_list(imagesTr_dir, labelsTr_dir)
    json_dict['num_of_training'] = len(tr_image_label_list)
    json_dict['training'] = tr_image_label_list

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, output_file)