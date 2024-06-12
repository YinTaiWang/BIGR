import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pickle
from .common_utils import find_task_by_id

def set_paths(root_path, task_id):
    raw_data_dir = os.path.join(root_path, "raw_data")
    task_name = find_task_by_id(raw_data_dir, task_id)
    RawTask_dir = os.path.join(raw_data_dir, task_name)
    
    # Save the path to "preprocessed/Task..."
    PreprocessTask_dir = os.path.join(root_path, "preprocessed", task_name)
    if not os.path.exists(PreprocessTask_dir):
        os.makedirs(PreprocessTask_dir, exist_ok=True)
    
    # Set the path to imagesTr and labelsTr in `raw_data` and `preprocessed` folders
    RawIMG_dir = os.path.join(RawTask_dir, "imagesTr")
    RawSEG_dir = os.path.join(RawTask_dir, "labelsTr")
    PreprocessIMG_dir = os.path.join(PreprocessTask_dir, "imagesTr")
    PreprocessSEG_dir = os.path.join(PreprocessTask_dir, "labelsTr")
    # Create subfolders in `preprocessed` folder
    if not os.path.exists(PreprocessIMG_dir):
        os.makedirs(PreprocessIMG_dir, exist_ok=True)
    if not os.path.exists(PreprocessSEG_dir):
        os.makedirs(PreprocessSEG_dir, exist_ok=True)
    return task_name, RawTask_dir, PreprocessTask_dir, RawIMG_dir, RawSEG_dir, PreprocessIMG_dir, PreprocessSEG_dir
        

def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def save_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
        
def load_json(file_path):
    with open(file_path, "r") as json_file:
        loaded_data = json.load(json_file)
    return loaded_data