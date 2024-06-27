import os
import re

def id_to_taskname(directory, task_id):
    # Ensure task_id is formatted as a three-digit number with leading zeros
    task_id_str = str(task_id).zfill(3)
    
    # Define the pattern with a capturing group for the XXX part
    pattern = re.compile(r"Task(\d{3})_.*")

    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        if match and os.path.isdir(os.path.join(directory, folder_name)):
            # Extract the captured group
            folder_task_id = match.group(1)
            if folder_task_id == task_id_str:
                return folder_name

    # If no matching folder is found, raise a ValueError
    raise ValueError(f"The folder is not found. Expect folder ./raw_data/Task{task_id_str}_*.")

def set_preprocess_path_by_id(root_path, task_id):
    raw_data_dir = os.path.join(root_path, "raw_data")
    task_name = id_to_taskname(raw_data_dir, task_id)
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

def set_results_path_by_id(root_path, task_id):
    preprocess_dir = os.path.join(root_path, "preprocessed")
    task_name = id_to_taskname(preprocess_dir, task_id)
    PreprocessTask_dir = os.path.join(preprocess_dir, task_name)
    
    # Save the path to "results/Task..."
    ResultsTask_dir = os.path.join(root_path, "results", task_name)
    if not os.path.exists(ResultsTask_dir):
        os.makedirs(ResultsTask_dir, exist_ok=True)
    
    return PreprocessTask_dir, ResultsTask_dir