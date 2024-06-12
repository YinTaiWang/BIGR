import os
import re


def find_task_by_id(directory, task_id):
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
