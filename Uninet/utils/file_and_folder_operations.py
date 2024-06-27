import os
import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
import json
import pickle

class CustomCompactEncoder(json.JSONEncoder):
    """Custom JSON encoder designed to compact arrays specifically."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            # Handle nested lists or simple lists uniformly
            try:
                if all(isinstance(x, list) for x in obj):  # Nested list
                    return '[{}]'.format(', '.join(self.default(sub) for sub in obj))
                else:  # Simple list
                    return '[{}]'.format(', '.join(json.dumps(x) for x in obj))
            except TypeError:
                return json.dumps(obj)
        return super().default(obj)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, cls=CustomCompactEncoder, indent=4, sort_keys=True)
        
def load_json(file_path):
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
    return loaded_data