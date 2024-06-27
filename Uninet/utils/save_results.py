import os
import csv
import numpy as np
import matplotlib.pyplot as plt; dpi=300

def write_csv(dictionary, save_dir):
    '''
    Args:
        dictionary: a dictionary containing the loss and metric values
        save_dir: directory to save the CSV file
    '''
    with open(save_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        # Find the maximum length of the data to set the number of epochs
        max_length = max(len(v) for v in dictionary.values())
        # Write the header
        writer.writerow(['Epoch'] + list(dictionary.keys()))

        # Write data for each epoch
        for i in range(max_length):
            row = [i + 1]  # Epoch number
            for key in dictionary.keys():
                try:
                    # Try to add the value for this epoch, if it exists
                    row.append(dictionary[key][i])
                except IndexError:
                    # If the value doesn't exist for this metric at this epoch, add a blank
                    row.append('')
            writer.writerow(row)

    print(f"{save_dir} created")

def plot_one_fold_results(train_history, val_history, save_dir):
    
    if isinstance(train_history, dict):
        keys = list(train_history.keys())
    elif hasattr(train_history, 'keys'):  # In case it is a DataFrame
        keys = list(train_history.keys())
    elif isinstance(train_history, tuple) and hasattr(train_history[0], 'keys'):
        train_history = train_history[0]  # Assuming the DataFrame is the first element
        val_history = val_history[0]  # Same assumption for val_history
        keys = list(train_history.keys())
    else:
        raise ValueError("Unsupported type for history")
    
    if 'Epoch' in keys:
        keys.remove('Epoch')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i, key in enumerate(keys):
        axes[i].plot(train_history[key], label='train')
        axes[i].plot(val_history[key], label='val')
        axes[i].legend()
        axes[i].set_xlabel("Epoch")
    axes[0].set_title("Epoch Average Loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Epoch Average Dice")
    axes[1].set_ylabel("Dice")
        
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close(fig)

def get_mean_std(list_w_dicts, metric):
    metric_values = [fold[metric] for fold in list_w_dicts]
    metric_values = list(zip(*metric_values))

    means = [np.mean(epoch) for epoch in metric_values]
    stds = [np.std(epoch) for epoch in metric_values]
    return np.array(means), np.array(stds)

def plot_cross_validation_results(CV_train_history, CV_val_history, save_dir):
    
    # Create map for title names
    map = {
        'total_loss': 'Overall Average Loss',
        'loss_dice': 'Overall Dice Loss',
        'loss_consis': 'Overall Consistency Loss',
        'loss_ncc': 'Overall Similarity Loss',
        'loss_smo': 'Overall Smoothness Loss',
        'metric_dice': 'Overall Average Dice',
        'metric_ncc': 'Overall Average NCC',
        }
    
    # Create proper setting for plots
    # if we have only two items in the history
    two_plots = False
    
    train_history = CV_train_history[0] # just take one
    if isinstance(train_history, dict):
        items = list(train_history.keys())
    elif hasattr(train_history, 'keys'):  # In case it is a DataFrame
        items = list(train_history.keys())
    elif isinstance(train_history, tuple) and hasattr(train_history[0], 'keys'):
        train_history = train_history[0]  # Assuming the DataFrame is the first element
        val_history = val_history[0]  # Same assumption for val_history
        items = list(train_history.keys())
    else:
        raise ValueError("Unsupported type for train_history")
    
    if 'Epoch' in items:
        items.remove('Epoch')
    
    n_subplots = len(items)
    items_1 = items
    
    # if we have more than two items
    if n_subplots > 2:
        two_plots = True
        n_subplots = 3
        n_subplots_2 = len(items) - n_subplots
        items_1 = [item for item in items if 'loss_' not in item]
        items_2 = [item for item in items if 'loss_' in item]
        
        
    fig, axes = plt.subplots(1, n_subplots, figsize=(20, 5))
    for i, item in enumerate(items_1):
        train_mean, train_std = get_mean_std(CV_train_history, metric=item)
        val_mean, val_std = get_mean_std(CV_val_history, metric=item)
        train_epochs = range(1, len(train_mean) + 1)
        val_epochs = range(1, len(val_mean) + 1)
        
        axes[i].plot(train_epochs, train_mean, label=f'train')
        axes[i].plot(val_epochs, val_mean, label=f'val')
        axes[i].fill_between(train_epochs, train_mean - train_std, train_mean + train_std, alpha=0.3)
        axes[i].fill_between(val_epochs, val_mean - val_std, val_mean + val_std, alpha=0.3)
        axes[i].legend()
        axes[i].set_title(map[item])
        axes[i].set_xlabel("Epoch")
    
    plt.tight_layout()
    file_name = "overview.png"
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close(fig)

    ##############################################################################
    # Plot separate losses
    if two_plots:
        fig, axes = plt.subplots(int(n_subplots_2/2), 2, figsize=(20, 10))
        
        for i, item in enumerate(items_2):
            train_mean, train_std = get_mean_std(CV_train_history, metric=item)
            val_mean, val_std = get_mean_std(CV_val_history, metric=item)
            train_epochs = range(1, len(train_mean) + 1)
            val_epochs = range(1, len(val_mean) + 1)
            
            row = i // 2
            col = i % 2
            axes[row, col].plot(train_epochs, train_mean, label=f'train')
            axes[row, col].plot(val_epochs, val_mean, label=f'val')
            axes[row, col].fill_between(train_epochs, train_mean - train_std, train_mean + train_std, alpha=0.3)
            axes[row, col].fill_between(val_epochs, val_mean - val_std, val_mean + val_std, alpha=0.3)
            axes[row, col].legend()
            axes[row, col].set_title(map[item])
            axes[row, col].set_xlabel("Epoch")

        plt.tight_layout()
        file_name = "overview_2.png"
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)