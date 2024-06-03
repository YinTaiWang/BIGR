import csv
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def create_history_dict():
    history = {'total_loss': [],
                'loss_dice': [],
                'loss_consis': [],
                'loss_simi': [],
                'loss_smooth': [],
                'metric_dice': [],
                'metric_ncc': []}
    return history

def scale(image, seg, scale):
    if scale < 1:
        scaled_image = F.interpolate(image, scale_factor=scale, mode='trilinear', align_corners=True, recompute_scale_factor=False)
        scaled_seg = F.interpolate(seg.to(torch.float32), scale_factor=scale, mode='nearest', recompute_scale_factor=False)
    else:
        scaled_image, scaled_seg = image, seg
    return scaled_image, scaled_seg

def scale_to_origin(scaled_image, scaled_seg, scaled_prediction, original_shape):
    image = F.interpolate(scaled_image, size=original_shape, mode='trilinear', align_corners=True, recompute_scale_factor=False)
    seg = F.interpolate(scaled_seg, size=original_shape, mode='nearest', recompute_scale_factor=False)
    prediction = F.interpolate(scaled_prediction, size=original_shape, mode='nearest', recompute_scale_factor=False)
    return image, seg, prediction

def add_padding_to_divisible(image, divisor):
    """
    Pads the image so that its dimensions are divisible by the specified divisor.
    
    Args:
        image (torch.Tensor): The input image tensor of shape (N, C, D, H, W)
        divisor (int): The number to which the dimensions should be divisible.
    
    Returns:
        torch.Tensor: The padded image tensor.
    """
    # Current size of the dimensions
    depth, height, width = image.shape[2], image.shape[3], image.shape[4]

    # Calculate the padding needed to make each dimension divisible by `divisor`
    pad_depth = (divisor - depth % divisor) % divisor
    pad_height = (divisor - height % divisor) % divisor
    pad_width = (divisor - width % divisor) % divisor

    # Define padding for depth, height, and width
    # Padding order in F.pad for 5D tensor is (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    padding = (0, pad_width, 0, pad_height, 0, pad_depth)

    # Apply padding
    padded_image = F.pad(image, padding, "constant", 0)
    # print(f"padding: {image.shape[2:]} -> {padded_image.shape[2:]}")

    return padded_image


def compute_losses(loss_functions, outputs_seg, ori_seg, phase, warped_input_image, template, disp_t2i):
    loss_dice = 0
    for c_phase, c in enumerate(range(0, outputs_seg.shape[1], 2)):
        phase_loss = loss_functions['dice'](outputs_seg[:,c:c+2,...], ori_seg)
        if c_phase != phase:
            phase_loss = phase_loss * 0.8 # give less weight on the phases without ground truth
        loss_dice += phase_loss
    pair = int(outputs_seg.shape[1]/2)
    loss_dice /= pair
            
    loss_consis = loss_functions['consis'](outputs_seg)
    loss_simi = loss_functions['ncc'](warped_input_image, template)
    loss_smooth = loss_functions['grad'](disp_t2i)
    ncc_metric = -loss_simi
    
    total_loss = (loss_dice + loss_simi) * 0.8 + (loss_consis + loss_smooth) * 0.2
    losses = [total_loss, loss_dice, loss_consis, loss_simi, loss_smooth, ncc_metric]
    return total_loss, losses

def update_epoch_stats(stats, losses):
    keys = list(stats.keys())
    for i, key in enumerate(keys): # skip metric_dice
        stats[key] += losses[i].item()
    return stats

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


def get_mean_std(list_w_dicts, metric):
    metric_values = [fold[metric] for fold in list_w_dicts]
    metric_values = list(zip(*metric_values))

    means = [np.mean(epoch) for epoch in metric_values]
    stds = [np.std(epoch) for epoch in metric_values]
    return np.array(means), np.array(stds)

def plot_cv(CV_train_history, CV_val_history, save_dir):
    
    # Create map for title names
    map = {
        'total_loss': 'Average Loss',
        'loss_dice': 'Dice Loss',
        'loss_consis': 'Consistency Loss',
        'loss_ncc': 'Similarity Loss',
        'loss_smo': 'Smoothness Loss',
        'metric_dice': 'Average Dice',
        'metric_ncc': 'Average NCC',
        }
    
    # Create proper setting for plots
    # if we have only two items in the history
    two_plots = False
    items = list(CV_train_history[0].keys())
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