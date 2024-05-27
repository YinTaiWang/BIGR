import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt; dpi=300
from matplotlib.patches import Patch

class Sanity_check():
    def __init__(self, outputs, data, save_dir):
        
        if len(outputs) == 1:
            self.outputs = outputs.cpu().detach()
        elif len(outputs) > 1:
            self.regoutputs = outputs[0].cpu().detach()
            self.segoutputs = outputs[1].cpu().detach()
        else:
            raise ValueError("Only takes one of the reg or seg output or both.")
        
        if len(data) == 2:
            self.image = data[0].cpu()
            self.seg = data[1].cpu()
        elif len(data) == 3:
            self.image = data[0].cpu()
            self.seg = data[1].cpu()
            self.phase = data[2]
        else:
            raise ValueError("data should contains image, segmentation, (phase) in a tuple.")
        
        self.save_dir = save_dir
        
    def find_middle_slice_with_label(self, label=1):
        first_slice = None
        last_slice = None
        slices = self.seg.shape[-1]
        
        for i in range(slices):
            if label in self.seg[..., i]:
                if first_slice is None:
                    first_slice = i  # Found the first slice with the label
                last_slice = i  # Update last slice with the label at each find
        
        if first_slice is not None:
            return round((first_slice + last_slice) / 2)
        else:
            return None  # Return None if the label is not found in any slice

    def reg_plots(self):
        '''
        Plot the result of registration along with the original image.
        '''
        middle_slice = self.find_middle_slice_with_label()

        cmaps = ['Blues', 'Reds', 'Greens', 'Oranges']
        phases = ['Precontrast', 'Arterial', 'Portal venous', 'Delayed']
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        plt.subplots_adjust(left=0.03, top=0.9, bottom=0.1, wspace=0.05, hspace=0.3)
        
        for i in range(self.image.shape[1]):
            axes[0, i].imshow(self.image[0, i, :, :, middle_slice], cmap='gray')
            axes[0, 4].imshow(self.image[0, i, :, :, middle_slice], cmap=cmaps[i], alpha=0.8 - 0.2 * i)
            axes[1, i].imshow(self.outputs[0, i, :, :, middle_slice], cmap='gray')
            axes[1, 4].imshow(self.outputs[0, i, :, :, middle_slice], cmap=cmaps[i], alpha=0.8 - 0.2 * i)

        proxies = [Patch(color=plt.get_cmap(cmap)(0.8 - 0.2 * i), label=phase) for i, (cmap, phase) in enumerate(zip(cmaps, phases))]
        axes[0, 4].legend(handles=proxies, loc='upper left', bbox_to_anchor=(1.05, 1))
        axes[1, 4].legend(handles=proxies, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        row_titles = ['Original Images', 'Warped Images']
        for ax, title in zip(axes[:, 2], row_titles):
            ax.set_title(title, size=18, y=1.03)
        
        plt.savefig(self.save_dir)
        plt.close(fig)

    def seg_plots(self):
        '''
        Plot the middle slice of the result from segmentation along with image and ground truth.
        '''
        middle_slice = self.find_middle_slice_with_label()

        fig, axes = plt.subplots(1, 3, figsize=(12,5))
        axes[0].set_title(f"Image slice {middle_slice}")
        axes[0].imshow(self.image[0, 0, :, :, middle_slice], cmap="gray")
        axes[1].set_title(f"Mask slice {middle_slice}")
        axes[1].imshow(self.seg[0, 0, :, :, middle_slice])
        axes[2].set_title(f"Output slice {middle_slice}")
        axes[2].imshow(torch.argmax(torch.tensor(self.outputs), dim=1)[0, :, :, middle_slice])
        plt.tight_layout()
        plt.savefig(self.save_dir)
        plt.close(fig)
        
    def seg4d_plots(self):
        '''
        Plot the middle slice of the result from segmentation along with image and ground truth.
        '''
        middle_slice = self.find_middle_slice_with_label()
        
        fig, axes = plt.subplots(2,5, figsize=(12,5))
        for i in range(self.image.shape[1]):
            axes[0,i].imshow(self.image[0, i, :, :, middle_slice], cmap="gray")
        axes[0,4].imshow(self.seg[0, 0, :, :, middle_slice])
        
        for i in range(4):
            j = 2 * i
            axes[1,i].imshow(torch.argmax(torch.tensor(self.outputs[:, j:j+2, ...]), dim=1)[0, :, :, middle_slice])
        
        axes[1,4].imshow(self.seg[0, 0, :, :, middle_slice])
        plt.tight_layout()
        plt.savefig(self.save_dir)
        plt.close(fig)
        
    def reg_seg_plots(self):
        '''
        Plot the middle slice of the result from registration and segmentation together,
        along with the original images and ground truth.
        '''
        middle_slice = self.find_middle_slice_with_label()
        n_images = self.image.shape[1]
        
        fig, axes = plt.subplots(3, n_images+1, figsize=(10,5))
        # plt.subplots_adjust(left=0.03, top=0.9, bottom=0.1, wspace=0.05, hspace=0.3)
        
        cmaps = ['Blues', 'Reds', 'Greens', 'Oranges']
        phases = ['Precontrast', 'Arterial', 'Portal venous', 'Delayed']
        
        for i in range(n_images):
            # original images
            axes[0, i].imshow(self.image[0, i, :, :, middle_slice], cmap="gray")
            axes[0, 4].imshow(self.image[0, i, :, :, middle_slice], cmap=cmaps[i], alpha=0.8 - 0.2 * i)
            # registration
            axes[1, i].imshow(self.regoutputs[0, i, :, :, middle_slice], cmap='gray')
            axes[1, 4].imshow(self.regoutputs[0, i, :, :, middle_slice], cmap=cmaps[i], alpha=0.8 - 0.2 * i)
            # segmentation
            j = 2 * i
            axes[2,i].imshow(torch.argmax(torch.tensor(self.segoutputs[:, j:j+2, ...]), dim=1)[0, :, :, middle_slice])
            axes[2,4].imshow(self.seg[0, 0, :, :, middle_slice])
            
        proxies = [Patch(color=plt.get_cmap(cmap)(0.8 - 0.2 * i), label=phase) for i, (cmap, phase) in enumerate(zip(cmaps, phases))]
        axes[0, 4].legend(handles=proxies, loc='upper left', bbox_to_anchor=(1.05, 1))
        axes[1, 4].legend(handles=proxies, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.savefig(self.save_dir)
        plt.close(fig)
    
    @classmethod
    def check_reg(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.reg_plots()
    
    @classmethod
    def check_seg(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.seg_plots()
    @classmethod
    def check_seg4d(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.seg4d_plots()
        
    @classmethod
    def check_reg_seg(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.reg_seg_plots()
        
        
###################
##   Functions   ##
###################
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
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    overviews = ['epoch_total_loss', 'epoch_metric_dice', 'epoch_metric_ncc']
    titles = ['Epoch Average Loss', 'Val Mean Dice', 'Val Mean NCC']
    
    for i, overview in enumerate(overviews):
        train_mean, train_std = get_mean_std(CV_train_history, metric=overview)
        val_mean, val_std = get_mean_std(CV_val_history, metric=overview)
        train_epochs = range(1, len(train_mean) + 1)
        val_epochs = range(1, len(val_mean) + 1)
        
        axes[i].plot(train_epochs, train_mean, label=f'train')
        axes[i].plot(val_epochs, val_mean, label=f'val')
        axes[i].fill_between(train_epochs, train_mean - train_std, train_mean + train_std, alpha=0.3)
        axes[i].fill_between(val_epochs, val_mean - val_std, val_mean + val_std, alpha=0.3)
        axes[i].legend()
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Epoch")
    
    plt.tight_layout()
    file_name = "overview.png"
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close(fig)

    ##############################################################################
    # plot separate losses
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    losses = ['epoch_loss_dice', 'epoch_loss_consis', 'epoch_loss_simi', 'epoch_loss_smooth']
    
    for i, loss in enumerate(losses):
        train_mean, train_std = get_mean_std(CV_train_history, metric=loss)
        val_mean, val_std = get_mean_std(CV_val_history, metric=loss)
        train_epochs = range(1, len(train_mean) + 1)
        val_epochs = range(1, len(val_mean) + 1)
        title = loss.split('_')[-1].capitalize()
        
        row = i // 2
        col = i % 2
        axes[row, col].plot(train_epochs, train_mean, label=f'train')
        axes[row, col].plot(val_epochs, val_mean, label=f'val')
        axes[row, col].fill_between(train_epochs, train_mean - train_std, train_mean + train_std, alpha=0.3)
        axes[row, col].fill_between(val_epochs, val_mean - val_std, val_mean + val_std, alpha=0.3)
        axes[row, col].legend()
        axes[row, col].set_title(f"{title} Loss")
        axes[row, col].set_xlabel("Epoch")

    plt.tight_layout()
    file_name = "losses.png"
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close(fig)