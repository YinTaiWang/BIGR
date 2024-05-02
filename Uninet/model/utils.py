# from . import uninet
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt; dpi=300
from matplotlib.patches import Patch

class Sanity_check():
    def __init__(self, outputs, data, save_dir):
        self.outputs = outputs.cpu().detach()
        self.image = data[0].cpu()
        self.seg = data[1].cpu()
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

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].set_title(f"Image slice {middle_slice}")
        axes[0].imshow(self.image[0, 0, :, :, middle_slice], cmap="gray")
        axes[1].set_title(f"Mask slice {middle_slice}")
        axes[1].imshow(self.seg[0, 0, :, :, middle_slice])
        axes[2].set_title(f"Output slice {middle_slice}")
        axes[2].imshow(torch.argmax(torch.tensor(self.outputs), dim=1)[0, :, :, middle_slice])
        plt.tight_layout()
        plt.savefig(self.save_dir)
        plt.close(fig)
        
    def seg_4d_plots(self):
        '''
        Plot the middle slice of the result from segmentation along with image and ground truth.
        '''
        middle_slice = self.find_middle_slice_with_label()
        # pair = int(self.outputs.shape[1] / 2) #4
        
        fig, axes = plt.subplots(2,5, figsize=(12,5))
        for i in range(self.image.shape[1]):
            axes[0,i].imshow(self.image[0, i, :, :, middle_slice], cmap="gray")
        axes[0,4].imshow(self.seg[0, 0, :, :, middle_slice])
        
        for i in range(4):
            j = 2 * i
            axes[1,i].imshow(torch.argmax(torch.tensor(self.outputs[:, j:j+2, ...]), dim=1)[0, :, :, middle_slice])
            
        # channel, pair = int(self.outputs.shape[1]), int(self.outputs.shape[1] / 2)
        # for x, y in zip(range(pair), range(pair, int(channel))):
        #     concatenated = torch.cat((self.outputs[:, x:x+1, :, :, :], self.outputs[:, y:y+1, :, :, :]), dim=1)
        #     axes[1,i].imshow(torch.argmax(concatenated, dim=1)[0, :, :, middle_slice])
        
        axes[1,4].imshow(self.seg[0, 0, :, :, middle_slice])
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
    def check_seg_4d(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.seg_4d_plots()
        
        
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
    return means, stds

def plot_cv(history_CV, save_dir):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tr_epoch_loss_m, tr_epoch_loss_std = get_mean_std(history_CV, metric="train_epoch_loss")
    v_epoch_loss_m, v_epoch_loss_std = get_mean_std(history_CV, metric='val_epoch_loss')
    tr_epochs = range(1, len(tr_epoch_loss_m) + 1)
    v_epochs = range(1, len(v_epoch_loss_m) + 1)
    axes[0].plot(tr_epochs, tr_epoch_loss_m, label=f'train')
    axes[0].plot(v_epochs, v_epoch_loss_m, label=f'val')
    axes[0].fill_between(tr_epochs, np.array(tr_epoch_loss_m)-np.array(tr_epoch_loss_std), np.array(tr_epoch_loss_m)+np.array(tr_epoch_loss_std), alpha=0.3)
    axes[0].fill_between(v_epochs, np.array(v_epoch_loss_m)-np.array(v_epoch_loss_std), np.array(v_epoch_loss_m)+np.array(v_epoch_loss_std), alpha=0.3)
    axes[0].legend()
    axes[0].set_title("Epoch Average Loss")
    axes[0].set_xlabel("epoch")

    tr_metric_m, tr_metric_sd = get_mean_std(history_CV, metric="train_metric_values")
    v_metric_m, v_metric_sd = get_mean_std(history_CV, metric='val_metric_values')
    tr_epochs = range(1, len(tr_metric_m) + 1)
    v_epochs = range(1, len(v_metric_m) + 1)
    axes[1].plot(tr_epochs, tr_metric_m, label=f'train')
    axes[1].plot(v_epochs, v_metric_m, label=f'val')
    axes[1].fill_between(tr_epochs, np.array(tr_metric_m) - np.array(tr_metric_sd), np.array(tr_metric_m) + np.array(tr_metric_sd), alpha=0.3)
    axes[1].fill_between(v_epochs, np.array(v_metric_m) - np.array(v_metric_sd), np.array(v_metric_m) + np.array(v_metric_sd), alpha=0.3)
    axes[1].legend()
    axes[1].set_title("Val Mean Dice")
    axes[1].set_xlabel("epoch")
    
    plt.savefig(save_dir)
    plt.close(fig)
    