import os
import torch
import numpy as np
import matplotlib.pyplot as plt; dpi=300
from matplotlib.patches import Patch

class Sanity_check():
    def __init__(self, outputs, data, save_dir):
        ## Check outputs
        # reg4d, data has length 1
        # seg4d, post processed data has length 4
        if len(outputs) == 1 or len(outputs) == 4:
            self.outputs = outputs
        elif len(outputs) == 2:
            self.regoutputs = outputs[0].cpu().detach()
            self.segoutputs = outputs[1].cpu().detach()
        else:
            raise ValueError("Only takes one of the reg or seg output or both.")
        
        ## Check data
        if len(data) == 4:
            self.image = data[0].cpu()
            self.seg = data[1].cpu()
            self.patient = data[2]
            self.phase = data[3]
        else:
            raise ValueError("data should contains (image, segmentation, patient, phase) in a tuple.")
        
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

    def reg4d_plots(self):
        '''
        Plot the result of registration along with the original image.
        '''
        middle_slice = self.find_middle_slice_with_label()

        cmaps = ['Blues', 'Reds', 'Greens', 'Oranges']
        phases = ['Precontrast', 'Arterial', 'Portal venous', 'Delayed']
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        fig.suptitle(f"{self.patient}_slice {middle_slice}", fontsize=16, fontweight='bold')
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
        
        plt.tight_layout()
        plt.savefig(self.save_dir)
        plt.close(fig)

    def seg3d_plots(self):
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
        fig.suptitle(f"{self.patient}_slice {middle_slice} (phase={self.phase})", fontsize=16, fontweight='bold')
        
        # First row -- images
        for i in range(self.image.shape[1]):
            axes[0,i].imshow(self.image[0, i, :, :, middle_slice], cmap="gray")
        
        # Second row -- predictions
        for i in range(4):
            axes[1,i].imshow(self.outputs[i].cpu().detach()[1, :, :, middle_slice])
            # First row, last pic -- images + seg
            if i == self.phase:
                axes[0,4].imshow(self.image[0, i, :, :, middle_slice], cmap="gray") # image
                axes[0,4].imshow(self.outputs[i].cpu().detach()[1, :, :, middle_slice], alpha=0.3) # seg
        
        # Second row, last pic -- ground truth
        axes[1,4].imshow(self.seg[0, 0, :, :, middle_slice])
                
        plt.tight_layout()
        plt.savefig(self.save_dir)
        plt.close(fig)
        
    def joint_plots(self):
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
    def check_reg4d(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.reg4d_plots()
    
    @classmethod
    def check_seg3d(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.seg3d_plots()
    @classmethod
    def check_seg4d(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.seg4d_plots()
        
    @classmethod
    def check_joint4d(cls, outputs, data, save_dir):
        instance = cls(outputs, data, save_dir)
        instance.joint4d_plots()
        