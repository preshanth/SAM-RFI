from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import threshold, normalize, interpolate
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai

from transformers import SamProcessor, SamModel

from datasets import Dataset
from PIL import Image

import numpy as np
from scipy import stats

import os
from datetime import datetime

from .syntheticrfi import SyntheticRFI
from .radiorfi import RadioRFI
from .utilities import *

class RFIDataset:
    def __init__(self, rfi_instance, dir_path=False):
        self.rfi_instance = rfi_instance

        if dir_path:
            if dir_path.endswith('/'):
                dir_path = dir_path[:-1]

            current_directory = str(dir_path)
        else:
            current_directory = os.getcwd()

        new_directory = os.path.join(current_directory, 'samrfi_data')

        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        
        self.directory = new_directory

    def apply_normalization(self, before_stretch=False):

        images_med = []

        print(f"\nApplying median normalization only to {len(self.patched_data)} patches...")

        for data in tqdm(self.patched_data):

            # Normalization
            data = data/np.nanmedian(data)

            images_med.append(data)

        images = np.stack(images_med)
        
        # The stretching only affects the final outcome of the masks, since we want the image itself to be as is when training.

        if before_stretch:
            self.patched_data_norm_only = images
        else:
            self.patched_data = images


    def apply_stretch(self, stretch='SQRT'):

        if stretch == 'SQRT':
            stretch_func = np.sqrt
        elif stretch == 'LOG10':
            stretch_func = np.log10
        else:
            raise ValueError("Invalid stretch. Use 'SQRT' or 'LOG10'.")

        print(f"\nApplying {stretch} stretch and normalization to {len(self.patched_data)} patches...")
        images_med = []

        for data in tqdm(self.patched_data):
            
            # epsilon = 1e-10  # A small positive number
            # data = np.where(data == 0, epsilon, data)

            data = stretch_func(np.abs(data))

            finite_data = data[np.isfinite(data)]
            mad = stats.median_abs_deviation(finite_data, nan_policy='omit')

            # Identify the indices of infinite values (inf and -inf)
            inf_mask = np.isinf(data)

            # Replace infinite values with the MAD
            data[inf_mask] = mad

            images_med.append(data)

        images = np.stack(images_med)

        self.patched_data = images

    def create_patched_flags(self,sigma=8):

        flags = []

        print(f"\nCreating sigma {sigma} flags for each data patch...")

        for data in tqdm(self.patched_data):
            stat = stats.median_abs_deviation(data, axis=None)
            median = np.nanmedian(data)

            # Calculate upper and lower thresholds
            upper_threshold = median + (stat * sigma)
            lower_threshold = median - (stat * sigma)

            # Flag data points outside the thresholds
            flag = (data > upper_threshold) | (data < lower_threshold)
            
            flags.append(flag)

        self.patched_flags = np.stack(flags)

    def rm_blank_patches(self):

        filtered_flags = [arr for arr in self.patched_flags if arr.any()]
        # Create a mask indicating which arrays contain only False values
        filtered_flags_im = [not arr.any() for arr in self.patched_flags]

        # Initialize an empty list to store the filtered arrays
        filtered_images = []
        filtered_images_norm_only = []
    
        # Iterate over the arrays and their corresponding mask values
        for arr, m in zip(self.patched_data, filtered_flags_im):
            # If the mask value is False, add the array to the filtered list
            if not m:
                filtered_images.append(arr)

        for arr, m in zip(self.patched_data_norm_only, filtered_flags_im):
            # If the mask value is False, add the array to the filtered list
            if not m:
                filtered_images_norm_only.append(arr)

        self.patched_data_norm_only = np.stack(filtered_images_norm_only)
        self.patched_flags = np.stack(filtered_flags)
        self.patched_data = np.stack(filtered_images)

    def randomize_patches(self,):

        # Shuffle the data and flags in unison
        indices = np.random.permutation(len(self.patched_data_norm_only))

        self.patched_data_norm_only = self.patched_data_norm_only[indices]
        self.patched_data = self.patched_data[indices]
        self.patched_flags = self.patched_flags[indices]

    def create_dataset(self, stretch='SQRT', flag_sigma=5, patch_method='patchify', patch_size=128, num_patches=None, apply_stretching=True, custom_flag=True):

        # Storing parameters
        self.dataset_params = {
            "stretch": stretch,
            "flag_sigma": flag_sigma,
            "patch_method": patch_method,
            "patch_size": patch_size,
            "num_patches": num_patches,
            "apply_stretching": apply_stretching,
            "custom_flag": custom_flag,
        }

        rfi_combined = four_rotations(self.rfi_instance.rfi_antenna_data)
        
        if patch_method == 'patchify':
            self.patched_data = create_patchify_patches(rfi_combined, patch_size=patch_size)

        # Store normalization without stretching in a seperate variable
        self.apply_normalization(before_stretch=True)

        if apply_stretching:
            self.apply_stretch(stretch=stretch)
        
        self.apply_normalization(before_stretch=False)

        if custom_flag == True:
        
            rfi_flags_combined = four_rotations(self.rfi_instance.flags)
            
            if patch_method == 'patchify':
                self.patched_flags = create_patchify_patches(rfi_flags_combined, patch_size=patch_size)
            
        else:
            self.create_patched_flags(sigma=flag_sigma)
            
        self.rm_blank_patches()
        self.randomize_patches()

        print(self.patched_data_norm_only.shape, self.patched_flags.shape)

        if num_patches:
            self.patched_data_norm_only = self.patched_data_norm_only[:num_patches]
            self.patched_flags = self.patched_flags[:num_patches]

        dataset_dict = {
            "image": [Image.fromarray(img) for img in self.patched_data_norm_only],
            "label": [Image.fromarray(mask) for mask in self.patched_flags],
        }

        # Create the dataset using the datasets.Dataset class
        dataset_dict["image"] = [img.convert("RGB") for img in dataset_dict["image"]]

        dataset = Dataset.from_dict(dataset_dict)
        
        self.dataset = dataset

    ### Add a SAVE method to save the dataset to a file

    def save_dataset(self, dataset_path=False):

        params = self.dataset_params

        stretch = params["stretch"]
        flag_sigma = params["flag_sigma"]
        patch_method = params["patch_method"]
        patch_size = params["patch_size"]
        custom_flag = params["custom_flag"]
        apply_stretching = params["apply_stretching"]
        num_patches = params["num_patches"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename_parts = [
            f"dataset",
            f"patch-{patch_method}",
            f"size-{patch_size}",
            timestamp
        ]

        filename_parts.insert(1, f"num_patches-{self.dataset.shape[0]}")

        if not custom_flag:
            filename_parts.insert(1, f"sigma-{flag_sigma}")

        if apply_stretching:
            filename_parts.insert(1, f"stretch-{stretch}")

        filename = "_".join(filename_parts)
        
        if dataset_path:
            try:
                self.dataset.save_to_disk(os.path.join(method_dir, dataset_path))
            except:
                print("Dataset path not found. Saving model to default directory.")
                method_dir = os.path.join(self.directory, 'datasets')
                
                if not os.path.exists(method_dir):
                    os.makedirs(method_dir)
                self.dataset.save_to_disk(os.path.join(method_dir, filename))
        else:
            method_dir = os.path.join(self.directory, 'datasets')
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            
            self.dataset.save_to_disk(os.path.join(method_dir, filename))