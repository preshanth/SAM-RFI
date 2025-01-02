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
    def __init__(self, rfi_instance):
        self.rfi_instance = rfi_instance

    def apply_stretch(self, stretch='SQRT'):
        
        images_med = []

        if stretch == 'SQRT':
            stretch_func = np.sqrt
        elif stretch == 'LOG10':
            stretch_func = np.log10
        else:
            raise ValueError("Invalid stretch. Use 'SQRT' or 'LOG10'.")

        print(f"\nApplying median normalization only to {len(self.patched_data)} patches...")

        for data in tqdm(self.patched_data):

            data = data/np.nanmedian(data)

            images_med.append(data)

        images = np.stack(images_med)

        self.patched_data_norm_only = images

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

            data = data/np.nanmedian(data)

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

    def create_dataset(self, num_patches=None):

        rfi_combined = four_rotations(self.rfi_instance.rfi_antenna_data)
        
        if patch_method == 'patchify':
            self.patched_data = create_patchify_patches(rfi_combined, patch_size=patch_size)

        if custom == True:
            self.apply_stretch(stretch=stretch)

            rfi_flags_combined = four_rotations(self.rfi_instance.flags)
            
            if patch_method == 'patchify':
                self.patched_flags = create_patchify_patches(rfi_flags_combined, patch_size=patch_size)
            
        else:
            self.apply_stretch(stretch=stretch)
            self.create_patched_flags(sigma=flag_sigma)
            
        self.rm_blank_patches()
        self.randomize_patches()
        self.create_dataset(num_patches=num_patches)

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

        self.train_dataset = SAMDataset(dataset=self.dataset, processor=processor)


    ### Add a SAVE method to save the dataset to a file

class SAMDataset(TorchDataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        # self.resize_transform = transforms.Compose([
        # transforms.Resize((2000, 2000)),
        # # Add other transformations here if necessary
        # ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        
        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)
        # input_pointsa = get_peak_points(real_array)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]],return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs