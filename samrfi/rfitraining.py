from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt


import torch
from torch.nn.functional import threshold, normalize, interpolate
from torchvision import transforms
from torch.utils.data import Dataset
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


class RFITraining:
    def __init__(self, rfi_instance, device='cuda'):
        self.rfi_instance = rfi_instance
        self.device = device

    ###############
    # Preprocessing
    ###############

    def apply_stretch(self, stretch='SQRT'):

        images_med = []

        if stretch == 'SQRT':
            stretch_func = np.sqrt
        elif stretch == 'LOG10':
            stretch_func = np.log10
        else:
            raise ValueError("Invalid stretch. Use 'SQRT' or 'LOG10'.")

        for data in tqdm(self.patched_data):
            
            data = stretch(data)

            data = data/np.median(data)

            images_med.append(data)

        images = np.stack(images_med)

        self.patched_data = images

    def create_patched_flags(self,sigma=8):

        flags = []

        for data in tqdm(images):
            stat = stats.median_abs_deviation(data, axis=None)
            median = np.median(data)

            flag = data > (median + (stat * sigma))
            
            flags.append(flag)

        self.patched_flags = np.stack(flags)

    def rm_blank_patches(self):

        # Create a mask indicating which arrays contain only False values
        filtered_flags = [not arr.any() for arr in self.patched_flags]

        # Initialize an empty list to store the filtered arrays
        filtered_images = []
    
        # Iterate over the arrays and their corresponding mask values
        for arr, m in zip(self.patched_data, filtered_flags):
            # If the mask value is False, add the array to the filtered list
            if not m:
                filtered_images.append(arr)

        self.patched_data = np.stack(filtered_images)
        self.patched_flags = np.stack(filtered_flags)

    def randomize_patches(self,):

        # Shuffle the data and flags in unison
        indices = np.random.permutation(filtered_images.shape[0])

        self.patched_data = self.patched_data[indices]
        self.patched_flags = self.patched_flags[indices]

    def create_dataset(self,):

        dataset_dict = {
            "image": [Image.fromarray(img) for img in filtered_images],
            "label": [Image.fromarray(mask) for mask in filtered_flags],
        }

        # Create the dataset using the datasets.Dataset class
        dataset_dict["image"] = [img.convert("RGB") for img in dataset_dict["image"]]

        dataset = Dataset.from_dict(dataset_dict)
        
        self.dataset = dataset

    def train(self, num_epochs=3, stretch='SQRT', flag_sigma=5, patch_method='patchify', patch_size=128, sam_checkpoint='huge', plot=True, model_path=None, trained_model_path=None):

        rfi_combined = four_rotations(self.rfi_instance.rfi_antenna_data)
        
        if patch_method == 'patchify':
            self.patched_data = create_patchify_patches(rfi_combined, patch_size=patch_size)

        self.apply_stretch(stretch=stretch)
        self.create_patched_flags(sigma=flag_sigma)
        self.rm_blank_patches()
        self.randomize_patches()
        self.create_dataset()

        if sam_checkpoint == 'huge':
            sam_type = "sam-vit-huge"
        elif sam_checkpoint == 'base':
            sam_type = "sam-vit-base"
        elif sam_checkpoint == 'large':
            sam_type = "sam-vit-large"
        else:
            raise ValueError("Invalid SAM checkpoint. Use 'huge', 'base', or 'large'.")

        processor = SamProcessor.from_pretrained(f"facebook/{sam_type}")

        # Create a new train_dataset with the updated dataset
        train_dataset = SAMDataset(dataset=dataset, processor=processor)

        model = SamModel.from_pretrained(f"facebook/{sam_type}")

        # Create a new train_dataloader with the updated train_dataset
        train_dataloader = DataLoader(train_dataset, batch_size=4,shuffle=True,)
        
        # make sure we only compute gradients for mask decoder
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        
        if model_path:
            model.load_state_dict(torch.load(model_path))

        optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

        #Try DiceFocalLoss, FocalLoss, DiceCELoss
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        # Training loop
        ave_meanloss = []

        model.to(self.device)
        model.train()

        for epoch in range(num_epochs):
            
            epoch_losses = []

            for batch in tqdm(train_dataloader):
                # forward pass
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                input_boxes=batch["input_boxes"].to(device),
                                multimask_output=False)

                # compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)

                # Ensure ground truth masks are resized to match the predicted masks
                if len(ground_truth_masks.shape) == 3:  # Add channel dimension if missing
                    ground_truth_masks = ground_truth_masks.unsqueeze(1)

                predicted_mask_size = predicted_masks.shape[-2:]  # Get the height and width of the predicted masks
                ground_truth_masks_resized = interpolate(ground_truth_masks, size=predicted_mask_size, mode='bilinear', align_corners=False)

                loss = seg_loss(predicted_masks, ground_truth_masks_resized)

                # backward pass (compute gradients of parameters w.r.t. loss)
                optimizer.zero_grad()
                loss.backward()

                # optimize
                optimizer.step()
                epoch_losses.append(loss.item())

            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')
            ave_meanloss.append(mean(epoch_losses))

            self.ave_meanloss = ave_meanloss


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_stretch{stretch}_sigma{flag_sigma}_patch{patch_method}_size{patch_size}_sam-{sam_checkpoint}_epochs{num_epochs}_{timestamp}.pth"


        if trained_model_path:
            try:
                torch.save(model.state_dict(), trained_model_path)
            except:
                print("Model path not found. Saving model to default directory.")
                method_dir = os.path.join(self.rfi_instance.directory, 'models')
                
                if not os.path.exists(method_dir):
                    os.makedirs(method_dir)
                torch.save(model.state_dict(), os.path.join(method_dir, filename))
        else:
            method_dir = os.path.join(self.rfi_instance.directory, 'models')
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            
            torch.save(model.state_dict(), os.path.join(method_dir, filename))

        if plot:
            plt.clf()

            fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

            ax.plot(self.ave_meanloss, label=f"Sigma {flag_sigma} {stretch} ({len(self.patched_data)})", color="blue")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean Loss")
            ax.set_title("Mean Loss vs Epoch")

            plt.legend()

            filename = f"loss_plot_model_stretch-{stretch}_sigma-{flag_sigma}_patch-{patch_method}_size-{patch_size}_sam-{sam_checkpoint}_{timestamp}.png"
            fig.savefig(os.path.join(method_dir, filename))
            
            plt.show()


class SAMDataset(Dataset):
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