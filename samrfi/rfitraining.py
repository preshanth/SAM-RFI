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
from .rfidataset import RFIDataset
from .utilities import *


class RFITraining:

    def __init__(self, rfidataset_instance, device='cuda', dir_path=None):
        self.device = device
        self.RFIDataset = rfidataset_instance

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

    def train(self, num_epochs=3, batch_size=4, sam_checkpoint='huge', plot=True, model_path=None, trained_model_path=None):


        if sam_checkpoint == 'huge':
            sam_type = "sam-vit-huge"
        elif sam_checkpoint == 'base':
            sam_type = "sam-vit-base"
        elif sam_checkpoint == 'large':
            sam_type = "sam-vit-large"
        else:
            raise ValueError("Invalid SAM checkpoint. Use 'huge', 'base', or 'large'.")

        processor = SamProcessor.from_pretrained(f"facebook/{sam_type}")

        train_dataset = SAMDataset(dataset=self.RFIDataset.dataset, processor=processor)

        model = SamModel.from_pretrained(f"facebook/{sam_type}")

        # Create a new train_dataloader with the updated train_dataset
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,)
        
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

        print(f"\nTraining model...")

        for epoch in range(num_epochs):
            
            epoch_losses = []

            for batch in tqdm(train_dataloader):
                # forward pass
                outputs = model(pixel_values=batch["pixel_values"].to(self.device),
                                input_boxes=batch["input_boxes"].to(self.device),
                                multimask_output=False)

                # compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)

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

        params = self.RFIDataset.dataset_params

        stretch = params["stretch"]
        flag_sigma = params["flag_sigma"]
        patch_method = params["patch_method"]
        patch_size = params["patch_size"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_stretch-{stretch}_sigma-{flag_sigma}_patch-{patch_method}_size-{patch_size}_sam-{sam_checkpoint}_epochs{num_epochs}_{timestamp}.pth"

        if trained_model_path:
            try:
                torch.save(model.state_dict(), trained_model_path)
            except:
                print("Model path not found. Saving model to default directory.")
                method_dir = os.path.join(self.directory, 'models')
                
                if not os.path.exists(method_dir):
                    os.makedirs(method_dir)
                torch.save(model.state_dict(), os.path.join(method_dir, filename))
        else:
            method_dir = os.path.join(self.directory, 'models')
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            
            torch.save(model.state_dict(), os.path.join(method_dir, filename))

        if plot:
            plt.clf()

            fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

            ax.plot(self.ave_meanloss, label=f"Sigma {flag_sigma} {stretch} — Epoch {num_epochs} Patches {len(self.RFIDataset.patched_data_norm_only)}", color="blue")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean Loss")
            ax.set_title("Mean Loss vs Epoch")

            plt.legend()

            filename = f"loss_plot_model_stretch-{stretch}_sigma-{flag_sigma}_patch-{patch_method}_size-{patch_size}_sam-{sam_checkpoint}_{timestamp}.png"
            fig.savefig(os.path.join(method_dir, filename))
            
            plt.show()

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