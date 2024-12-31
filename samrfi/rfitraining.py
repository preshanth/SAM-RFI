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


class RFITraining:

    def __init__(self, rfidataset_instance, device='cuda'):
        self.device = device
        self.RFIDataset = rfidataset_instance

    def train(self, num_epochs=3, stretch='SQRT', flag_sigma=5, patch_method='patchify', patch_size=128, num_patches=None, batch_size=4, sam_checkpoint='huge', plot=True, model_path=None, trained_model_path=None):

        if sam_checkpoint == 'huge':
            sam_type = "sam-vit-huge"
        elif sam_checkpoint == 'base':
            sam_type = "sam-vit-base"
        elif sam_checkpoint == 'large':
            sam_type = "sam-vit-large"
        else:
            raise ValueError("Invalid SAM checkpoint. Use 'huge', 'base', or 'large'.")

        processor = SamProcessor.from_pretrained(f"facebook/{sam_type}")
        model = SamModel.from_pretrained(f"facebook/{sam_type}")

        # Create a new train_dataloader with the updated train_dataset
        train_dataloader = DataLoader(RFIDataset.train_dataset, batch_size=batch_size,shuffle=True,)
        
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


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_stretch-{stretch}_sigma-{flag_sigma}_patch-{patch_method}_size-{patch_size}_sam-{sam_checkpoint}_epochs{num_epochs}_{timestamp}.pth"


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

            ax.plot(self.ave_meanloss, label=f"Sigma {flag_sigma} {stretch} — Epoch {num_epochs} Patches {len(self.patched_data_norm_only)}", color="blue")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean Loss")
            ax.set_title("Mean Loss vs Epoch")

            plt.legend()

            filename = f"loss_plot_model_stretch-{stretch}_sigma-{flag_sigma}_patch-{patch_method}_size-{patch_size}_sam-{sam_checkpoint}_{timestamp}.png"
            fig.savefig(os.path.join(method_dir, filename))
            
            plt.show()