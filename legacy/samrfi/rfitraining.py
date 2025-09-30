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

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

        self.checkpoint_path = None

        self.sam2_ckpt = None
        self.sam2_cfg = None

    ################
    # SAM Training
    ################

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
        
        ##
        self.train_dataloader_sam1 = train_dataloader

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

    ################
    # SAM 2 Training
    ################

    # Adapted from https://www.datacamp.com/tutorial/sam2-fine-tuning
    def train_sam2(self, num_epochs=3, batch_size=4, sam_checkpoint='small', min_point_distance = 16, step_size = 20, gamma = 0.2, num_points=128, threshold=0.95, plot=True, model_path=None, trained_model_path=None):
        """
        Fine-tune SAM 2 model (instead of the original SAM).
        Valid values for 'sam_checkpoint' are: 'tiny', 'small', 'base_plus', or 'large'.
        """

        # Map user input to valid SAM 2 checkpoints and config files
        checkpoint_config_map = {
            "tiny":      ("sam2_hiera_tiny.pt",      "sam2_hiera_t.yaml"),
            "small":     ("sam2_hiera_small.pt",     "sam2_hiera_s.yaml"),
            "base_plus": ("sam2_hiera_base_plus.pt","sam2_hiera_b+.yaml"),
            "large":     ("sam2_hiera_large.pt",     "sam2_hiera_l.yaml")
        }

        if sam_checkpoint not in checkpoint_config_map:
            raise ValueError("Invalid SAM2 checkpoint. Use 'tiny', 'small', 'base_plus', or 'large'.")

        sam2_ckpt, sam2_cfg = checkpoint_config_map[sam_checkpoint]

        # Build SAM 2 model
        sam2_model = build_sam2(self.sam2_cfg, self.sam2_ckpt, device=self.device)
        predictor = SAM2ImagePredictor(sam2_model)

        # Make sure we only compute (or allow) gradients for the parts we want to train.
        # For SAM 2, we might typically train the mask decoder + prompt encoder.
        predictor.model.sam_mask_decoder.train(True)
        predictor.model.sam_prompt_encoder.train(True)

        scaler = torch.cuda.amp.GradScaler()

        FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"
        # Optionally load a pre-trained state
        if model_path:
            predictor.model.load_state_dict(torch.load(model_path))

        # Create dataset and dataloader with minimal changes to existing structure.
        train_dataset = SAM2Dataset(dataset=self.RFIDataset.dataset)  # Removed the processor usage
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.train_dataloader = train_dataloader

        # Define an optimizer (Adam is kept from the original code, but you could use AdamW)
        # optimizer = Adam(predictor.model.parameters(), lr=1e-5, weight_decay=0)
        optimizer = torch.optim.AdamW(params=predictor.model.parameters(),lr=0.0001,weight_decay=1e-4) #1e-5, weight_decay = 4e-5
        scaler = torch.cuda.amp.GradScaler()
        # Example segmentation loss
        #seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        # Store average mean loss for each epoch
        ave_meanloss = []

        # Set model to training mode
        predictor.model.to(self.device)

        print(f"\nTraining SAM 2 model...")


        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # 500 , 250, gamma = 0.1
        accumulation_steps = 4  # Number of steps to accumulate gradients before updating

        for epoch in range(1, num_epochs + 1):
            epoch_losses = []

            predictor.model.train()
            optimizer.zero_grad()

            for step, batch in tqdm(enumerate(train_dataloader)):

                single_image = batch["image"]  # shape [3, H, W]
                ground_truth_mask = batch["ground_truth_mask"]

                np_image = single_image.cpu().numpy()

                if np_image.shape[0] == 3:
                    np_image = np.transpose(np_image, (1, 2, 0))  # 
                    
                #input_points = np.array(get_peak_points(np_image[:,:,0], min_distance=min_point_distance))

                rows, cols = np.where(ground_truth_mask > 0)
                coords = np.stack((rows, cols), axis=-1)
                np.random.shuffle(coords)
                coords = coords[:num_points]
                coords_xy = coords[:, [1, 0]]

                input_points = coords
                input_labels = np.ones((input_points.shape[0],), dtype=int)

                # fig, ax = plt.subplots(1,2, figsize=(10,5))
                # ax[0].imshow(np_image)
                # ax[0].scatter(input_points[:, 1], input_points[:, 0], c='r', s=10)
                # ax[1].imshow(ground_truth_mask)
                # ax[1].scatter(input_points[:, 1], input_points[:, 0], c='r', s=10)
                # plt.show()

                bounding_box = get_bounding_box(ground_truth_mask.cpu().numpy())
                bounding_box = [float(coord) for coord in bounding_box]
                bounding_box_tensor = torch.tensor([bounding_box], device=self.device).float().unsqueeze(0)

                with torch.cuda.amp.autocast():
                    predictor.set_image(np_image)
                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_points, input_labels, box=bounding_box_tensor, mask_logits=None, normalize_coords=False)
                    if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                        print("No valid points found. Skipping this batch.")
                        continue

                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels),
                        boxes=unnorm_box,
                        masks=None
                    )

                    batched_mode = unnorm_coords.shape[0] > 1
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=batched_mode,
                        high_res_features=high_res_features,
                    )

                    prd_masks = predictor._transforms.postprocess_masks(
                        low_res_masks, predictor._orig_hw[-1]
                    )

                    gt_mask = ground_truth_mask.cuda()
                    prd_mask = torch.sigmoid(prd_masks[:, 0])
                    prd_mask = prd_mask.squeeze(0)

                    prd_mask = 1 - prd_mask
                    # prd_mask = (prd_mask > threshold)

                    # gt_mask = gt_mask.float()
                    # prd_mask = prd_mask.float()
                    
                    self.gt_mask = gt_mask
                    self.prd_mask = prd_mask

                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

                    #print('seg_loss:', seg_loss)

                    inter = (gt_mask * (prd_mask > 0.5)).sum()

                    iou = inter / (gt_mask.sum() + (prd_mask > 0.5).sum() - inter)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = seg_loss + score_loss * 0.05

                    #print(iou.shape)

                    loss = loss / accumulation_steps

                    #print(loss)

                    # Check for NaN before backprop
                    total_loss = seg_loss + 0.05 * (torch.abs(prd_scores[:, 0] - iou).mean())
                    if torch.isnan(total_loss):
                        print("NaN encountered. Skipping this batch.")
                        continue  # skip update

                    scaler.scale(loss).backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

                step = epoch

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_losses.append(loss.item())


            scheduler.step()

                #mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

                #print(mean_iou)

                #print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)


            # End of epoch
            #print(f"EPOCH: {epoch}")
            #print(f"Accuracy (IoU): {mean(epoch_losses)}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean(epoch_losses)}")
            ave_meanloss.append(mean(epoch_losses))

        self.ave_meanloss = ave_meanloss

        # Build file name for saving
        params = self.RFIDataset.dataset_params
        stretch = params["stretch"]
        flag_sigma = params["flag_sigma"]
        patch_method = params["patch_method"]
        patch_size = params["patch_size"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"model_stretch-{stretch}_sigma-{flag_sigma}_patch-{patch_method}_size-{patch_size}_"
            f"sam2-{sam_checkpoint}_epochs{num_epochs}_{timestamp}.pth"
        )

        # Save the trained model
        if trained_model_path:
            try:
                torch.save(predictor.model.state_dict(), trained_model_path)
            except:
                print("Model path not found. Saving model to default directory.")
                method_dir = os.path.join(self.directory, 'models')
                if not os.path.exists(method_dir):
                    os.makedirs(method_dir)
                torch.save(predictor.model.state_dict(), os.path.join(method_dir, filename))
        else:
            method_dir = os.path.join(self.directory, 'models')
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            torch.save(predictor.model.state_dict(), os.path.join(method_dir, filename))

        # Plot if required
        if plot:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
            ax.plot(
                self.ave_meanloss,
                label=(
                    f"Sigma {flag_sigma} {stretch} — "
                    f"Epoch {num_epochs} Patches {len(self.RFIDataset.patched_data_norm_only)}"
                ),
                color="blue"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean Loss")
            ax.set_title("Mean Loss vs Epoch")
            plt.legend()

            filename = (
                f"loss_plot_model_stretch-{stretch}_sigma-{flag_sigma}_patch-{patch_method}_"
                f"size-{patch_size}_sam2-{sam_checkpoint}_{timestamp}.png"
            )
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
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

class SAM2Dataset(TorchDataset):
    """
    Minimal changes: we remove references to huggingface SamProcessor.
    Keep bounding box logic. Return the same item structure needed for SAM 2.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Convert the PIL image to a NumPy array
        image = np.array(item["image"], dtype=np.float32)  # shape (H, W, C)

        ground_truth_mask = np.array(item["label"], dtype=np.float32)  # shape (H, W)
        # Get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        return {
            "image": image,                 # shape (C, H, W)
            "ground_truth_mask": ground_truth_mask,  # shape (H, W)
            "input_boxes": prompt
        }