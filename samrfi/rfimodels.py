import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pandas as pd
from casatools import table
from astropy.visualization import ZScaleInterval, ImageNormalize
from transformers import SamModel, SamConfig, SamProcessor
import torch
from PIL import Image
import time
import os
from tqdm import tqdm
from scipy import stats

import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .radiorfi import RadioRFI
from .utilities import *

class RFIModels:

    def __init__(self, radiorfi_instance, device='cuda',):

        self.RadioRFI = radiorfi_instance

        # sam_checkpoint = str(sam_checkpoint)
        # sam = sam_model_registry[self.sam_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)

        # self.mask_generator = SamAutomaticMaskGenerator(sam)
        # self.predictor = SamPredictor(sam)

        print(self.RadioRFI.rfi_antenna_data.shape)

        self.RadioRFI.update_flags('Flags updated')

        self.sam_type = None
        self.sam2_cfg = None
        self.sam2_ckpt = None
        self.device = device

    def run_sam(self,remove_largest=True,pad_width=50):

        self.pad_width = pad_width
        self.pad_spectrograph(pad_width=pad_width)
        self.create_RGB_channels()

        masks = self.mask_generator.generate(self.test_image)

        if remove_largest:
            masks = self.rm_largest_mask(masks)
        
        self.flags = self.create_flags(masks)

        self.flags = self.flags[pad_width:-pad_width,pad_width:-pad_width]
        self.spectrograph = self.temp_spectrograph[pad_width:-pad_width,pad_width:-pad_width]

    def run_sam_predict(self,pad_width=50):

        self.pad_width = pad_width
        # self.pad_spectrograph(pad_width=pad_width)
        # self.create_RGB_channels()
        self.predictor.set_image(self.test_image)

        self.find_spectrograph_peaks()
        masks, scores, logits = self.predictor.predict(
            point_coords=self.max_peaks[0],
            point_labels=self.max_peaks[1],
            multimask_output=False,
        )

        self.masks = masks
        self.scores = scores
        self.logits = logits
        self.flags = np.logical_not(masks[0])
        # self.flags = self.flags[pad_width:-pad_width,pad_width:-pad_width]
        # self.spectrograph = self.spectrograph[pad_width:-pad_width,pad_width:-pad_width]

    def load_model(self,model_path):
        # "/home/gpuhost002/ddeal/RFI-AI/models/derod_checkpoint_large_real_data_test_v3.pth"
        # Load the model configuration
        model_path = str(model_path)

        if self.sam_type == 'vit_l':
            self.model_config = SamConfig.from_pretrained(f"facebook/sam-vit-large")
            self.processor = SamProcessor.from_pretrained(f"facebook/sam-vit-large")
        if self.sam_type == 'vit_b':
            self.model_config = SamConfig.from_pretrained(f"facebook/sam-vit-base")
            self.processor = SamProcessor.from_pretrained(f"facebook/sam-vit-base")
        if self.sam_type == 'vit_h':
            self.model_config = SamConfig.from_pretrained(f"facebook/sam-vit-huge")
            self.processor = SamProcessor.from_pretrained(f"facebook/sam-vit-huge")

        # Create an instance of the model architecture with the loaded configuration
        self.model = SamModel(config=self.model_config)

        # Update the model by loading the weights from saved file.
        self.model.load_state_dict(torch.load(model_path))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def load_model_sam2(self,model_path):

        self.sam2_model = build_sam2(self.sam2_cfg, self.sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.sam2_predictor.model.load_state_dict(torch.load(model_path))


    def run_rfi_model(self, pad_width=50, patch_run=False, sliding_patch=False, adding_patch=False, threshold=0.5, save=False):

        self.pad_width = pad_width

        print("SAMRFI Progress...")


        if not patch_run:

            pol_flags_list = []

            for baseline in tqdm(range(self.RadioRFI.rfi_antenna_data.shape[0])):

                flags = []

                for pol in range(self.RadioRFI.rfi_antenna_data.shape[1]):

                    data = self.RadioRFI.rfi_antenna_data[baseline,pol,:,:]

                    single_data = data/np.nanmedian(data)
                    
                    single_patch = Image.fromarray(single_data).convert("RGB")
                    bbox = get_bounding_box(single_data)

                    inputs = self.processor(single_patch, input_boxes=[[bbox]], return_tensors="pt")

                    # Move the input tensor to the GPU if it's not already there
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model.eval()

                    # forward pass
                    with torch.no_grad():
                        outputs = self.model(**inputs,multimask_output=False)

                        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
                        masks = masks[0].cpu().numpy().squeeze()

                    flags.append(masks)
                pol_flags = np.stack(flags)
                pol_flags_list.append(pol_flags)

            self.pol_flags_list = pol_flags_list
            baseline_flags = np.stack(pol_flags_list)
            
            self.flags = baseline_flags

            self.RadioRFI.update_flags(self.flags)

        elif patch_run:

            pol_flags_list = []
            pol_flags_prob_list = []


            for baseline in tqdm(range(self.RadioRFI.rfi_antenna_data.shape[0])):
                

                flags = []
                flags_prob = []

                for pol in range(self.RadioRFI.rfi_antenna_data.shape[1]):

                    data = self.RadioRFI.rfi_antenna_data[baseline,pol,:,:]

                    single_data = data/np.nanmedian(data)


                    if adding_patch:
                        patches, positions = extract_patches_with_context(single_data)
                        patches = [patches[i] for i in range(patches.shape[0])]
                    elif not sliding_patch:
                        patches, original_shape, padded_shape = create_patches(single_data)
                    elif sliding_patch:
                        patches, positions = extract_patches(single_data, window_size=256, overlap=128)
                        patches = [patches[i] for i in range(patches.shape[0])]

                    self.patches = patches

                    patch_flags = []
                    patch_flags_prob = []

                    for patch in patches:

                        single_patch = Image.fromarray(patch).convert("RGB")

                        bbox = get_bounding_box(patch)

                        inputs = self.processor(single_patch, input_boxes=[[bbox]], logits=True, return_tensors="pt")

                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        self.inputs = inputs
                        
                        self.model.eval()

                        with torch.no_grad():
                            outputs = self.model(**inputs, multimask_output=False)
                            
                            single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                            single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
                            single_patch_prediction = single_patch_prob > threshold

                            self.outputs = outputs

                        patch_flags.append(single_patch_prediction)
                        patch_flags_prob.append(single_patch_prob)

                    if adding_patch:
                        cropped_patches = crop_patches(np.stack(patch_flags))
                        master_flag = reconstruct_from_patches_adding(cropped_patches, positions, single_data.shape)

                        cropped_patches_prob = crop_patches(np.stack(patch_flags_prob))
                        master_flag_prob = reconstruct_from_patches_adding(cropped_patches_prob, positions, single_data.shape)
                    elif not sliding_patch:
                        master_flag = reconstruct_image(patch_flags, original_shape, padded_shape)
                        master_flag_prob = reconstruct_image(patch_flags_prob, original_shape, padded_shape)
                    elif sliding_patch:
                        master_flag = reconstruct_from_patches(np.stack(patch_flags), positions, single_data.shape, window_size=256, overlap=128)
                        master_flag_prob = reconstruct_from_patches(np.stack(patch_flags_prob), positions, single_data.shape, window_size=256, overlap=128)

                    flags.append(master_flag)
                    flags_prob.append(master_flag_prob)


                pol_flags = np.stack(flags)
                pol_flags_list.append(pol_flags)
                
                pol_flags_prob = np.stack(flags_prob)
                pol_flags_prob_list.append(pol_flags_prob)


            self.pol_flags_list = pol_flags_list
            baseline_flags = np.stack(pol_flags_list)

            self.flags = baseline_flags

            self.pol_flags_prob_list = pol_flags_prob_list
            baseline_flags_prob = np.stack(pol_flags_prob_list)

            self.flags_prob = baseline_flags_prob

            self.RadioRFI.update_flags(baseline_flags)
        
        if save:    
            np.save(f"{self.RadioRFI.directory}/flags.npy",baseline_flags)

    def run_model_sam2(self, threshold=0.95, patch_size=1024, num_points=256, point_threshold=1, reuse_logics=False, save=False):

        pol_flags_list = []
        pol_logits_list = []

        counter = 0

        master_logits = None

        for baseline in tqdm(range(self.RadioRFI.rfi_antenna_data.shape[0])):

            flags = []
            logitss = []

            for pol in range(self.RadioRFI.rfi_antenna_data.shape[1]):

                # Normalize the data before running the model.
                data = self.RadioRFI.rfi_antenna_data[baseline,pol,:,:]

                single_data = data/np.nanmedian(data)

                patches, original_shape, padded_shape = create_patches(single_data, patch_size=patch_size)

                if reuse_logics:
                    if counter > 0:
                        logits_reuse, original_shapel, padded_shapel = create_patches(master_logits, patch_size=patch_size)
                        logits_reuse = np.array(logits_reuse)

                self.patches = patches

                patch_flags = []
                patch_logits = []

                for idx, patch in enumerate(patches):

                    single_patch = np.stack([patch] * 3, axis=-1) #(3, 1024, 1024)

                    single_patch = single_patch.astype(np.float32)

                    bbox = np.array(get_bounding_box(patch))

                    # From https://www.datacamp.com/tutorial/sam2-fine-tuning

                    rows, cols = np.where(patch > point_threshold)
                    coords = np.stack((rows, cols), axis=-1)
                    np.random.shuffle(coords)
                    coords = coords[:num_points]
                    #coords_xy = coords[:, [1, 0]]

                    self.input_points = coords
                    self.point_labels = np.ones((self.input_points.shape[0],), dtype=int)

                    # fig, ax = plt.subplots(figsize=(10,5))
                    # ax.imshow(patch)
                    # ax.scatter(self.input_points[:, 1], self.input_points[:, 0], c='r', s=10)
                    # plt.show()

                    bounding_box = np.array(get_bounding_box(patch))

                    # bounding_box = [float(coord) for coord in bounding_box]
                    # bounding_box_tensor = torch.tensor([bounding_box], device=self.device).float().unsqueeze(0)

                    with torch.no_grad():

                        self.sam2_predictor.set_image(single_patch)

                        if reuse_logics:
                            if counter > 0:

                                logit_resue_tensor = torch.tensor(logits_reuse[idx,:,:])
                                logit_resue_tensor = logit_resue_tensor.unsqueeze(0)

                                masks, scores, logits = self.sam2_predictor.predict(
                                    point_coords=self.input_points,
                                    point_labels=self.point_labels,
                                    box=bounding_box,
                                    mask_input=logit_resue_tensor,
                                    multimask_output=False,
                                )
                            
                            else:
                                masks, scores, logits = self.sam2_predictor.predict(
                                    point_coords=self.input_points,
                                    point_labels=self.point_labels,
                                    box=bounding_box,
                                    multimask_output=False,
                                )

                        else:
                            masks, scores, logits = self.sam2_predictor.predict(
                                point_coords=self.input_points,
                                point_labels=self.point_labels,
                                box=bounding_box,
                                multimask_output=False,
                            )
                        
                    self.test_masks = masks

                    binary_mask = (masks[0] > threshold).astype(np.uint8)

                    patch_flags.append(binary_mask)
                    patch_logits.append(logits[0])
                    
                counter += 1

                master_flag = reconstruct_image(patch_flags, original_shape, padded_shape, patch_size=patch_size)
                master_logits = reconstruct_image(patch_logits, original_shape, padded_shape, patch_size=patch_size)

                flags.append(master_flag)
                logitss.append(master_logits)

            pol_flags = np.stack(flags)
            pol_flags_list.append(pol_flags)

            pol_logits = np.stack(logitss)
            pol_logits_list.append(pol_logits)

        baseline_flags = np.logical_not(np.stack(pol_flags_list))
        baseline_logits = np.stack(pol_logits_list)

        self.flags = baseline_flags
        self.logits = 1 - torch.sigmoid(torch.tensor(baseline_logits))

        self.RadioRFI.update_flags(baseline_flags)

        if save:
            np.save(f"{self.RadioRFI.directory}/sam2_flags.npy",baseline_flags)                

    def create_RGB_channels(self,zeroR=False,zeroG=False,zeroB=False):
        """
        Generate an RGB combined image from a spectrograph.

        Parameters:
            spectrograph (numpy.ndarray): The input spectrograph image.

        Returns:
            numpy.ndarray: The RGB combined image.

        """

        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(self.spectrograph, n_points, radius, method='uniform')
        edges = feature.canny(self.spectrograph, sigma=1)

        scaler = MinMaxScaler()

        # self.image_R = scaler.fit_transform(lbp)
        # self.image_G = scaler.fit_transform(self.spectrograph)
        # self.image_B = scaler.fit_transform(edges)
        
        # self.image_R = self.image_G/self.image_R
        # self.image_G = self.image_G*2
        # self.image_B = self.image_G*self.image_B

        self.image_G = self.spectrograph/255

        if zeroR:
            self.image_R = np.zeros((self.spectrograph.shape[0], self.spectrograph.shape[1]))
        if zeroG:
            self.image_G = np.zeros((self.spectrograph.shape[0], self.spectrograph.shape[1]))
        if zeroB:
            self.image_B = np.zeros((self.spectrograph.shape[0], self.spectrograph.shape[1]))

        self.test_image = np.dstack([self.image_R,self.image_G,self.image_B])