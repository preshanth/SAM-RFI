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

from .radiorfi import RadioRFI
from .utilities import *

class RFIModels:

    def __init__(self, sam_checkpoint, sam_type, radiorfi_instance, device='cuda',):

        self.RadioRFI = radiorfi_instance
        sam_checkpoint = str(sam_checkpoint)
        self.sam_type = sam_type

        sam = sam_model_registry[self.sam_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.predictor = SamPredictor(sam)

        print(self.RadioRFI.rfi_antenna_data.shape)

        self.RadioRFI.update_flags('Flags updated')

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


    def run_rfi_model(self, pad_width=50, patch_run=False, threshold=0.5, save=False):

        self.pad_width = pad_width

        print("SAMRFI Progress...")


        if not patch_run:

            pol_flags_list = []

            for baseline in tqdm(range(self.RadioRFI.rfi_antenna_data.shape[0])):

                flags = []

                for pol in range(self.RadioRFI.rfi_antenna_data.shape[1]):

                    data = self.RadioRFI.rfi_antenna_data[baseline,pol,:,:]

                    single_data = data/np.median(data)
                    
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

        elif patch_run:

            pol_flags_list = []

            for baseline in tqdm(range(self.RadioRFI.rfi_antenna_data.shape[0])):

                flags = []

                for pol in range(self.RadioRFI.rfi_antenna_data.shape[1]):

                    data = self.RadioRFI.rfi_antenna_data[baseline,pol,:,:]

                    single_data = data/np.median(data)

                    patches, original_shape, padded_shape = create_patches(single_data)

                    patch_flags = []

                    for patch in patches:

                        single_patch = Image.fromarray(patch).convert("RGB")

                        bbox = get_bounding_box(patch)

                        inputs = self.processor(single_patch, input_boxes=[[bbox]], return_tensors="pt")

                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        self.model.eval()

                        with torch.no_grad():
                            outputs = self.model(**inputs,multimask_output=False)

                            single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                            single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
                            single_patch_prediction = (single_patch_prob > threshold)

                        patch_flags.append(single_patch_prediction > 0)

                    master_flag = reconstruct_image(patch_flags, original_shape, padded_shape)

                    flags.append(master_flag)

                pol_flags = np.stack(flags)
                pol_flags_list.append(pol_flags)

            self.pol_flags_list = pol_flags_list
            baseline_flags = np.stack(pol_flags_list)

            self.flags = baseline_flags

        self.RadioRFI.update_flags(self.flags)
        
        if save:    
            np.save(f"{self.RadioRFI.directory}/flags.npy",baseline_flags)


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