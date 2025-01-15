import os

import sys
sys.path.append('../../')

from samrfi import RFIModels
import numpy as np

from samrfi import RFIModels, RadioRFI

import torch

dir_path = '/home/gpuhost002/ddeal/RFI-AI/'

original_calib = '/home/gpuhost002/ddeal/RFI-AI/one_antenna_3C129_tfcrop.ms'
#original_calib = '/home/gpuhost002/ddeal/RFI-AI/original_calib/calib_phase_rflag.ms'

datarfi_3C129 = RadioRFI(vis=original_calib, dir_path=dir_path)
datarfi_3C129.load(mode='DATA', ant_i=1) # using first two antennas

#datarfi_3C129.load(mode='FLAG', ant_i=1) # using first two antennas
#datarfi_3C129.flags = np.abs(datarfi_3C129.ms_flags)

datarfi_3C129.rfi_antenna_data = datarfi_3C129.rfi_antenna_data[0:2]

sam2_path = "/home/gpuhost002/ddeal/RFI-AI/sam2/notebooks/"

model_path = "/home/gpuhost002/ddeal/RFI-AI/samrfi_data/models/model_tfcrop_twice_patch_size-256_sam2-large_epochs60_20250110_070533.pth"
model = RFIModels(radiorfi_instance=datarfi_3C129, device='cuda',)

os.chdir('/home/gpuhost002/ddeal/RFI-AI/sam2/notebooks')

model.sam2_ckpt = "../checkpoints/sam2.1_hiera_large.pt"
model.sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

model.device = 'cuda'

model.load_model_sam2(model_path)
model.run_model_sam2(patch_size=256, num_points=1024*6, point_threshold=0.5, threshold=.9, multimask_output=False, reuse_logits=True)

logits = model.logits.detach().cpu().numpy()

logits_thrs = logits > 0.95

datarfi_3C129.flags = logits_thrs

datarfi_3C129.radio_metrics.test_realdata()