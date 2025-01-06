import sys
sys.path.append('../')

import numpy as np

from samrfi import RadioRFI, RFIDataset, RFITraining

dir_path = '/home/gpuhost002/ddeal/RFI-AI/'

original_calib = '/home/gpuhost002/ddeal/RFI-AI/original_calib/calib_phase_tfcrop.ms'

datarfi_original = RadioRFI(vis=original_calib, dir_path=dir_path)
datarfi_original.load(mode='DATA', ant_i=2) # using first two antennas

reflected_part = np.flip(datarfi_original.rfi_antenna_data, axis=-1)  # Alternatively, use arr[..., ::-1]
# Step 2: Concatenate along the last axis
new_arr = np.concatenate((datarfi_original.rfi_antenna_data, reflected_part), axis=-1)
datarfi_original.rfi_antenna_data = new_arr

datarfi_original.load(ant_i=2, mode='FLAG')

datarfi_original.flags = np.abs(datarfi_original.ms_flags)

reflected_part = np.flip(datarfi_original.flags, axis=-1)  # Alternatively, use arr[..., ::-1]
# Step 2: Concatenate along the last axis
new_arr = np.concatenate((datarfi_original.flags, reflected_part), axis=-1)
datarfi_original.flags = new_arr

rfi_dataset = RFIDataset(datarfi_original, dir_path=dir_path)
rfi_dataset.create_dataset(num_patches=800, patch_size=256, apply_stretching=False, custom_flag=True)

rfi_dataset.save_dataset()

new_model = RFITraining(rfi_dataset, device='cuda', dir_path=dir_path)
new_model.train(num_epochs=120, batch_size=4)