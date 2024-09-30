import sys
sys.path.append('../')

from samrfi import RadioRFI, RFITraining
import numpy as np

dir_path = '/home/gpuhost001/ddeal/RFI-AI/'

original_calib = '/home/gpuhost001/ddeal/RFI-AI/original_calib/calib_phase.ms'

datarfi_original = RadioRFI(vis=original_calib, dir_path=dir_path)
datarfi_original.load(mode='DATA', ant_i=1) # using first antenna data

reflected_part = np.flip(datarfi_original.rfi_antenna_data, axis=-1)  # Alternatively, use arr[..., ::-1]
# Step 2: Concatenate along the last axis
new_arr = np.concatenate((datarfi_original.rfi_antenna_data, reflected_part), axis=-1)

datarfi_original.rfi_antenna_data = new_arr
print('WORKING')

calib_phase_ori = RFITraining(datarfi_original, device='cuda')
calib_phase_ori.train(num_epochs=70, stretch='SQRT', num_patches=600, flag_sigma=5, patch_size=256, plot=False)

np.save('/home/gpuhost001/ddeal/RFI-AI/samrfi_data/models/sqrt_sigma5_29_1.npy', calib_phase_ori.ave_meanloss)

calib_phase_ori = RFITraining(datarfi_original, device='cuda')
calib_phase_ori.train(num_epochs=70, stretch='LOG10', num_patches=600, flag_sigma=8, patch_size=256, plot=False)

np.save('/home/gpuhost001/ddeal/RFI-AI/samrfi_data/models/log10_sigma5_29_1.npy', calib_phase_ori.ave_meanloss)