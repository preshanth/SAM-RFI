import sys
sys.path.append('../')

from samrfi import RadioRFI, RFITraining
import numpy as np

dir_path = '/home/gpuhost001/ddeal/RFI-AI/'

original_calib = '/home/gpuhost001/ddeal/RFI-AI/original_calib/calib_phase.ms'

datarfi_original = RadioRFI(vis=original_calib, dir_path=dir_path)
datarfi_original.load(mode='DATA', ant_i=1) # using first antenna data
# datarfi_original.plotter.plot(baseline=0, polarization=0)

calib_phase_ori = RFITraining(datarfi_original, device='cuda')
calib_phase_ori.train(num_epochs=40, stretch='SQRT', num_patches=500, flag_sigma=5)

np.save('/home/gpuhost001/ddeal/RFI-AI/samrfi_data/models/calib_phase_ori_epoch_40_numpatch_500_sigma_5.npy', calib_phase_ori.ave_meanloss)