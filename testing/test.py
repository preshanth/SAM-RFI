import sys
sys.path.append('../')

from samrfi import RadioRFI, RFIModels
import numpy as np
ms_path = '/home/gpuhost001/ddeal/RFI-AI/one_antenna_3C219_sqrt.ms'
dir_path = '/home/gpuhost001/ddeal/RFI-AI/'

datarfi = RadioRFI(vis=ms_path, dir_path=dir_path)
datarfi.load(mode='DATA', ant_i=1)

sam_checkpoint = "/home/gpuhost001/ddeal/RFI-AI/models/sam_vit_h_4b8939.pth"
sam_type = "vit_h"

model_path = "/home/gpuhost001/ddeal/RFI-AI/models/derod_checkpoint_huge_calib_phase_patch_epoch40_sigma5_sqrt_custom_perpatch.pth"
model = RFIModels(sam_checkpoint, sam_type, radiorfi_instance=datarfi, device='cuda',)
model.load_model(model_path)
model.run_rfi_model(patch_run=False)

# print(datarfi.flags.shape)
# # print(datarfi.rfi_antenna_data.shape)

# # Calculate statistics
# data = datarfi.rfi_antenna_data

# max_value = np.max(data)
# min_value = np.min(data)
# mean_value = np.mean(data)
# median_value = np.median(data)
# std_dev = np.std(data)
# variance = np.var(data)
# percentile_25 = np.percentile(data, 25)
# percentile_75 = np.percentile(data, 75)
# skewness = np.mean((data - mean_value)**3) / std_dev**3
# kurtosis = np.mean((data - mean_value)**4) / std_dev**4
# rms = np.sqrt(np.mean(data**2))
# snr = mean_value / std_dev

# # Print statistics
# print("RFI Statistics")
# print(f"Max Value: {max_value}")
# print(f"Min Value: {min_value}")
# print(f"Mean Value: {mean_value}")
# print(f"Median Value: {median_value}")
# print(f"Standard Deviation: {std_dev}")
# print(f"Variance: {variance}")
# print(f"25th Percentile: {percentile_25}")
# print(f"75th Percentile: {percentile_75}")
# print(f"Skewness: {skewness}")
# print(f"Kurtosis: {kurtosis}")
# print(f"RMS: {rms}")
# print(f"SNR: {snr}")

datarfi.plotter.plot(mode='FLAG', baseline=0, polarization=0)

datarfi.radio_metrics.calculate_metrics(datarfi.flags)