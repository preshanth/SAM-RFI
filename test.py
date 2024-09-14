from samrfi import RadioRFI, Plotter, RFIModels, SyntheticRFI

ms_path = '/home/gpuhost001/ddeal/RFI-AI/one_antenna_3C219_sqrt.ms'
dir_path = '/home/gpuhost001/ddeal/RFI-AI/'

datarfi = RadioRFI(vis=ms_path, dir_path=dir_path)
datarfi.load(mode='DATA', ant_i=1)

sam_checkpoint = "/home/gpuhost001/ddeal/RFI-AI/models/sam_vit_h_4b8939.pth"
sam_type = "vit_h"

model_path = "/home/gpuhost001/ddeal/RFI-AI/models/derod_checkpoint_huge_calib_phase_patch_epoch40_sigma5_sqrt_custom_perpatch.pth"
model = RFIModels(sam_checkpoint, sam_type, radiorfi_instance=datarfi,  device='cuda',)
model.load_model(model_path)
model.run_rfi_model(patch_run=False)

print(datarfi.flags.shape)
print(datarfi.rfi_antenna_data.shape)

datarfi.plotter.plot(mode='FLAG', baseline=0, polarization=0)