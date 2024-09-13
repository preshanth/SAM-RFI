from samrfi import RadioRFI
from samrfi import Plotter
from samrfi import RFIModels

ms_path = '/home/gpuhost001/ddeal/RFI-AI/one_antenna_3C219_sqrt.ms'

datarfi = RadioRFI(vis=ms_path)
datarfi.load(mode='DATA', ant_i=1)
print('Done!')

sam_checkpoint = "/home/gpuhost001/ddeal/RFI-AI/models/sam_vit_h_4b8939.pth"
sam_type = "vit_h"

model_path = "/home/gpuhost001/ddeal/RFI-AI/models/derod_checkpoint_huge_calib_phase_patch_epoch40_sigma5_sqrt_custom_perpatch.pth"
model = RFIModels(sam_checkpoint, sam_type, radiorfi_instance=datarfi,  device='cuda',)
model.load_model(model_path)

print(datarfi.flags)
model.run_rfi_model(patch_run=True)

print(datarfi.flags)

# pl = Plotter(datarfi.rfi_antenna_data, model.flags)
# pl.plot(mode='DATA', baseline=0, polarization=0)

# syn = SyntheticRFI()