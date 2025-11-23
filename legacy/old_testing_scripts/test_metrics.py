import sys
sys.path.append('../')

from samrfi import SyntheticRFI, RFIModels

dir_path = '/home/gpuhost001/ddeal/RFI-AI/'

synthetic_rfi = SyntheticRFI(min_freq=2000, max_freq=4000, min_time=0, max_time=500, points_freq=2000, points_time=500,)

synthetic_rfi.generate_waterfall(pers_freq_gauss=2, pers_time_gauss=1, inter_freq_square=2, pers_freq_square=3, pers_time_square=2, 
                                noise=10, mean=5, mean_rfi=50, std_rfi=10,edge_buffer=50)
sam_checkpoint = "/home/gpuhost001/ddeal/RFI-AI/models/sam_vit_h_4b8939.pth"
sam_type = "vit_h"

model_path = "/home/gpuhost001/ddeal/RFI-AI/models/derod_checkpoint_huge_calib_phase_patch_60_epoch_log10_median_patches_flip_synthethic_only_plus_40.pth"
model = RFIModels(sam_checkpoint, sam_type, radiorfi_instance=synthetic_rfi,  device='cuda',)
model.load_model(model_path)
model.run_rfi_model(patch_run=True)

print(synthetic_rfi.synthetic_metrics.calculate_metrics(threshold=1))
synthetic_rfi.plotter.plot(mode='FLAG', baseline=0, polarization=0) 