from samrfi import SyntheticRFI

dir_path = '/home/gpuhost001/ddeal/RFI-AI/'

synthetic_rfi = SyntheticRFI(min_freq=2000, max_freq=4000, min_time=0, max_time=2000, points_freq=2000, points_time=500,)

synthetic_rfi.generate_waterfall(pers_freq_gauss=5, pers_time_gauss=1, inter_freq_square=5, pers_freq_square=10, pers_time_square=4, noise=10, mean=5, edge_buffer=50)

print(synthetic_rfi.spectrograph.shape)

print(synthetic_rfi.rfi_antenna_data.shape)

print(synthetic_rfi.rfi_table)
# synthetic_rfi.plotter.plot(mode='DATA', baseline=0, polarization=0)