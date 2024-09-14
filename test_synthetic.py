from samrfi import SyntheticRFI

dir_path = '/home/gpuhost001/ddeal/RFI-AI/'

synthetic_rfi = SyntheticRFI()

synthetic_rfi.generate_rfi_waterfall(min_freq=2000, max_freq=5000, time_int=2000, points=3000, 
                                        num_persistent=6, num_intermittent=6, noise=10, edge_buffer=100)

print(synthetic_rfi.spectrograph.shape)

print(synthetic_rfi.rfi_antenna_data.shape)
synthetic_rfi.plotter.plot(mode='DATA', baseline=0, polarization=0)