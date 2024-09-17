from samrfi import RadioRFI, RFITraining 

ms_path = '/home/gpuhost001/ddeal/RFI-AI/one_antenna_3C219_sqrt.ms'
dir_path = '/home/gpuhost001/ddeal/RFI-AI/'

datarfi = RadioRFI(vis=ms_path, dir_path=dir_path)
datarfi.load(mode='DATA', ant_i=1)

datarfi.rfi_antenna_data = datarfi.rfi_antenna_data[0:2]

print(datarfi.rfi_antenna_data.shape)

new_model = RFITraining(datarfi, device='cuda')
new_model.train(num_epochs=2, stretch='SQRT', num_patches=50)