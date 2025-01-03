from casatasks import flagmanager, flagdata

flagdata(vis='/home/gpuhost002/ddeal/RFI-AI/original_calib/calib_phase_tfcrop.ms', mode='tfcrop', datacolumn='DATA', writeflags=True, flagbackup=True)