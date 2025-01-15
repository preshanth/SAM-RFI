import time
from casatasks import flagmanager, flagdata

# Define the measurement set file
ms_file = '/home/gpuhost002/ddeal/RFI-AI/one_antenna_3C129_tfcrop.ms'

flagdata(vis=ms_file, mode='unflag', flagbackup=False)
print('done')
# Start timing
start_time = time.time()

mode = 'tfcrop'
# Run the flagdata tasks, twice
flagdata(vis=ms_file, mode=mode, datacolumn='DATA', timecutoff=5, freqcutoff=5)
print('done')
flagdata(vis=ms_file, mode=mode, datacolumn='DATA', timecutoff=5, freqcutoff=5)
print('done')

# End timing
end_time = time.time()
duration = end_time - start_time

ms_name = ms_file.split('/')[-1].replace('.ms', '')
output_filename = f'/home/gpuhost002/ddeal/RFI-AI/SAM-RFI/notebooks/test_notebooks/flagdata_duration_{ms_name}_{mode}.txt'
with open(output_filename, 'w') as f:
    f.write(f"Flagdata operation on {ms_name}.ms took {duration:.2f} seconds on {mode}.\n")