import time
from casatasks import flagmanager, flagdata

# Define the measurement set file
ms_file = '/home/gpuhost002/ddeal/RFI-AI/one_antenna_3C129_tfcrop.ms'

flagdata(vis=ms_file, mode='unflag')
# Start timing
start_time = time.time()

mode = 'tfcrop'
# Run the flagdata task
flagdata(vis=ms_file, mode=mode, datacolumn='DATA', writeflags=True, flagbackup=True)

# End timing
end_time = time.time()
duration = end_time - start_time

# Extract the measurement set name from the path
ms_name = ms_file.split('/')[-1].replace('.ms', '')

# Write the duration to a text file
output_filename = f'/home/gpuhost002/ddeal/RFI-AI/SAM-RFI/notebooks/test_notebooks/flagdata_duration_{ms_name}_{mode}.txt'
with open(output_filename, 'w') as f:
    f.write(f"Flagdata operation on {ms_name}.ms took {duration:.2f} seconds on {mode}.\n")