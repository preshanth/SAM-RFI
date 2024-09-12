import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skimage import feature
from skimage.feature import peak_local_max, local_binary_pattern
from skimage.transform import resize
from casatools import table
from astropy.visualization import ZScaleInterval, ImageNormalize
from transformers import SamModel, SamConfig, SamProcessor
import torch
from PIL import Image
import time
import os
from tqdm import tqdm
from scipy import stats


### TODOList
# add create residuals function

class SamRFI:
    def __init__(self, vis=False, time_int=2000, points=2000, device='cuda', dir_path=None):
        self.rfi_table = pd.DataFrame(columns=['rfi_type', 'amplitude', 'center_freq', 'bandwidth', 'duty_cycle', 'time_period', 'time_offset'])
        
        self.spectrograph = np.zeros((time_int, points))
        self.flags = None
        self.noise = None
        self.frequencies = None
        self.time_int = time_int
        self.points = points
        self.pad_width = None
        self.min_freq = None
        self.test_image = None
        self.mean = None


        # # sam_checkpoint = '/home/galagabits/astro/sam_vit_h_4b8939.pth'
        # # sam_checkpoint = "/home/gpuhost002/ddeal/RFI-AI/models/sam_vit_l_0b3195.pth"
        # sam_checkpoint = "/home/gpuhost002/ddeal/RFI-AI/models/sam_vit_l_0b3195.pth"
        # # sam_checkpoint = '/home/gpuhost002/ddeal/RFI-AI/models/derod_checkpoint_large.pth'
        # model_type = "vit_l"

        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)

        # self.mask_generator = SamAutomaticMaskGenerator(sam)
        # self.predictor = SamPredictor(sam)

        sam_checkpoint = '/home/galagabits/astro/sam_vit_h_4b8939.pth'
        sam_checkpoint = "/Users/galagabits/Developer/RFI-AI/model-checkpoints/sam_vit_h_4b8939.pth"
        sam_checkpoint = "/home/nemo2/derod/RFI-AI/sam_models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        # sam_checkpoint = '/home/galagabits/astro/sam_vit_h_4b8939.pth'
        # sam_checkpoint = "/home/gpuhost002/ddeal/RFI-AI/models/sam_vit_l_0b3195.pth"
        # sam_checkpoint = "/home/gpuhost002/ddeal/RFI-AI/models/sam_vit_l_0b3195.pth"
        # sam_checkpoint = '/home/gpuhost002/ddeal/RFI-AI/models/derod_checkpoint_large.pth'
        # model_type = "vit_l"


        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.predictor = SamPredictor(sam)


        ################
        #Directory logic
        ################

        if dir_path:
            if dir_path.endswith('/'):
                dir_path = dir_path[:-1]

            current_directory = str(dir_path)
        else:
            current_directory = os.getcwd()

        new_directory = os.path.join(current_directory, 'samrfi_data')

        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        self.directory = new_directory
        

        if vis:
            # Path to ms
            self.vis = str(vis)
            
            # Number of antenna 
            tb_antenna = table()
            tb_antenna.open(self.vis+'/ANTENNA')
            self.num_antennas = tb_antenna.nrows()
            tb_antenna.close()

            # Number of spectral windows
            tb_spw = table()
            tb_spw.open(self.vis+'/SPECTRAL_WINDOW')
            self.num_spw = tb_spw.nrows()
            self.channels_per_spw = tb_spw.getcol('NUM_CHAN')
            tb_spw.close()

            # Tables
            self.tb = table()
            self.tb.open(vis, nomodify=False)
        else:
            self.vis = None
        


    def load(self, vis=None, mode='DATA',ant_i=2):

        
        if not self.vis:
            self.vis = str(vis)

        # combined_data = np.zeros([4,1024,140],dtype='complex128')
        subtable = self.tb.query(f'DATA_DESC_ID=={0} && ANTENNA1=={0} && ANTENNA2=={1}')
        self.time_tb = len(subtable.getcol('TIME'))

        channels_per_spw_list = self.channels_per_spw

        spw_array = list(range(self.num_spw))
        channels_per_spw_array = self.channels_per_spw

        same_spw_array = []
        same_channels_per_spw_array = []

        for spw, spw_numchan in zip(spw_array, channels_per_spw_array):
            if spw_numchan == channels_per_spw_array[0]:
                same_spw_array.append(spw)
                same_channels_per_spw_array.append(spw_numchan)

        init_chan = same_channels_per_spw_array[0]
        same_num_spw = len(same_spw_array)

        rfi_list = []

        antenna_baseline_map = []


        self.num_antennas_i = ant_i
        for i in tqdm(range(self.num_antennas_i)):
       # for i in tqdm(range(self.num_antennas)):
            for j in tqdm(range(i + 1, self.num_antennas)):
                combined_data = np.zeros([4,same_num_spw*init_chan,self.time_tb],dtype='complex128')

                for spw_spec, spw, num_chan in zip(same_spw_array, range(same_num_spw), same_channels_per_spw_array):
                    # input field number as well
                    subtable = self.tb.query(f'DATA_DESC_ID=={spw_spec} && ANTENNA1=={i} && ANTENNA2=={j}')
                    combined_data[:,spw*init_chan:(spw+1)*init_chan,:] += subtable.getcol(mode)
                rfi_list.append(combined_data)

                antenna_baseline_map.append((i,j))

        self.antenna_baseline_map = antenna_baseline_map
        self.spw = same_spw_array
        self.channels_per_spw = same_channels_per_spw_array
        
        if mode == 'DATA':
            self.rfi_antenna_data_complex = np.stack(rfi_list)
            self.rfi_antenna_data = np.abs(self.rfi_antenna_data_complex)
        
        if mode == 'FLAG':
            self.ms_flags = np.stack(rfi_list)

    def plot_waterfall(self, baseline=0, polarization=0,):
        fig, ax = plt.subplots(figsize=(12,10), dpi=200)

        data = np.abs(self.rfi_antenna_data[0,0,:,:])

        norm = ImageNormalize(data, interval=ZScaleInterval())
        im = plt.imshow(data, interpolation='none', norm=norm, aspect='auto')

        plt.colorbar(im)
        plt.show()

    def baseline_profile(self):
        """
        Generates a baseline profile using a normal distribution.

        Returns:
            numpy.ndarray: The baseline profile.
        """
        return np.random.normal(self.mean, self.noise, self.temp_spectrograph.shape[1])

    # Generate synthetic RFI signals
    def generate_rfi(self, amplitude, center_freq, bandwidth):
        """
        Generates a radio frequency interference (RFI) signal.

        Parameters:
            amplitude (float): The amplitude of the RFI signal.
            center_freq (float): The center frequency of the RFI signal.
            bandwidth (float): The bandwidth of the RFI signal.

        Returns:
            numpy.ndarray: An array representing the RFI signal.
        """
        return amplitude * np.exp(-((self.frequencies - center_freq) ** 2) / (2 * (bandwidth ** 2)))

    
    def square_function(self, amplitude, center_freq ,bandwidth):
        return amplitude * np.where(np.abs(self.frequencies - center_freq) <= bandwidth/2, 1, 0)


    def add_rfi(self, data, amplitude, center_freq, bandwidth):
        return data+self.generate_rfi(amplitude, center_freq, bandwidth)


    def add_square_rfi(self, data, amplitude, center_freq, bandwidth):
        return data+self.square_function(amplitude, center_freq, bandwidth)

    def add_rifi_spectrograph(self, amplitude, center_freq, bandwidth, horizontal=False, table=False):

        ## note to self, make sure to add time to the table
        if horizontal:
            self.temp_spectrograph = np.apply_along_axis(self.add_rfi, axis=0, arr=self.temp_spectrograph, amplitude=amplitude, center_freq=center_freq, bandwidth=bandwidth)
            new_row = pd.DataFrame({'rfi_type':['persistent'], 'amplitude': [amplitude], 'center_freq': [center_freq], 'bandwidth': [bandwidth], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        else:
            self.temp_spectrograph = np.apply_along_axis(self.add_rfi, axis=1, arr=self.temp_spectrograph, amplitude=amplitude, center_freq=center_freq, bandwidth=bandwidth)
            new_row = pd.DataFrame({'rfi_type':['persistent'], 'amplitude': [amplitude], 'center_freq': [center_freq], 'bandwidth': [bandwidth], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        
        if table:
            self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)

    
    def add_square_rifi_spectrograph(self, amplitude, center_freq, bandwidth, horizontal=False, table=False):

        ## note to self, make sure to add time to the table
        if horizontal:
            self.temp_spectrograph = np.apply_along_axis(self.add_square_rfi, axis=0, arr=self.temp_spectrograph, amplitude=amplitude, center_freq=center_freq, bandwidth=bandwidth)
            new_row = pd.DataFrame({'rfi_type':['persistent_sq'], 'amplitude': [amplitude], 'center_freq': [center_freq], 'bandwidth': [bandwidth], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        else:
            self.temp_spectrograph = np.apply_along_axis(self.add_square_rfi, axis=1, arr=self.temp_spectrograph, amplitude=amplitude, center_freq=center_freq, bandwidth=bandwidth)
            new_row = pd.DataFrame({'rfi_type':['persistent_sq'], 'amplitude': [amplitude], 'center_freq': [center_freq], 'bandwidth': [bandwidth], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        
        if table:
            self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)


    def create_spectrograph(self,):
        temporal_baseline = []
        for i in range(self.temp_spectrograph.shape[0]):
            temporal_baseline.append(self.baseline_profile())

        spectrograph = np.vstack(temporal_baseline)

        return spectrograph
        
    def add_spectrographs(self, spectrograph1, spectrograph2):
        return spectrograph1 + spectrograph2

    def create_RGB_channels(self,zeroR=False,zeroG=False,zeroB=False):
        """
        Generate an RGB combined image from a spectrograph.

        Parameters:
            spectrograph (numpy.ndarray): The input spectrograph image.

        Returns:
            numpy.ndarray: The RGB combined image.

        """

        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(self.spectrograph, n_points, radius, method='uniform')
        edges = feature.canny(self.spectrograph, sigma=1)

        scaler = MinMaxScaler()

        # self.image_R = scaler.fit_transform(lbp)
        # self.image_G = scaler.fit_transform(self.spectrograph)
        # self.image_B = scaler.fit_transform(edges)
        
        # self.image_R = self.image_G/self.image_R
        # self.image_G = self.image_G*2
        # self.image_B = self.image_G*self.image_B

        self.image_G = self.spectrograph/255

        if zeroR:
            self.image_R = np.zeros((self.spectrograph.shape[0], self.spectrograph.shape[1]))
        if zeroG:
            self.image_G = np.zeros((self.spectrograph.shape[0], self.spectrograph.shape[1]))
        if zeroB:
            self.image_B = np.zeros((self.spectrograph.shape[0], self.spectrograph.shape[1]))

        self.test_image = np.dstack([self.image_R,self.image_G,self.image_B])


    def mask_generator_timed(self):
        """
        Generates masks for the given image and measures the time taken.

        Parameters:
            image: The input image for which masks need to be generated.

        Returns:
            The elapsed time in seconds taken to generate the masks.
        """
        start = time.time()

        masks = self.mask_generator.generate(self.test_image)

        elapsed_time = time.time() - start

        print(f'Time: {elapsed_time} sec')

        return elapsed_time

    def create_model_spectrograph(self,):
        """
        Create a model spectrograph based on the given table of RFI parameters.

        Parameters:
            table (pandas.DataFrame): A table containing RFI parameters such as amplitude, center frequency, and bandwidth.
            frequencies (numpy.ndarray): An array of frequencies.
            time_int (int): The number of time intervals.
            points (int): The number of points in the spectrograph.

        Returns:
            numpy.ndarray: The model spectrograph.

        """

        self.temp_spectrograph = np.zeros((self.time_int, self.points))
        for index, row in self.rfi_table.iterrows():
            if row['rfi_type'] == 'persistent':
                self.add_rifi_spectrograph(row['amplitude'], 
                                                        row['center_freq'],  row['bandwidth'])
            if row['rfi_type'] == 'intermittent':
                self.intermittent_rfi(row['amplitude'], row['center_freq'],
                                                row['bandwidth'], row['time_period'], row['duty_cycle'], row['time_offset'])

            if row['rfi_type'] == 'persistent_sq':
                self.add_square_rifi_spectrograph(row['amplitude'], 
                                                        row['center_freq'],  row['bandwidth'])

            
        self.model_spectrograph = self.temp_spectrograph

    def intermittent_rfi(self, amplitude, center_freq, bandwidth, time_period, duty_cycle, time_offset=0, table=False):
        # Generated by Copilot
        
        """
        Add intermittent RFI to the spectrograph in two channels with a specified frequency offset and time period.
        
        Parameters:
        - spectrograph: The input spectrograph to which RFI will be added.
        - frequencies: Array of frequency values.
        - amplitude: Amplitude of the RFI signal.
        - center_freq: Center frequency of the first RFI channel.
        - bandwidth: Bandwidth of the RFI signal.
        - time_period: Time period for the intermittent RFI.
        - duty_cycle: Fraction of the time period during which the RFI is active.
        - time_offset: Offset in time for the intermittent RFI (default: 0).
        
        Returns:
        - modified_spectrograph: The spectrograph with added intermittent RFI.
        """
        time_int, points = self.temp_spectrograph.shape
        modified_spectrograph = self.temp_spectrograph
        
        # Create the time mask for intermittent RFI
        time_mask = np.zeros(time_int)
        period_indices = np.arange(time_offset, time_int, time_period)
        for start_idx in period_indices:
            end_idx = min(start_idx + int(time_period * duty_cycle), time_int)
            time_mask[int(start_idx):int(end_idx)] = 1
        
        # Generate the RFI signals for both channels
        rfi_signal_1 = self.generate_rfi(amplitude, center_freq, bandwidth)
        
        # Add the intermittent RFI to the spectrograph
        for t in range(time_int):
            if time_mask[t] == 1:
                modified_spectrograph[t, :] += rfi_signal_1
        
        # Update the RFI table
        if table:
            new_rows = pd.DataFrame({
                'rfi_type': ['intermittent'],
                'amplitude': [amplitude],
                'center_freq': [center_freq],
                'bandwidth': [bandwidth],
                'duty_cycle': [duty_cycle],
                'time_period': [time_period],
                'time_offset': [time_offset],
            })
            self.rfi_table = pd.concat([self.rfi_table, new_rows], ignore_index=True)
        
        self.temp_spectrograph = modified_spectrograph

    def generate_rfi_waterfall(self, min_freq, max_freq, time_int, points, num_persistent=6, num_intermittent=2, noise=10, mean=5, edge_buffer=50):
        """
        Generate a waterfall plot with simulated radio frequency interference (RFI).

        Parameters:
        - min_freq (float): The minimum frequency of the waterfall plot.
        - max_freq (float): The maximum frequency of the waterfall plot.
        - time_int (int): The time interval of the waterfall plot.
        - points (int): The number of frequency points in the waterfall plot.
        - num_persistent (int): The number of persistent RFI signals to add to the waterfall plot. Default is 6.
        - num_intermittent (int): The number of intermittent RFI signals to add to the waterfall plot. Default is 2.
        - noise (float): The noise level of the waterfall plot. Default is 10.
        - pad_width (int): The width of the padding to add around the waterfall plot. Default is 200.
        - edge_buffer (int): The buffer size around the edges of the waterfall plot. Default is 50.

        Returns:
        - modified_test_spectrograph (numpy.ndarray): The generated waterfall plot with simulated RFI.
        """
        self.frequencies = np.linspace(min_freq, max_freq, points)
        self.time_int = time_int
        self.points = points
        self.edge_buffer = edge_buffer
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.noise = noise
        self.mean = mean

        # base_spectrograph = create_spectrograph(frequencies, 5, 1, time_int, points)

        self.temp_spectrograph = np.zeros((self.time_int, self.points))

        # self.add_rifi_spectrograph(amplitude=np.random.normal(50,10), 
        #                                                 center_freq=np.random.uniform(min_freq+edge_buffer, max_freq-edge_buffer), bandwidth=np.abs(np.random.normal(50, 3)), table=True)
        
        # self.add_rifi_spectrograph(amplitude=np.random.normal(60,10), 
        #                                                 center_freq=np.random.uniform(min_freq+edge_buffer, max_freq-edge_buffer), bandwidth=np.abs(np.random.normal(5, 1)), table=True, horizontal=True)
        for i in np.arange(num_persistent):
            self.add_rifi_spectrograph(amplitude=np.random.normal(50,10), 
                                                        center_freq=np.random.uniform(min_freq+edge_buffer, max_freq-edge_buffer), bandwidth=np.abs(np.random.normal(11, 5)), table=True)
            
        for i in np.arange(num_intermittent):
            self.intermittent_rfi(amplitude=np.random.normal(50,10), center_freq=np.random.uniform(min_freq+edge_buffer, max_freq-edge_buffer),
                                                bandwidth=np.abs(np.random.normal(50, 10)), time_period=np.random.randint(1,500), duty_cycle=np.random.uniform(0,1), time_offset=np.random.randint(1, 100), table=True)

        self.spectrograph = self.temp_spectrograph + self.create_spectrograph()

    def pad_spectrograph(self,pad_width=50):

        self.pad_width = pad_width
        self.spectrograph = np.pad(self.spectrograph, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=40)

    def rm_largest_mask(self, masks):
        max_area = max(mask['area'] for mask in masks)

        new_masks = []
        for mask in masks:
            if mask['area'] != max_area:
                new_masks.append(mask)
        return new_masks

    def create_flags(self, masks):

        combined_segmentation = np.sum(np.stack([mask['segmentation'] for mask in masks]), axis=0)
        combined_segmentation = (np.clip(combined_segmentation, 0,1).astype(bool))

        return combined_segmentation

    def find_spectrograph_peaks(self, min_distance=10, threshold_abs=30):
        """
        Find peaks in the spectrograph image.

        Parameters:
            spectrograph (numpy.ndarray): The input spectrograph image.
            min_distance (int): The minimum distance between peaks. Default is 10.
            threshold_abs (int): The minimum intensity value for peaks. Default is 25.

        Returns:
            numpy.ndarray: An array of peak coordinates.
        """
        max_peaks = peak_local_max(self.spectrograph, min_distance=min_distance, threshold_abs=threshold_abs)
        self.max_peaks = max_peaks, np.ones(len(max_peaks))

    def get_bounding_box(self, ground_truth_map):
        # get bounding box from mask
        ## https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox

    #########################
    # Preprocessing
    #########################

    def create_patches(self, image, patch_size=256):
        # ChatGPT assisted with this function
        # Get image dimensions
        rows, cols = image.shape
        
        # Calculate padding size
        pad_rows = (patch_size - rows % patch_size) % patch_size
        pad_cols = (patch_size - cols % patch_size) % patch_size

        # Pad the image to ensure it can be evenly divided into patches
        padded_image = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        # Create patches
        patches = []
        for i in range(0, padded_image.shape[0], patch_size):
            for j in range(0, padded_image.shape[1], patch_size):
                patch = padded_image[i:i + patch_size, j:j + patch_size]
                patches.append(patch)
        
        return patches, image.shape, padded_image.shape    

    def reconstruct_image(self, patches, original_shape, padded_shape, patch_size=256):
        # ChatGPT assisted with this function
        # Create an empty array to hold the reconstructed image
        reconstructed_image = np.zeros(padded_shape)
        
        patch_index = 0
        for i in range(0, padded_shape[0], patch_size):
            for j in range(0, padded_shape[1], patch_size):
                reconstructed_image[i:i + patch_size, j:j + patch_size] = patches[patch_index]
                patch_index += 1
        
        # Remove the padding to get the original image size
        return reconstructed_image[:original_shape[0], :original_shape[1]]

    # # Example usage
    # reconstructed_image = reconstruct_image(flags, original_shape, padded_shape)
    # print(f'Reconstructed image shape: {reconstructed_image.shape}')

    # # Example usage
    # image = rfi_antenna_data[0]  # Replace with your image
    # image = np.sqrt(image)
    # image = image/np.median(image)
    # stat = stats.median_abs_deviation(image, axis=None)
        
    # median = np.median(image)
    # # image = np.clip(image, (median + (stat * 0)), (median + (stat * 10)))

    # patches, original_shape, padded_shape = create_patches(image)
    # print(f'Number of patches: {len(patches)}')



    #########################
    # Running Models
    #########################

    def run_sam(self,remove_largest=True,pad_width=50):

        self.pad_width = pad_width
        self.pad_spectrograph(pad_width=pad_width)
        self.create_RGB_channels()

        masks = self.mask_generator.generate(self.test_image)

        if remove_largest:
            masks = self.rm_largest_mask(masks)
        
        self.flags = self.create_flags(masks)

        self.flags = self.flags[pad_width:-pad_width,pad_width:-pad_width]
        self.spectrograph = self.temp_spectrograph[pad_width:-pad_width,pad_width:-pad_width]

    def run_sam_predict(self,pad_width=50):

        self.pad_width = pad_width
        # self.pad_spectrograph(pad_width=pad_width)
        # self.create_RGB_channels()
        self.predictor.set_image(self.test_image)

        self.find_spectrograph_peaks()
        masks, scores, logits = self.predictor.predict(
            point_coords=self.max_peaks[0],
            point_labels=self.max_peaks[1],
            multimask_output=False,
        )

        self.masks = masks
        self.scores = scores
        self.logits = logits
        self.flags = np.logical_not(masks[0])
        # self.flags = self.flags[pad_width:-pad_width,pad_width:-pad_width]
        # self.spectrograph = self.spectrograph[pad_width:-pad_width,pad_width:-pad_width]

    def load_model(self,model_path, sam_type="huge"):
        # "/home/gpuhost002/ddeal/RFI-AI/models/derod_checkpoint_large_real_data_test_v3.pth"
        # Load the model configuration

        model_path = str(model_path)
        sam_type = str(sam_type)

        self.model_config = SamConfig.from_pretrained(f"facebook/sam-vit-{sam_type}")
        self.processor = SamProcessor.from_pretrained(f"facebook/sam-vit-{sam_type}")

        # Create an instance of the model architecture with the loaded configuration
        self.model = SamModel(config=self.model_config)
        # Update the model by loading the weights from saved file.
        self.model.load_state_dict(torch.load(model_path))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)


    def run_rfi_model(self, pad_width=50, patch_run=False, threshold=0.5, save=False):

        self.pad_width = pad_width

        print("SAMRFI Progress...")


        if not patch_run:

            pol_flags_list = []

            for baseline in tqdm(range(self.rfi_antenna_data.shape[0])):

                flags = []

                for pol in range(self.rfi_antenna_data.shape[1]):

                    data = self.rfi_antenna_data[baseline,pol,:,:]

                    single_data = data/np.median(data)
                    
                    stat = stats.median_abs_deviation(single_data, axis=None)
                    median = np.median(single_data)
                    single_data = np.clip(single_data, (median + (stat * 1)),(median + (stat * 10)))

                    single_data = np.pad(single_data, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=(median*.5 + (stat * 1)))

                    single_patch = Image.fromarray(single_data).convert("RGB")
                    bbox = self.get_bounding_box(single_data)

                    inputs = self.processor(single_patch, input_boxes=[[bbox]], return_tensors="pt")

                    # Move the input tensor to the GPU if it's not already there
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model.eval()

                    # forward pass
                    with torch.no_grad():
                        outputs = self.model(**inputs,multimask_output=False)

                        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
                        masks = masks[0].cpu().numpy().squeeze()

                    masks = masks[pad_width:-pad_width,pad_width:-pad_width]

                    flags.append(masks)
                pol_flags = np.stack(flags)
                pol_flags_list.append(pol_flags)

            self.pol_flags_list = pol_flags_list
            baseline_flags = np.stack(pol_flags_list)
            
            self.flags = baseline_flags


            np.save(f"{self.directory}/flags.npy",baseline_flags)

        elif patch_run:

            pol_flags_list = []

            for baseline in tqdm(range(self.rfi_antenna_data.shape[0])):

                flags = []

                for pol in range(self.rfi_antenna_data.shape[1]):

                    data = self.rfi_antenna_data[baseline,pol,:,:]
                    single_data = np.sqrt(data)
                    single_data = single_data/np.median(single_data)

                    patches, original_shape, padded_shape = self.create_patches(single_data)

                    patch_flags = []

                    for patch in patches:

                        single_patch = Image.fromarray(patch).convert("RGB")

                        bbox = self.get_bounding_box(patch)

                        inputs = self.processor(single_patch, input_boxes=[[bbox]], return_tensors="pt")

                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        self.model.eval()

                        with torch.no_grad():
                            outputs = self.model(**inputs,multimask_output=False)

                            single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                            single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
                            single_patch_prediction = (single_patch_prob > threshold)

                        patch_flags.append(single_patch_prediction > 0)

                    master_flag = self.reconstruct_image(patch_flags, original_shape, padded_shape)

                    flags.append(master_flag)

                pol_flags = np.stack(flags)
                pol_flags_list.append(pol_flags)

            self.pol_flags_list = pol_flags_list
            baseline_flags = np.stack(pol_flags_list)

            self.flags = baseline_flags

            self.create_residuals()



    #########################
    # Post processing
    #########################

    def create_residuals(self,):
        self.residuals = np.where(np.logical_not(self.flags), self.rfi_antenna_data, 0)
    #########################
    # Segmentation Statistics
    #########################


    def calculate_snr_performance(self,):

        self.rfi_table = self.rfi_table.sort_values('center_freq')
        signal_to_noise_rfi = self.rfi_table['amplitude']/self.noise
        self.rfi_table['SNR_rfi'] = signal_to_noise_rfi
        
        self.temp_spectrograph = self.spectrograph
        self.create_model_spectrograph()

        snr_spec = self.model_spectrograph/self.noise

        # Determine how to calculate negative performance for false positives or false negatives
        snr_stacks = []
        performance = []
        for snr in self.rfi_table['SNR_rfi']:
            snr_2d_array = np.where((snr_spec >= snr-0.1) & (snr_spec <= snr+0.1), snr_spec, 0)
            performance.append(np.sum((snr_2d_array > 0) & self.flags) / np.sum(snr_2d_array>0))
            snr_stacks.append(snr_2d_array)
            
        snr_stacks = np.dstack(snr_stacks)
        self.rfi_table['snr_performance'] = performance

        return self.rfi_table

    def calculate_metrics(self, threshold=0.1):
        """
        Calculates performance metrics for the RFI detection model.

        Parameters:
            threshold (float): The threshold value for converting SNR to binary classification. Default is 0.1.

        Returns:
            dict: A dictionary containing the performance metrics:
                - 'True Positives': The number of true positive detections.
                - 'True Negatives': The number of true negative detections.
                - 'False Positives': The number of false positive detections.
                - 'False Negatives': The number of false negative detections.
                - 'Precision': The precision score, which is the ratio of true positives to the sum of true positives and false positives.
                - 'Recall': The recall score, which is the ratio of true positives to the sum of true positives and false negatives.
                - 'False Positive Rate': The false positive rate, which is the ratio of false positives to the sum of false positives and true negatives.
        """

        # Assisted with Copilot
        
        self.rfi_table = self.rfi_table.sort_values('center_freq')
        signal_to_noise_rfi = self.rfi_table['amplitude'] / self.noise
        self.rfi_table['SNR_rfi'] = signal_to_noise_rfi
        
        self.temp_spectrograph = self.spectrograph
        self.create_model_spectrograph()

        snr_spec = self.model_spectrograph / self.noise

        # Convert SNR to binary classification based on threshold
        model = (snr_spec >= threshold).astype(int)
        detections = self.flags.astype(int)

        # Calculate TP, TN, FP, FN
        TP = np.sum((model == 1) & (detections == 1))
        TN = np.sum((model == 0) & (detections == 0))
        FP = np.sum((model == 1) & (detections == 0))
        FN = np.sum((model == 0) & (detections == 1))

        # Calculate performance metrics
        performance = {
            'True Positives': TP,
            'True Negatives': TN,
            'False Positives': FP,
            'False Negatives': FN,
            'Precision': TP / (TP + FP) if TP + FP > 0 else 0,
            'Recall': TP / (TP + FN) if TP + FN > 0 else 0,
            'False Positive Rate': FP / (FP + TN) if FP + TN > 0 else 0,
        }
        
        self.metrics = pd.DataFrame([performance], columns=performance.keys())
    
    def calculate_detection_performance(self,threshold=10):

        self.rfi_table = self.rfi_table.sort_values('center_freq')
        signal_to_noise_rfi = self.rfi_table['amplitude']/self.noise
        self.rfi_table['SNR_rfi'] = signal_to_noise_rfi
        
        self.temp_spectrograph = np.zeros((self.time_int, self.points))

        snr_spec = self.model_spectrograph/self.noise

        # Determine how to calculate negative performance for false positives or false negatives
        snr_stacks = []
        performance = []
        for rfi in self.rfi_table.iterrows():
            self.temp_spectrograph = np.zeros((self.time_int, self.points))
            if rfi[1]['rfi_type'] == 'persistent':
                self.add_rifi_spectrograph(rfi[1]['amplitude'], rfi[1]['center_freq'], rfi[1]['bandwidth'])
                performance.append(np.sum((self.temp_spectrograph > threshold) & self.flags) / np.sum(self.temp_spectrograph>threshold))
                snr_stacks.append(self.temp_spectrograph)
            elif rfi[1]['rfi_type'] == 'intermittent':
                self.intermittent_rfi(rfi[1]['amplitude'], rfi[1]['center_freq'],
                                                rfi[1]['bandwidth'], rfi[1]['time_period'], rfi[1]['duty_cycle'], rfi[1]['time_offset'])
                performance.append(np.sum((self.temp_spectrograph > threshold) & self.flags) / np.sum(self.temp_spectrograph>threshold))
                snr_stacks.append(self.temp_spectrograph)
            
        snr_stacks = np.dstack(snr_stacks)
        self.rfi_table['detection_performance'] = performance

        return self.rfi_table

    def overplot_detection(self,save=False):

        masked_spectrograph = np.where(np.logical_not(self.flags), self.spectrograph, 0)


        fig, ax = plt.subplots(3,1, figsize=(10, 12),)

        ax[0].imshow(self.spectrograph, aspect='auto', cmap='viridis')
        ax[1].imshow(self.flags, aspect='auto', cmap='viridis')
        ax[2].imshow(masked_spectrograph, aspect='auto', cmap='viridis')
        ax[0].scatter(self.max_peaks[0][:,1], self.max_peaks[0][:,0], c='r', s=0.5)

        ax[0].set_title('Original Spectrograph')
        ax[1].set_title('Flags')
        ax[2].set_title('Residual')

        if save:

            base_filename = f"model_l_noise{self.noise}overplot_detection_broad_"
            file_number = 1
            output_filename = f"{base_filename}{file_number}.png"

            while os.path.exists(output_filename):
                file_number += 1
                output_filename = f"{base_filename}{file_number}.png"
                
            #plt.savefig(output_filename)
            fig.savefig(output_filename)
        else:
            plt.show()

        fig.clf() 
        plt.clf()
        

    def test_realdata(self, save=True,):

        rfi_per = []
        rfi_scr = []
        baseline_id = []
        pol_id = []   

        method_name = self.test_realdata.__name__

        if save:
            method_dir = os.path.join(self.directory, method_name)

            if not os.path.exists(method_dir):
                os.makedirs(method_dir)


        for i in tqdm(range(self.rfi_antenna_data.shape[0])):
            for j in range(self.rfi_antenna_data.shape[1]):
                per, scr = self.runtest(self.rfi_antenna_data[i,j,:,:], self.flags[i,j,:,:])

                rfi_per.append(per)
                rfi_scr.append(scr)

                baseline_id.append(i)
                pol_id.append(j)

                if save:
                    fig, ax = plt.subplots(3,1, figsize=(14, 8),)
                    norm1 = ImageNormalize(self.rfi_antenna_data[i,j,:,:].T, interval=ZScaleInterval())
                    norm2 = ImageNormalize(self.flags[i,j,:,:].T, interval=ZScaleInterval())

                    residual = np.where(np.logical_not(self.flags[i,j,:,:].T), self.rfi_antenna_data[i,j,:,:].T, 0)

                    norm3 = ImageNormalize(residual, interval=ZScaleInterval())

                    im1 = ax[0].imshow(self.rfi_antenna_data[i,j,:,:].T, norm=norm1, aspect='auto')
                    im2 = ax[1].imshow(self.flags[i,j,:,:].T, norm=norm2, aspect='auto')
                    im3 = ax[2].imshow(residual, norm=norm3, aspect='auto')

                    ax[0].set_title('Baseline')
                    ax[1].set_title('Flags')
                    ax[2].set_title('Residual')

                    plt.colorbar(im1)
                    plt.colorbar(im2)
                    plt.colorbar(im3)

                    plt.close(fig)

                    fig.suptitle(f'Baseline {i} - Polarization {j} - RFI Percent Flagged: {per:.2f} - Score: {scr:.2f}')
                    fig.savefig(f'{method_dir}/real_data_test_baseline_{i}_pol_{j}.png')

        self.realdata_results = pd.DataFrame({'Baseline': baseline_id , 'Polarization': pol_id, 'RFI Percent Flagged':rfi_per, 'Score':rfi_scr})
        
        if save:
            self.realdata_results.to_csv(f'{method_dir}/real_data_test_results.csv')
        
        return self.realdata_results

    def test_calcquality(self, save=True, plot=True,):

        method_name = self.test_calcquality.__name__

        if save:
            method_dir = os.path.join(self.directory, method_name)

            if not os.path.exists(method_dir):
                os.makedirs(method_dir)

        self.synthetic_rfi_table = self.rfi_table

        self.rfi_table = pd.DataFrame(columns=['rfi_type', 'amplitude', 'center_freq', 'bandwidth', 'duty_cycle', 'time_period', 'time_offset'])

        self.min_freq=0
        self.max_freq=2000
        self.time_int=2000
        self.points=2000

        self.frequencies = np.linspace(self.min_freq, self.max_freq, self.points)
        self.temp_spectrograph = np.zeros((self.time_int, self.points))

        norfi = self.temp_spectrograph

        self.mean = 20
        self.noise = 5

        noisey_spec = self.create_spectrograph()

        test1 = self.calcquality(noisey_spec, norfi)


        self.add_rifi_spectrograph(amplitude=10, center_freq=500, bandwidth=20, table=True)
        self.create_model_spectrograph()

        onerfi = self.model_spectrograph

        data1 = onerfi + noisey_spec

        statflag1 = stats.median_abs_deviation(onerfi, axis=None)
        medianflag1 = np.median(onerfi)

        flag1 = onerfi > 5
        test2 = self.calcquality(data1, flag1)
        test2_noflag = self.calcquality(data1, norfi)

        self.add_rifi_spectrograph(amplitude=10, center_freq=1500, bandwidth=20, table=True)
        self.create_model_spectrograph()
        tworfi = self.model_spectrograph

        data2 = tworfi + noisey_spec

        flag2 = tworfi > 5

        test3 = self.calcquality(data2, flag2)
        test3_noflag = self.calcquality(data2, norfi)

        data3 = data2
        flag3 = tworfi > .625

        test4 = self.calcquality(data3, flag3)
        test4_noflag = self.calcquality(data3, norfi)
        self.calcquality_results = pd.DataFrame({'NoRFI': [test1], 'OneRFI': [test2], 'TwoRFI': [test3], 'DoubleArea': [test4]})

        if save:
            self.calcquality_results.to_csv(f'{method_dir}/test_calcquality_results.csv')

        if plot:
            # Test 1
            fig, ax = plt.subplots(2,1, figsize=(7, 5),)
            ax[0].imshow(noisey_spec, aspect='auto', cmap='viridis')
            ax[1].imshow(norfi, aspect='auto', cmap='viridis')

            ax[0].set_title(f'calcquality score - {test1:.2f}/{test1:.2f} - No RFI (Test 1)')
            fig.tight_layout()
            plt.show()
            fig.savefig(f'{method_dir}/calcquality_test1.png')
            fig.clf()

            # Test 2
            fig, ax = plt.subplots(2,1, figsize=(7, 5),)
            ax[0].imshow(data1, aspect='auto', cmap='viridis')
            ax[1].imshow(flag1, aspect='auto', cmap='viridis')

            ax[0].set_title(f'calcquality score - {test2_noflag:.2f}/{test2:.2f} - One RFI (Test 2)')

            ax[0].set_title(f'calcquality score - {test2:.2f} - One RFI (Test 2)')

            fig.tight_layout()
            plt.show()
            fig.savefig(f'{method_dir}/calcquality_test2.png')
            fig.clf()

            # Test 3
            fig, ax = plt.subplots(2,1, figsize=(7, 5),)
            ax[0].imshow(data2, aspect='auto', cmap='viridis')
            ax[1].imshow(flag2, aspect='auto', cmap='viridis')

            ax[0].set_title(f'calcquality score - {test3_noflag:.2f}/{test3:.2f} - Two RFI (Test 3)')

            ax[0].set_title(f'calcquality score - {test3:.2f} - Two RFI (Test 3)')

            fig.tight_layout()
            plt.show()
            fig.savefig(f'{method_dir}/calcquality_test3.png')
            fig.clf()

            # Test 4
            fig, ax = plt.subplots(2,1, figsize=(7, 5),)
            ax[0].imshow(data3, aspect='auto', cmap='viridis')
            ax[1].imshow(flag3, aspect='auto', cmap='viridis')

            ax[0].set_title(f'calcquality score - {test4_noflag:.2f}/{test4:.2f} - Double Area (Test 4)')

            ax[0].set_title(f'calcquality score - {test4:.2f} - Double Area (Test 4)')

            fig.tight_layout()
            plt.show()
            fig.savefig(f'{method_dir}/calcquality_test4.png')
            fig.clf()

        return self.calcquality_results

    ##########
    # Plotting
    ##########

    def plot(self, mode='DATA', baseline=0, polarization=0):

        if mode == 'DATA':
            data = self.rfi_antenna_data
        if mode == 'FLAG':
            data = self.flags

        norm = ImageNormalize(data[baseline,polarization,:,:].T, interval=ZScaleInterval())

        fig, ax = plt.subplots(figsize=(16, 8),dpi=300)
        ax.imshow(data[baseline,polarization,:,:].T, aspect='auto', cmap='viridis', norm=norm)


    ############
    # Save Flags
    ############

    def save_flags(self,):

        for baseline, antennas in enumerate(tqdm(self.antenna_baseline_map)):
            main_flags = self.flags[baseline,:,:,:]
            for spw in self.spw:
                flags = main_flags[:,0+(spw*self.channels_per_spw[0]):self.channels_per_spw[0]+(spw*self.channels_per_spw[0]),:]
                self.subtable = self.tb.query(f'DATA_DESC_ID=={spw} && ANTENNA1=={antennas[0]} && ANTENNA2=={antennas[1]}')
                self.subtable.putcol('FLAG', flags)

        


    ######################
    # Real Data Statistics
    ######################

    # Functions from Dr. Preshanth Jagannathan

    def runtest(self, dat, flag):

        # plotit(dat, flag)

        # print('% Flagged : ', np.sum(flag) / (1.0 * np.prod(flag.shape)) * 100.0)

        return (np.sum(flag) / (1.0 * np.prod(flag.shape)) * 100.0), self.calcquality(dat, flag)


    def calcquality(self, dat, flag):
        """ Need to minimize the score that it returns"""

        shp = dat.shape

        npts = 0
        sumsq = 0.0
        maxval = 0.0
        leftover = []
        flagged = []
        for chan in range(0, shp[0]):
            for tm in range(0, shp[1]):
                val = np.abs(dat[chan, tm])
                if flag[chan, tm] == False:
                    leftover.append(val)
                else:
                    flagged.append(val)

        dmax, dmean, dstd = self.printstats(np.abs(dat[:, :]))
        rmax, rmean, rstd = self.printstats(leftover)
        fmax, fmean, fstd = self.printstats(flagged)

        maxdev = (rmax - rmean) / rstd
        fdiff = fmean - rmean
        sdiff = fstd - rstd

        # print("Max deviation after flagging : ", maxdev)
        # print("Diff in mean of flagged and unflagged : ", fdiff)
        # print("Std after flagging : ", rstd)
        
        ## Maximum deviation from the mean is 3 sigma. => Gaussian stats. 
        ## => What's leftover is noise-like and without significant outliers.
        aa = np.abs(np.abs(maxdev) - 3.0)

        ## Flagged data has a higher mean than what is left over => flagged only RFI. Maximize the difference between the means
        bb = 1.0 / ((np.abs(fdiff) - rstd) / rstd)
        
        ## Maximize the difference between the std of the flagged and leftover data => Assumes that RFI is widely varying...
        cc = 1.0 / (np.abs(sdiff) / rstd)

        ## Overflagging is bad
        dd = 0.0
        pflag = (len(flagged) / (1.0 * shp[0] * shp[1])) * 100.0
        if pflag > 70.0:
            dd = (pflag - 70.0)/10.0
        
        res = np.sqrt(aa ** 2 + bb ** 2 + cc * 2 + dd * 2)

        if (fdiff < 0.0):
            res = res + res + 10.0

        # print("Score : ", res)

        return res


    def printstats(self, arr):
        if (len(arr) == 0):
            return 0, 0, 1

        med = np.median(arr)
        std = np.std(arr)
        maxa = np.max(arr)
        mean = np.mean(arr)
        # print 'median : ', med
        # print 'std : ', std
        # print 'max : ', maxa
        # print 'mean : ', mean
        # print " (Max - mean)/std : ", ( maxa - mean ) / std

        return maxa, mean, std


    def getvals(self, col='DATA', vis="", spw="", scan=""):

        # print("SPW:", spw, "DDID:", ddid)

        self.tb.open(vis)
        if (spw and scan):
            self.tb.open(vis + '/DATA_DESCRIPTION')
            spwids = self.tb.getcol('SPECTRAL_WINDOW_ID')
            ddid = str(np.where(spwids == eval(spw))[0][0])
            tb1 = self.tb.query('SCAN_NUMBER==' + scan + ' && DATA_DESC_ID==' + ddid + ' && ANTENNA1=1 && ANTENNA2=2')
        else:
            tb1 = self.tb.query('ANTENNA1=1 && ANTENNA2=2')
        dat = tb1.getcol(col)
        tb1.close()
        self.tb.close()
        return dat


    def plotit(self, dat, flag):
        plt.clf()

        fig, ax = plt.subplots(2, 1, figsize=(10, 3),dpi=150)
        ax[0].imshow(np.abs(dat), vmin=0, vmax=100)
        ax[1].imshow(np.abs(dat * (1 - flag)), vmin=0, vmax=100)
