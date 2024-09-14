import numpy as np
import pandas as pd

from .radiorfi import RadioRFI

class SyntheticRFI(RadioRFI):

    def __init__(self, dir_path=False):
        
        super().__init__(dir_path)
        
        self.time_int = None
        self.points = None
        self.edge_buffer = None
        self.min_freq = None
        self.max_freq = None
        self.noise = None
        self.mean = None
        self.temp_spectrograph = None
        self.model_spectrograph = None

        self.rfi_table = pd.DataFrame(columns=['rfi_type', 'amplitude', 'center_freq', 'bandwidth', 'duty_cycle', 'time_period', 'time_offset'])

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

    # def add_rifi_spectrograph(self, amplitude, center_freq, bandwidth, table=False, horizontal=False):
    #     rfi_signal = self.generate_rfi(amplitude, center_freq, bandwidth)
        
    #     if horizontal:
    #         for i in range(self.temp_spectrograph.shape[0]):
    #             self.temp_spectrograph[i, :] += rfi_signal
    #             new_row = pd.DataFrame({'rfi_type':['persistent'], 'amplitude': [amplitude], 'center_freq': [center_freq], 'bandwidth': [bandwidth], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
    #     else:
    #         for i in range(self.temp_spectrograph.shape[1]):
    #             self.temp_spectrograph[:, i] += rfi_signal
    #             new_row = pd.DataFrame({'rfi_type':['persistent'], 'amplitude': [amplitude], 'center_freq': [center_freq], 'bandwidth': [bandwidth], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})

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

        self.rfi_antenna_data = self.spectrograph.reshape(1, 1, self.points, self.time_int)