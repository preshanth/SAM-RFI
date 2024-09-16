import numpy as np
import pandas as pd

from .radiorfi import RadioRFI

class SyntheticRFI(RadioRFI):

    def __init__(self, points_freq=None, points_time=None, min_freq=None, max_freq=None,
                    max_time=None, min_time=None, dir_path=False):
        
        super().__init__(dir_path)
        
        self.points_time = points_time
        self.points_freq = points_freq

        self.min_freq = min_freq
        self.max_freq = max_freq

        self.min_time = min_time
        self.max_time = max_time

        self.noise = None
        self.mean = None
        self.temp_spectrograph = None
        self.model_spectrograph = None

        self.persistent_rfi_spec = np.zeros((self.points_freq, self.points_time))
        self.blank_spectrograph = np.zeros((self.points_freq, self.points_time))

        self.frequencies = np.linspace(min_freq, max_freq, points_freq)
        self.times = np.linspace(min_time, max_time, points_time)

        self.rfi_table = pd.DataFrame(columns=['rfi_type', 'amplitude', 'center_freq', 'bandwidth', 'center_time', 'timewidth', 'duty_cycle', 'time_period', 'time_offset'])

    def baseline_profile(self):
        """
        Generates a baseline profile using a normal distribution.

        Returns:
            numpy.ndarray: The baseline profile.
        """
        return np.random.normal(self.mean, self.noise, self.points_time)

    # Generate synthetic RFI signals
    def gaussian_function(self, x, amplitude, center, width):
        """
        Generates a radio frequency interference (RFI) signal.

        Parameters:
            amplitude (float): The amplitude of the RFI signal.
            center_freq (float): The center frequency of the RFI signal.
            bandwidth (float): The bandwidth of the RFI signal.

        Returns:
            numpy.ndarray: An array representing the RFI signal.
        """
        return amplitude * np.exp(-((x - center) ** 2) / (2 * (width ** 2)))

    
    def square_function(self, x, amplitude, center ,width):
        return amplitude * np.where(np.abs(x - center) <= width/2, 1, 0)

    #####
    # Adding functions to data
    #####

    def add_gaussian_rfi(self, data, x, amplitude, center, width):
        return data+self.gaussian_function(x, amplitude, center, width)


    def add_square_rfi(self, data, x, amplitude, center, width):
        return data+self.square_function(x, amplitude, center, width)

    def add_gaussian_rfi_spectrograph(self, amplitude, center, width, rfi_axis='FREQ', table=False):
        """
        Add Gaussian RFI to the spectrograph along the specified axis.

        Parameters:
            amplitude (float): The amplitude of the RFI signal.
            center (float): The center frequency or time of the RFI signal.
            width (float): The bandwidth or timewidth of the RFI signal.
            rfi_axis (str): The axis along which to add the RFI ('FREQ' or 'TIME').
            table (bool): Whether to add the RFI parameters to the RFI table.

        Returns:
            numpy.ndarray: The spectrograph with added Gaussian RFI.
        """
        if rfi_axis == 'FREQ':
            persistent_rfi_spec = np.apply_along_axis(self.add_gaussian_rfi, axis=0, arr=self.blank_spectrograph, x=self.frequencies, amplitude=amplitude, center=center, width=width)
            new_row = pd.DataFrame({'rfi_type': ['gauss_persistent_freq'], 'amplitude': [amplitude], 'center_freq': [center], 'bandwidth': [width], 'center_time': [np.nan], 'timewidth': [np.nan], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        elif rfi_axis == 'TIME':
            persistent_rfi_spec = np.apply_along_axis(self.add_gaussian_rfi, axis=1, arr=self.blank_spectrograph, x=self.times, amplitude=amplitude, center=center, width=width)
            new_row = pd.DataFrame({'rfi_type': ['gauss_persistent_time'], 'amplitude': [amplitude], 'center_freq': [np.nan], 'bandwidth': [np.nan], 'center_time': [center], 'timewidth': [width], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        else:
            raise ValueError("Invalid rfi_axis value. Use 'FREQ' or 'TIME'.")

        if table:
            self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)

        return persistent_rfi_spec

    def add_square_rfi_spectrograph(self, amplitude, center, width, rfi_axis='FREQ', table=False):
        """
        Add Square RFI to the spectrograph along the specified axis.

        Parameters:
            amplitude (float): The amplitude of the RFI signal.
            center (float): The center frequency or time of the RFI signal.
            width (float): The bandwidth or timewidth of the RFI signal.
            rfi_axis (str): The axis along which to add the RFI ('FREQ' or 'TIME').
            table (bool): Whether to add the RFI parameters to the RFI table.

        Returns:
            numpy.ndarray: The spectrograph with added Square RFI.
        """
        if rfi_axis == 'FREQ':
            persistent_rfi_spec = np.apply_along_axis(self.add_square_rfi, axis=0, arr=self.blank_spectrograph, x=self.frequencies, amplitude=amplitude, center=center, width=width)
            new_row = pd.DataFrame({'rfi_type': ['square_persistent_freq'], 'amplitude': [amplitude], 'center_freq': [center], 'bandwidth': [width], 'center_time': [np.nan], 'timewidth': [np.nan], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        elif rfi_axis == 'TIME':
            persistent_rfi_spec = np.apply_along_axis(self.add_square_rfi, axis=1, arr=self.blank_spectrograph, x=self.times, amplitude=amplitude, center=center, width=width)
            new_row = pd.DataFrame({'rfi_type': ['square_persistent_time'], 'amplitude': [amplitude], 'center_freq': [np.nan], 'bandwidth': [np.nan], 'center_time': [center], 'timewidth': [width], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        else:
            raise ValueError("Invalid rfi_axis value. Use 'FREQ' or 'TIME'.")

        if table:
            self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)

        return persistent_rfi_spec

    def create_spectrograph(self,):
        temporal_baseline = []
        for i in range(self.points_freq):
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
            points_time (int): The number of time intervals.
            points_freq (int): The number of points_freq in the spectrograph.

        Returns:
            numpy.ndarray: The model spectrograph.

        """

        self.temp_spectrograph = np.zeros((self.points_time, self.points_freq))
        for index, row in self.rfi_table.iterrows():
            if row['rfi_type'] == 'gauss_persistent_freq':
                self.add_gaussian_rfi_spectrograph(row['amplitude'], 
                                                        row['center_freq'],  row['bandwidth'])

            if row['rfi_type'] == 'gauss_persistent_time':
                self.add_gaussian_rfi_spectrograph(row['amplitude'], 
                                                        row['center_freq'],  row['bandwidth'])

            if row['rfi_type'] == 'intermittent':
                self.intermittent_rfi(row['amplitude'], row['center_freq'],
                                                row['bandwidth'], row['time_period'], row['duty_cycle'], row['time_offset'])

            if row['rfi_type'] == 'persistent_sq':
                self.add_square_rfi_spectrograph(row['amplitude'], 
                                                        row['center_freq'],  row['bandwidth'])

            
        self.model_spectrograph = self.temp_spectrograph

    def intermittent_rfi(self, amplitude, center_freq, bandwidth, time_period, duty_cycle, time_offset=0, func_type='GAUSS', table=False, ):
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
        modified_spectrograph = self.blank_spectrograph
        
        # Create the time mask for intermittent RFI
        time_mask = np.zeros(self.points_time)
        period_indices = np.arange(time_offset, self.points_time, time_period)
        for start_idx in period_indices:
            end_idx = min(start_idx + int(time_period * duty_cycle), self.points_time)
            time_mask[int(start_idx):int(end_idx)] = 1
        
        # Generate the RFI signals for both channels
        if func_type == 'GAUSS':
            rfi_signal_1 = self.gaussian_function(self.frequencies, amplitude, center_freq, bandwidth)
            rfi_type = 'intermittent_gauss'
        if func_type == 'SQUARE':
            rfi_signal_1 = self.square_function(self.frequencies, amplitude, center_freq, bandwidth)
            rfi_type = 'intermittent_square'
        else:
            raise ValueError("Invalid func_type value. Use 'GAUSS' or 'SQUARE'.")
        
        print(rfi_signal_1.shape)

        # Add the intermittent RFI to the spectrograph
        for t in range(self.points_time):
            if time_mask[t] == 1:
                modified_spectrograph[:, t] += rfi_signal_1
        
        # Update the RFI table
        if table:
            new_rows = pd.DataFrame({
                'rfi_type': [rfi_type],
                'amplitude': [amplitude],
                'center_freq': [center_freq],
                'bandwidth': [bandwidth],
                'center_time': [np.nan],
                'timewidth': [np.nan],
                'duty_cycle': [duty_cycle],
                'time_period': [time_period],
                'time_offset': [time_offset],
            })
            self.rfi_table = pd.concat([self.rfi_table, new_rows], ignore_index=True)
        
        return modified_spectrograph

    def generate_waterfall(self, pers_freq_gauss=1, pers_time_gauss=0, inter_freq_gauss=0, inter_freq_square=0, pers_freq_square=0, pers_time_square=0, noise=10, mean=5, edge_buffer=50):
        """
        Generate a waterfall plot with simulated radio frequency interference (RFI).

        Parameters:
            pers_freq_gauss (int): The number of persistent RFI signals to add to the waterfall plot in frequency. Default is 1.
            pers_time_gauss (int): The number of persistent RFI signals to add to the waterfall plot in time. Default is 0.
            inter_freq_gauss (int): The number of intermittent RFI signals to add to the waterfall plot. Default is 0.
            noise (float): The noise level of the waterfall plot. Default is 10.
            mean (float): The mean level of the waterfall plot. Default is 5.
            edge_buffer (int): The buffer size around the edges of the waterfall plot. Default is 50.

        Returns:
            None
        """
        self.noise = noise
        self.mean = mean

        persistent_rfi_spec = self.blank_spectrograph
        persistent_rfi_time = self.blank_spectrograph
        intermittent_rfi_gauss = self.blank_spectrograph
        intermittent_rfi_square = self.blank_spectrograph
        square_rfi_spec = self.blank_spectrograph
        square_rfi_time = self.blank_spectrograph

        spec_list = []
        for i in np.arange(pers_freq_gauss):
            spec_list.append(self.add_gaussian_rfi_spectrograph(
                amplitude=np.random.normal(50, 10),
                center=np.random.uniform(self.min_freq + edge_buffer, self.max_freq - edge_buffer),
                width=np.abs(np.random.normal(50, 3)),
                rfi_axis='FREQ',
                table=True
            ))
        persistent_rfi_spec = np.sum(spec_list, axis=0)

        spec_list = []
        for i in np.arange(pers_time_gauss):
            spec_list.append(self.add_gaussian_rfi_spectrograph(
                amplitude=np.random.normal(50, 10),
                center=np.random.uniform(self.min_time + edge_buffer, self.max_time - edge_buffer),
                width=np.abs(np.random.normal(50, 3)),
                rfi_axis='TIME',
                table=True
            ))
        persistent_rfi_time = np.sum(spec_list, axis=0)

        spec_list = []
        for i in np.arange(inter_freq_gauss):
            spec_list.append(self.intermittent_rfi(
                amplitude=np.random.normal(50, 10),
                center_freq=np.random.uniform(self.min_freq + edge_buffer, self.max_freq - edge_buffer),
                bandwidth=np.abs(np.random.normal(50, 10)),
                time_period=np.random.randint(1, 500),
                duty_cycle=np.random.uniform(0, 1),
                time_offset=np.random.randint(1, 100),
                func_type='GAUSS',
                table=True
            ))
        intermittent_rfi_gauss = np.sum(spec_list, axis=0)

        spec_list = []
        for i in np.arange(inter_freq_square):
            spec_list.append(self.intermittent_rfi(
                amplitude=np.random.normal(50, 10),
                center_freq=np.random.uniform(self.min_freq + edge_buffer, self.max_freq - edge_buffer),
                bandwidth=np.abs(np.random.normal(50, 10)),
                time_period=np.random.randint(1, 500),
                duty_cycle=np.random.uniform(0, 1),
                time_offset=np.random.randint(1, 100),
                func_type='SQUARE',
                table=True
            ))
        intermittent_rfi_square = np.sum(spec_list, axis=0)

        spec_list = []
        for i in np.arange(pers_freq_square):
            spec_list.append(self.add_square_rfi_spectrograph(
                amplitude=np.random.uniform(50, 10), 
                center=np.random.uniform(self.min_freq + edge_buffer, self.max_freq - edge_buffer), 
                width=np.random.uniform(1, 10), 
                rfi_axis='FREQ',
                table=True
            ))
        square_rfi_spec = np.sum(spec_list, axis=0)

        spec_list = []
        for i in np.arange(pers_time_square):
            spec_list.append(self.add_square_rfi_spectrograph(
                amplitude=np.random.uniform(50, 10), 
                center=np.random.uniform(self.min_time + edge_buffer, self.max_time - edge_buffer), 
                width=np.random.uniform(75, 10), 
                rfi_axis='TIME',
                table=True))
        square_rfi_time = np.sum(spec_list, axis=0)

        self.spectrograph = persistent_rfi_spec + persistent_rfi_time + intermittent_rfi_gauss + intermittent_rfi_square + square_rfi_spec + square_rfi_time + self.create_spectrograph()
        self.rfi_antenna_data = self.spectrograph.reshape(1, 1, self.points_freq, self.points_time)