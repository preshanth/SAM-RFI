# syntheticrfi.py

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
        return np.random.normal(self.mean, self.noise, self.temp_spectrograph.shape[1])

    def generate_rfi(self, amplitude, center_freq, bandwidth, horizontal=False):
        rfi_signal = amplitude * np.exp(-((self.frequencies - center_freq) ** 2) / (2 * (bandwidth ** 2)))
        if horizontal:
            return rfi_signal.reshape(1, -1)
        else:
            return rfi_signal.reshape(-1, 1)

    def square_function(self, amplitude, center_freq, bandwidth, horizontal=False):
        rfi_signal = amplitude * np.where(np.abs(self.frequencies - center_freq) <= bandwidth/2, 1, 0)
        if horizontal:
            return rfi_signal.reshape(1, -1)
        else:
            return rfi_signal.reshape(-1, 1)

    def add_rfi(self, data, amplitude, center_freq, bandwidth, horizontal=False):
        return data + self.generate_rfi(amplitude, center_freq, bandwidth, horizontal)

    def add_square_rfi(self, data, amplitude, center_freq, bandwidth, horizontal=False):
        return data + self.square_function(amplitude, center_freq, bandwidth, horizontal)

    def add_rifi_spectrograph(self, amplitude, center_freq, bandwidth, horizontal=False, table=False):
        axis = 1 if horizontal else 0
        self.temp_spectrograph = np.apply_along_axis(self.add_rfi, axis=axis, arr=self.temp_spectrograph, amplitude=amplitude, center_freq=center_freq, bandwidth=bandwidth, horizontal=horizontal)
        new_row = pd.DataFrame({'rfi_type': ['persistent'], 'amplitude': [amplitude], 'center_freq': [center_freq], 'bandwidth': [bandwidth], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        if table:
            self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)

    def add_square_rifi_spectrograph(self, amplitude, center_freq, bandwidth, horizontal=False, table=False):
        axis = 1 if horizontal else 0
        self.temp_spectrograph = np.apply_along_axis(self.add_square_rfi, axis=axis, arr=self.temp_spectrograph, amplitude=amplitude, center_freq=center_freq, bandwidth=bandwidth, horizontal=horizontal)
        new_row = pd.DataFrame({'rfi_type': ['persistent_sq'], 'amplitude': [amplitude], 'center_freq': [center_freq], 'bandwidth': [bandwidth], 'duty_cycle': [np.nan], 'time_period': [np.nan], 'time_offset': [np.nan]})
        if table:
            self.rfi_table = pd.concat([self.rfi_table, new_row], ignore_index=True)

    def create_spectrograph(self):
        temporal_baseline = []
        for i in range(self.temp_spectrograph.shape[0]):
            temporal_baseline.append(self.baseline_profile())
        spectrograph = np.vstack(temporal_baseline)
        return spectrograph

    def add_spectrographs(self, spectrograph1, spectrograph2):
        return spectrograph1 + spectrograph2

    def create_model_spectrograph(self):
        self.temp_spectrograph = np.zeros((self.time_int, self.points))
        for index, row in self.rfi_table.iterrows():
            if row['rfi_type'] == 'persistent':
                self.add_rifi_spectrograph(row['amplitude'], row['center_freq'], row['bandwidth'])
            if row['rfi_type'] == 'intermittent':
                self.intermittent_rfi(row['amplitude'], row['center_freq'], row['bandwidth'], row['time_period'], row['duty_cycle'], row['time_offset'])
            if row['rfi_type'] == 'persistent_sq':
                self.add_square_rifi_spectrograph(row['amplitude'], row['center_freq'], row['bandwidth'])
        self.model_spectrograph = self.temp_spectrograph

    def intermittent_rfi(self, amplitude, center_freq, bandwidth, time_period, duty_cycle, time_offset=0, table=False):
        time_int, points = self.temp_spectrograph.shape
        modified_spectrograph = self.temp_spectrograph
        time_mask = np.zeros(time_int)
        period_indices = np.arange(time_offset, time_int, time_period)
        for start_idx in period_indices:
            end_idx = min(start_idx + int(time_period * duty_cycle), time_int)
            time_mask[int(start_idx):int(end_idx)] = 1
        rfi_signal_1 = self.generate_rfi(amplitude, center_freq, bandwidth, horizontal=False)
        for t in range(time_int):
            if time_mask[t] == 1:
                modified_spectrograph[t, :] += rfi_signal_1
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
        self.frequencies = np.linspace(min_freq, max_freq, points)
        self.time_int = time_int
        self.points = points
        self.edge_buffer = edge_buffer
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.noise = noise
        self.mean = mean
        self.temp_spectrograph = np.zeros((self.time_int, self.points))
        
        self.add_rifi_spectrograph(amplitude=np.random.normal(50, 10), center_freq=np.random.uniform(min_freq + edge_buffer, max_freq - edge_buffer), bandwidth=np.abs(np.random.normal(50, 3)), table=True)
        self.add_rifi_spectrograph(amplitude=np.random.normal(60, 10), center_freq=np.random.uniform(min_freq + edge_buffer, max_freq - edge_buffer), bandwidth=np.abs(np.random.normal(5, 1)), table=True, horizontal=True)
        for i in np.arange(num_persistent):
            self.add_rifi_spectrograph(amplitude=np.random.normal(50, 10), center_freq=np.random.uniform(min_freq + edge_buffer, max_freq - edge_buffer), bandwidth=np.abs(np.random.normal(11, 5)), table=True)
        for i in np.arange(num_intermittent):
            self.intermittent_rfi(amplitude=np.random.normal(50, 10), center_freq=np.random.uniform(min_freq + edge_buffer, max_freq - edge_buffer), bandwidth=np.abs(np.random.normal(50, 10)), time_period=np.random.randint(1, 500), duty_cycle=np.random.uniform(0, 1), time_offset=np.random.randint(1, 100), table=True)
        self.spectrograph = self.temp_spectrograph + self.create_spectrograph()
        self.rfi_antenna_data = self.spectrograph.reshape(1, 1, self.points, self.time_int)