import os
import numpy as np
import pandas as pd
from casatools import table
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from tqdm import tqdm

from .plotter import Plotter
from .metricscalculator import RadioRFIMetricsCalculator, SyntheticRFIMetricsCalculator
class RadioRFI:

    def __init__(self, vis=False, dir_path=None):

        self.plotter = Plotter(self)
        self.radio_metrics = RadioRFIMetricsCalculator(self)
        self.synthetic_metrics = SyntheticRFIMetricsCalculator(self)

        self.rfi_table = pd.DataFrame(columns=['rfi_type', 'amplitude', 'center_freq', 'bandwidth', 'duty_cycle', 'time_period', 'time_offset'])

        self.rfi_antenna_data = None
        self.flags = None

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

    def load(self, vis=None, mode='DATA', ant_i=2):

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
            print(self.rfi_antenna_data.shape)
        
        if mode == 'FLAG':
            self.ms_flags = np.stack(rfi_list)

    def update_flags(self, flags):
        self.flags = flags

    def create_residuals(self,):
        self.residuals = np.where(np.logical_not(self.flags), self.rfi_antenna_data, 0)

    def save_flags(self,):

        for baseline, antennas in enumerate(tqdm(self.antenna_baseline_map)):
            main_flags = self.flags[baseline,:,:,:]
            for spw in self.spw:
                flags = main_flags[:,0+(spw*self.channels_per_spw[0]):self.channels_per_spw[0]+(spw*self.channels_per_spw[0]),:]
                self.subtable = self.tb.query(f'DATA_DESC_ID=={spw} && ANTENNA1=={antennas[0]} && ANTENNA2=={antennas[1]}')
                self.subtable.putcol('FLAG', flags)

        