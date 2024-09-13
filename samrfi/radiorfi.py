import os
import numpy as np
import pandas as pd
from casatools import table
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from tqdm import tqdm

class RadioRFI:

    def __init__(self, vis=False, dir_path=None):

        from .plotter import Plotter
        self.plotter = Plotter(self)

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

    #########################
    # Running Models
    #########################

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

    def save_flags(self,):

        for baseline, antennas in enumerate(tqdm(self.antenna_baseline_map)):
            main_flags = self.flags[baseline,:,:,:]
            for spw in self.spw:
                flags = main_flags[:,0+(spw*self.channels_per_spw[0]):self.channels_per_spw[0]+(spw*self.channels_per_spw[0]),:]
                self.subtable = self.tb.query(f'DATA_DESC_ID=={spw} && ANTENNA1=={antennas[0]} && ANTENNA2=={antennas[1]}')
                self.subtable.putcol('FLAG', flags)

        