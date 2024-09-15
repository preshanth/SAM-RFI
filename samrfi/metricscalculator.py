import numpy as np
import pandas as pd


def calculate_metrics(self, flags, ground_truth):
    
    flags = flags.astype(int)
    ground_truth = ground_truth.astype(int)

    # Calculate TP, TN, FP, FN
    TP = np.sum((flags == 1) & (ground_truth == 1))
    TN = np.sum((flags == 0) & (ground_truth == 0))
    FP = np.sum((flags == 1) & (ground_truth == 0))
    FN = np.sum((flags == 0) & (ground_truth == 1))

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
    
    return performance
    # def calculate_metrics(self, threshold=0.1):
    #     """
    #     Calculates performance metrics for the RFI detection model.

    #     Parameters:
    #         threshold (float): The threshold value for converting SNR to binary classification. Default is 0.1.

    #     Returns:
    #         dict: A dictionary containing the performance metrics:
    #             - 'True Positives': The number of true positive detections.
    #             - 'True Negatives': The number of true negative detections.
    #             - 'False Positives': The number of false positive detections.
    #             - 'False Negatives': The number of false negative detections.
    #             - 'Precision': The precision score, which is the ratio of true positives to the sum of true positives and false positives.
    #             - 'Recall': The recall score, which is the ratio of true positives to the sum of true positives and false negatives.
    #             - 'False Positive Rate': The false positive rate, which is the ratio of false positives to the sum of false positives and true negatives.
    #     """

    #     # Assisted with Copilot
        
    #     self.rfi_table = self.rfi_table.sort_values('center_freq')
    #     signal_to_noise_rfi = self.rfi_table['amplitude'] / self.noise
    #     self.rfi_table['SNR_rfi'] = signal_to_noise_rfi
        
    #     self.temp_spectrograph = self.spectrograph
    #     self.create_model_spectrograph()

    #     snr_spec = self.model_spectrograph / self.noise

    #     # Convert SNR to binary classification based on threshold
    #     model = (snr_spec >= threshold).astype(int)
    #     detections = self.flags.astype(int)

    #     # Calculate TP, TN, FP, FN
    #     TP = np.sum((model == 1) & (detections == 1))
    #     TN = np.sum((model == 0) & (detections == 0))
    #     FP = np.sum((model == 1) & (detections == 0))
    #     FN = np.sum((model == 0) & (detections == 1))

    #     # Calculate performance metrics
    #     performance = {
    #         'True Positives': TP,
    #         'True Negatives': TN,
    #         'False Positives': FP,
    #         'False Negatives': FN,
    #         'Precision': TP / (TP + FP) if TP + FP > 0 else 0,
    #         'Recall': TP / (TP + FN) if TP + FN > 0 else 0,
    #         'False Positive Rate': FP / (FP + TN) if FP + TN > 0 else 0,
    #     }
        
    #     self.metrics = pd.DataFrame([performance], columns=performance.keys())
    
class RadioRFIMetricsCalculator:

    def __init__(self, radiorfi_instance):

        self.RadioRFI = radiorfi_instance

    def calculate_metrics(self, ground_truth,):
        columns = ['Baseline', 'Polarization', 'True Positives', 'True Negatives', 'False Positives',
                    'False Negatives', 'Precision', 'Recall', 'False Positive Rate']
        metrics_df = pd.DataFrame(columns=columns)

        for baseline in range(self.RadioRFI.flags.shape[0]):
            for polarization in range(self.RadioRFI.flags.shape[1]):
                performance = calculate_metrics(self.RadioRFI.flags[baseline, polarization, :, :], ground_truth[baseline, polarization, :, :], ground_truth)

                # Create a dictionary with the metrics and additional information
                row = {
                    'Baseline': baseline,
                    'Polarization': polarization,
                    'True Positives': performance['True Positives'],
                    'True Negatives': performance['True Negatives'],
                    'False Positives': performance['False Positives'],
                    'False Negatives': performance['False Negatives'],
                    'Precision': performance['Precision'],
                    'Recall': performance['Recall'],
                    'False Positive Rate': performance['False Positive Rate']
                }
                
                # Append the dictionary as a new row to the DataFrame
                row_df = pd.DataFrame([row])

                metrics_df = pd.concat([metrics_df, row_df], ignore_index=True)

                self.metrics_results = metrics_df

    def test_realdata(self, save=True,):

        rfi_per = []
        rfi_scr = []
        baseline_id = []
        pol_id = []   

        method_name = self.test_realdata.__name__

        if save:
            method_dir = os.path.join(self.RadioRFI.directory, method_name)

            if not os.path.exists(method_dir):
                os.makedirs(method_dir)


        for i in tqdm(range(self.RadioRFI.rfi_antenna_data.shape[0])):
            for j in range(self.RadioRFI.rfi_antenna_data.shape[1]):
                per, scr = self.runtest(self.RadioRFI.rfi_antenna_data[i,j,:,:], self.RadioRFI.flags[i,j,:,:])

                rfi_per.append(per)
                rfi_scr.append(scr)

                baseline_id.append(i)
                pol_id.append(j)

                if save:
                    fig, ax = plt.subplots(3,1, figsize=(14, 8),)
                    norm1 = ImageNormalize(self.RadioRFI.rfi_antenna_data[i,j,:,:].T, interval=ZScaleInterval())
                    norm2 = ImageNormalize(self.RadioRFI.flags[i,j,:,:].T, interval=ZScaleInterval())

                    residual = np.where(np.logical_not(self.RadioRFI.flags[i,j,:,:].T), self.RadioRFI.rfi_antenna_data[i,j,:,:].T, 0)

                    norm3 = ImageNormalize(residual, interval=ZScaleInterval())

                    im1 = ax[0].imshow(self.RadioRFI.rfi_antenna_data[i,j,:,:].T, norm=norm1, aspect='auto')
                    im2 = ax[1].imshow(self.RadioRFI.flags[i,j,:,:].T, norm=norm2, aspect='auto')
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

class SyntheticRFIMetricsCalculator:

    def __init__(self, syntheticrfi_instance):
        self.SyntheticRFI = syntheticrfi_instance

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
