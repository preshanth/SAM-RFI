class MetricsCalculator:
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
