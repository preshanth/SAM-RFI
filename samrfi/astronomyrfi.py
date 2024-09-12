class AstronomyRFI:

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

        