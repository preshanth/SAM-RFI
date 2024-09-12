class Plotter:

    def overplot_detection(self,save=False):

        plt.clf()    
        
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

    def plotit(self, dat, flag):
        plt.clf()

        fig, ax = plt.subplots(2, 1, figsize=(10, 3),dpi=150)
        ax[0].imshow(np.abs(dat), vmin=0, vmax=100)
        ax[1].imshow(np.abs(dat * (1 - flag)), vmin=0, vmax=100)