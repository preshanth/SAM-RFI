from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt

from .radiorfi import RadioRFI

class Plotter:
    def __init__(self, objectRFI):
        # Call the constructor of the base class
        self.objectRFI = objectRFI


    def plot(self, mode='DATA', baseline=0, polarization=0):

        if mode == 'DATA':
            data = self.objectRFI.rfi_antenna_data
        if mode == 'FLAG':
            data = self.objectRFI.flags

        norm = ImageNormalize(data[baseline,polarization,:,:].T, interval=ZScaleInterval())

        fig, ax = plt.subplots(figsize=(16, 8),dpi=300)
        ax.imshow(data[baseline,polarization,:,:].T, aspect='auto', cmap='viridis', norm=norm)
        plt.show()


    def overplot_detection(self,save=False):

        fig.clf() 
        plt.clf()    
        
        masked_spectrograph = np.where(np.logical_not(self.objectRFI.flags), self.objectRFI.rfi_antenna_data, 0)

        fig, ax = plt.subplots(3,1, figsize=(10, 12),)

        ax[0].imshow(self.objectRFI.rfi_antenna_data, aspect='auto', cmap='viridis')
        ax[1].imshow(self.objectRFI.flags, aspect='auto', cmap='viridis')
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

    def plotit(self,):

        fig, ax = plt.subplots(2, 1, figsize=(10, 3),dpi=150)
        ax[0].imshow(np.abs(self.objectRFI.rfi_antenna_data), vmin=0, vmax=100)
        ax[1].imshow(np.abs(self.objectRFI.rfi_antenna_data * (1 - self.objectRFI.flags)), vmin=0, vmax=100)