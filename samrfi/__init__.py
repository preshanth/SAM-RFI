from .astronomyrfi import AstronomyRFI
from .syntheticrfi import SyntheticRFI
from .metricscalculator import MetricsCalculator
from .plotter import Plotter
from .utilities import Utilities

class SamRFI:
    def __init__(self, sam, vis=False, time_int=2000, points=2000, device='cuda', dir_path=None):
        self.rfi_table = pd.DataFrame(columns=['rfi_type', 'amplitude', 'center_freq', 'bandwidth', 'duty_cycle', 'time_period', 'time_offset'])
        
        self.spectrograph = np.zeros((time_int, points))
        self.flags = None
        self.noise = None
        self.frequencies = None
        self.time_int = time_int
        self.points = points
        self.pad_width = None
        self.min_freq = None
        self.test_image = None
        self.mean = None

        sam_checkpoint = str(sam)
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.predictor = SamPredictor(sam)

        ################
        #Directory logic
        ################

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