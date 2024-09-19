# SamRFI - AI RFI Segmentation and CASA Flagging Pipeline 

![](https://github.com/preshanth/SAM-RFI/blob/main/samrfi.png)

-------------------------------------------------------------------------------------

**Authors:** Derod Deal (dealderod@gmail.com), Preshanth Jagannathana (pjaganna@nrao.edu)

`SamRFI` is a python package that ultilizes Meta's Segment Anything Model (SAM) for Radio Frequency Interference (RFI) segmentation.


## Statement of need

Radio Frequency Interference (RFI) is an obstacle to radio astronomy, which lowers sensitivity and increases noise in datasets. Several mitigation methods have been developed to decrease the amount of RFI in measurement sets by flagging and removing data with RFI. These algorithms include TFCrop, RFFlag, and aoflagger. Although each algorithm and software is nominally automatic, it requires substantial user input to select and identify interference to flag. `SamRFI` provides RFI segmentation tools that can be used to flag RFI using Meta’s Segment Anything Model (SAM). This pipeline also includes tools for loading measurement sets, generating synthetic waterfall plots, retraining custom `SamRFI` models, and measuring AI performance of flagging using metrics.

## Installation

To install `SamRFI`, use `pip install`.

```
pip install samrfi
```

## Using the library

Get started with `SamRFI` with a few lines of code.

```python

from samrfi import RadioRFI

ms_path = '/home/gpuhost001/ddeal/RFI-AI/one_antenna_3C219_sqrt.ms'

datarfi = RadioRFI(vis=ms_path)
datarfi.load(ant_i=1)

```

Using the `RadioRFI` class, `SamRFI` loads in data from a specified measurement set as time-frequency spectrograms. We can also plot the waterfall plots using the `Plotter` class:

```python

datarfi.plotter.plot(mode='DATA', baseline=0, polarization=0)

```

![](https://github.com/preshanth/SAM-RFI/blob/main/plots/samrfi_3C219.png)

The data is loaded per baseline per polarization for each waterfall plot. The y-axis represents the time steps while the x-axis represents the number of channels (all spectral window channels attached together).


After the data is loaded, you can immediately run a retrained `SamRFI` model to segment RFI and generate flags.

```python
from samrfi import RFIModels

sam_checkpoint = "/home/gpuhost001/ddeal/RFI-AI/models/sam_vit_h_4b8939.pth"
sam_type = "vit_h"

model_path = "/home/gpuhost001/ddeal/RFI-AI/models/derod_checkpoint_huge_calib_phase_patch_epoch40_sigma5_sqrt_custom_perpatch.pth"
model = RFIModels(sam_checkpoint, sam_type, radiorfi_instance=datarfi, device='cuda',)
model.load_model(model_path)
model.run_rfi_model(patch_run=False)

```
Using the Sigma 5 SQRT Model, a SAM model retrained off of real P-Band data, we get segmented flags we can plot using `Plotter`:

```python
datarfi.plotter.plot(mode='FLAG', baseline=0, polarization=0)
```

![](https://github.com/preshanth/SAM-RFI/blob/main/plots/samrfi_3C219_flags.png)


To save these flags to the measurement set, use `RadioRFI.save_flags`:

```
datarfi.save_flags()
```

In this pipeline, you can also generate synthetic rfi waterfall data using `SyntheticRFI`. Data from either `RadioRFI` or `SyntheticRFI` can be used to train your own models using `RFITraining`. The peformance of flagging is handled with calculating metrics using `RadioRFIMetricsCalculator` and     `SyntheticRFIMetricsCalculator`. Please visit our [readthedocs](https://sam-rfi.readthedocs.io/en/latest/) or our example notebooks for more information.

#### SamRFI notebooks

- `RadioRFI`
- `SyntheticRFI`
- `RFIModels`
- `RFITraining`
    - [Retraining SAM for RFI Segementation](https://github.com/preshanth/SAM-RFI/blob/main/notebooks/training_sammodels.ipynb)
- Metrics Classes
- `Plotter`

This software is currently under development on [GitHub](https://github.com). To report bugs or to send feature requests, send us an email or [open an issue](https://github.com/preshanth/SAM-RFI/issues) on GitHub.

## Licence and attribution

This project is under the MIT License, which can be viewed [here](https://github.com/preshanth/SAM-RFI/blob/main/LICENSE).

## Acknowledgments

We thank the National Radio Astronomy Observatory (NRAO) and the National Astronomy Consortium (NAC) for their funding and mentorship. We highlight the use of OpenAI's ChatGPT and GitHub's Copilot for the development of this software. Furthermore, we implemented code from [this notebook](https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb).
