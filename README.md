# SamRFI - AI RFI Segmentation and CASA Flagging Pipeline 

![](https://github.com/preshanth/SAM-RFI/blob/main/samrfi.png)

-------------------------------------------------------------------------------------

**Authors:** Derod Deal (dealderod@gmail.edu), Preshanth Jagannathana (pjaganna@nrao.edu)

`SamRFI` is a python package that ultilizes Meta's Segment Anything Model (SAM) for radio frequency interference (RFI) segmentation.


## Statement of need

Radio Frequency Interference (RFI) is an obstacle to radio astronomy, which lowers sensitivity and increases noise in datasets. Several mitigation methods have been developed to decrease the amount of RFI in measurement sets by flagging and removing data with RFI. These algorithms include TFCrop, RFFlag, and aoflagger. Although each algorithm and software is nominally automatic, it requires substantial user input to select and identify interference to flag. `SamRFI` provides RFI segmentation tools that can be used to flag RFI using Meta’s Segment Anything Model (SAM). This pipeline also includes tools for loading measurement sets, generating synthetic waterfall plots, retraining custom `SamRFI` models, and measuring AI performance of flagging using metrics.

## Installation

To install `SamRFI`, use `pip install`.

```
pip install samrfi
```

## Using the library

Get started with `SamRFI` with only two lines of code.