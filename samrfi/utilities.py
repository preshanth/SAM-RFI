import numpy as np
from patchify import patchify

def get_bounding_box(ground_truth_map):
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


def find_spectrograph_peaks(spectrograph, min_distance=10, threshold_abs=30):
    """
    Find peaks in the spectrograph image.

    Parameters:
        spectrograph (numpy.ndarray): The input spectrograph image.
        min_distance (int): The minimum distance between peaks. Default is 10.
        threshold_abs (int): The minimum intensity value for peaks. Default is 25.

    Returns:
        numpy.ndarray: An array of peak coordinates.
    """
    max_peaks = peak_local_max(spectrograph, min_distance=min_distance, threshold_abs=threshold_abs)
    max_peaks = max_peaks, np.ones(len(max_peaks))
    
    return max_peaks


def runtest(dat, flag):

    # plotit(dat, flag)

    # print('% Flagged : ', np.sum(flag) / (1.0 * np.prod(flag.shape)) * 100.0)

    return (np.sum(flag) / (1.0 * np.prod(flag.shape)) * 100.0), calcquality(dat, flag)


def calcquality(dat, flag):
    """ Need to minimize the score that it returns"""

    shp = dat.shape

    npts = 0
    sumsq = 0.0
    maxval = 0.0
    leftover = []
    flagged = []
    for chan in range(0, shp[0]):
        for tm in range(0, shp[1]):
            val = np.abs(dat[chan, tm])
            if flag[chan, tm] == False:
                leftover.append(val)
            else:
                flagged.append(val)

    dmax, dmean, dstd = printstats(np.abs(dat[:, :]))
    rmax, rmean, rstd = printstats(leftover)
    fmax, fmean, fstd = printstats(flagged)

    maxdev = (rmax - rmean) / rstd
    fdiff = fmean - rmean
    sdiff = fstd - rstd

    # print("Max deviation after flagging : ", maxdev)
    # print("Diff in mean of flagged and unflagged : ", fdiff)
    # print("Std after flagging : ", rstd)
    
    ## Maximum deviation from the mean is 3 sigma. => Gaussian stats. 
    ## => What's leftover is noise-like and without significant outliers.
    aa = np.abs(np.abs(maxdev) - 3.0)

    ## Flagged data has a higher mean than what is left over => flagged only RFI. Maximize the difference between the means
    bb = 1.0 / ((np.abs(fdiff) - rstd) / rstd)
    
    ## Maximize the difference between the std of the flagged and leftover data => Assumes that RFI is widely varying...
    cc = 1.0 / (np.abs(sdiff) / rstd)

    ## Overflagging is bad
    dd = 0.0
    pflag = (len(flagged) / (1.0 * shp[0] * shp[1])) * 100.0
    if pflag > 70.0:
        dd = (pflag - 70.0)/10.0
    
    res = np.sqrt(aa ** 2 + bb ** 2 + cc * 2 + dd * 2)

    if (fdiff < 0.0):
        res = res + res + 10.0

    # print("Score : ", res)

    return res


def printstats(arr):
    if (len(arr) == 0):
        return 0, 0, 1

    med = np.median(arr)
    std = np.std(arr)
    maxa = np.max(arr)
    mean = np.mean(arr)
    # print 'median : ', med
    # print 'std : ', std
    # print 'max : ', maxa
    # print 'mean : ', mean
    # print " (Max - mean)/std : ", ( maxa - mean ) / std

    return maxa, mean, std


def getvals(tb, col='DATA', vis="", spw="", scan=""):

    # print("SPW:", spw, "DDID:", ddid)

    tb.open(vis)
    if (spw and scan):
        tb.open(vis + '/DATA_DESCRIPTION')
        spwids = tb.getcol('SPECTRAL_WINDOW_ID')
        ddid = str(np.where(spwids == eval(spw))[0][0])
        tb1 = tb.query('SCAN_NUMBER==' + scan + ' && DATA_DESC_ID==' + ddid + ' && ANTENNA1=1 && ANTENNA2=2')
    else:
        tb1 = tb.query('ANTENNA1=1 && ANTENNA2=2')
    dat = tb1.getcol(col)
    tb1.close()
    tb.close()
    return dat

def four_rotations(rfi_antenna_data)

    rfi_combined = []

    for baseline in rfi_antenna_data:
        for per_pol in baseline:
            rfi_combined.append(per_pol)
            rfi_combined.append(np.flip(per_pol, axis=0))
            rfi_combined.append(per_pol.T)
            rfi_combined.append(np.flip(per_pol.T, axis=0))

    rfi_combined = np.stack(rfi_combined)

    return rfi_combined

def create_patchify_patches(rfi_combined, patch_size=128,):
    """
    Create patches from a list of images.

    Parameters:
        rfi_combined (list of numpy.ndarray): List of images to be patched.
        patch_size (int): Size of each patch. Default is 128.
        step (int): Step size for patching. Default is 128.

    Returns:
        numpy.ndarray: Array of image patches.
    """
    patch_size = patch_size
    step = patch_size

    all_img_patches = []
    for img in rfi_combined:
        large_image = img
        patches_img = patchify(large_image, (patch_size, patch_size), step=step)

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                all_img_patches.append(single_patch_img)

    patches = np.array(all_img_patches)

    return patches

def create_patches(image, patch_size=256):
    # ChatGPT assisted with this function
    # Get image dimensions
    rows, cols = image.shape
    
    # Calculate padding size
    pad_rows = (patch_size - rows % patch_size) % patch_size
    pad_cols = (patch_size - cols % patch_size) % patch_size

    # Pad the image to ensure it can be evenly divided into patches
    padded_image = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

    # Create patches
    patches = []
    for i in range(0, padded_image.shape[0], patch_size):
        for j in range(0, padded_image.shape[1], patch_size):
            patch = padded_image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    
    return patches, image.shape, padded_image.shape    

def reconstruct_image(patches, original_shape, padded_shape, patch_size=256):
    # ChatGPT assisted with this function
    # Create an empty array to hold the reconstructed image
    reconstructed_image = np.zeros(padded_shape)
    
    patch_index = 0
    for i in range(0, padded_shape[0], patch_size):
        for j in range(0, padded_shape[1], patch_size):
            reconstructed_image[i:i + patch_size, j:j + patch_size] = patches[patch_index]
            patch_index += 1
    
    # Remove the padding to get the original image size
    return reconstructed_image[:original_shape[0], :original_shape[1]]