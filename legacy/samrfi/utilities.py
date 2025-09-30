import numpy as np
from patchify import patchify
from skimage.feature import peak_local_max


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
    """Need to minimize the score that it returns"""

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
        dd = (pflag - 70.0) / 10.0

    res = np.sqrt(aa**2 + bb**2 + cc * 2 + dd * 2)

    if fdiff < 0.0:
        res = res + res + 10.0

    # print("Score : ", res)

    return res


def printstats(arr):
    if len(arr) == 0:
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


def getvals(tb, col="DATA", vis="", spw="", scan=""):

    # print("SPW:", spw, "DDID:", ddid)

    tb.open(vis)
    if spw and scan:
        tb.open(vis + "/DATA_DESCRIPTION")
        spwids = tb.getcol("SPECTRAL_WINDOW_ID")
        ddid = str(np.where(spwids == eval(spw))[0][0])
        tb1 = tb.query(
            "SCAN_NUMBER==" + scan + " && DATA_DESC_ID==" + ddid + " && ANTENNA1=1 && ANTENNA2=2"
        )
    else:
        tb1 = tb.query("ANTENNA1=1 && ANTENNA2=2")
    dat = tb1.getcol(col)
    tb1.close()
    tb.close()
    return dat


def four_rotations(rfi_antenna_data):

    rfi_combined = []

    for baseline in rfi_antenna_data:
        for per_pol in baseline:
            rfi_combined.append(per_pol)
            rfi_combined.append(np.flip(per_pol, axis=0))
            rfi_combined.append(per_pol.T)
            rfi_combined.append(np.flip(per_pol.T, axis=0))

    return rfi_combined


def create_patchify_patches(
    rfi_combined,
    patch_size=128,
):
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
    padded_image = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)

    # Create patches
    patches = []
    for i in range(0, padded_image.shape[0], patch_size):
        for j in range(0, padded_image.shape[1], patch_size):
            patch = padded_image[i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches, image.shape, padded_image.shape


def reconstruct_image(patches, original_shape, padded_shape, patch_size=256):
    # ChatGPT assisted with this function
    # Create an empty array to hold the reconstructed image
    reconstructed_image = np.zeros(padded_shape)

    patch_index = 0
    for i in range(0, padded_shape[0], patch_size):
        for j in range(0, padded_shape[1], patch_size):
            reconstructed_image[i : i + patch_size, j : j + patch_size] = patches[patch_index]
            patch_index += 1

    # Remove the padding to get the original image size
    return reconstructed_image[: original_shape[0], : original_shape[1]]


def compute_start_indices(size, window_size, stride):
    if size <= window_size:
        return [0]
    else:
        starts = list(range(0, size - window_size + 1, stride))
        if starts[-1] + window_size < size:
            starts.append(size - window_size)
        return starts


def extract_patches(array, window_size=256, overlap=128):
    """
    Extract overlapping patches from a 2D NumPy array.

    Parameters:
    - array: Input 2D NumPy array.
    - window_size: Size of the window (patch).
    - overlap: Number of pixels to overlap between patches.

    Returns:
    - patches_array: NumPy array of patches with shape (num_patches, window_size, window_size).
    - positions: List of (y, x) positions where each patch was extracted.
    """
    stride = window_size - overlap
    height, width = array.shape
    x_starts = compute_start_indices(width, window_size, stride)
    y_starts = compute_start_indices(height, window_size, stride)

    patches = []
    positions = []
    for y in y_starts:
        for x in x_starts:
            patch = array[y : y + window_size, x : x + window_size]
            patches.append(patch)
            positions.append((y, x))
    patches_array = np.stack(patches)
    return patches_array, positions


def reconstruct_from_patches(patches, positions, array_shape, window_size=256, overlap=128):
    """
    Reconstruct the original array from patches, applying logical AND in overlapping regions.

    Parameters:
    - patches: NumPy array of patches with shape (num_patches, window_size, window_size).
    - positions: List of (y, x) positions where each patch should be placed.
    - array_shape: Shape of the original array (height, width).
    - window_size: Size of the window (patch).
    - overlap: Number of pixels that overlap between patches.

    Returns:
    - output_array: Reconstructed 2D NumPy array.
    """
    # Initialize the output array to all True values
    output_array = np.ones(array_shape, dtype=bool)
    count_array = np.zeros(array_shape, dtype=int)

    for idx, (y, x) in enumerate(positions):
        patch = patches[idx]
        # Create a mask for the current patch
        patch_mask = np.zeros(array_shape, dtype=bool)
        patch_mask[y : y + window_size, x : x + window_size] = True

        # Update the count of overlaps
        count_array[y : y + window_size, x : x + window_size] += 1

        # Apply logical AND operation
        output_array[y : y + window_size, x : x + window_size] = np.logical_and(
            output_array[y : y + window_size, x : x + window_size], patch
        )

    # Optional: You might want to consider only areas where the overlap count is more than 1
    # For example, setting areas with no overlap to the original detection values
    # This can be customized based on your specific requirements

    return output_array


def extract_patches_with_context(array, patch_size=192, context_size=256):
    """
    Extract patches from a 2D NumPy array and expand each patch with extra context.

    Parameters:
    - array: Input 2D NumPy array.
    - patch_size: Size of the central patch (192x192).
    - context_size: Size of the patch including context (256x256).

    Returns:
    - patches_array: NumPy array of patches with shape (num_patches, context_size, context_size).
    - positions: List of (y, x) positions where each patch was extracted.
    """
    height, width = array.shape
    stride = patch_size  # Non-overlapping patches
    x_starts = compute_start_indices(width, patch_size, stride)
    y_starts = compute_start_indices(height, patch_size, stride)

    patches = []
    positions = []
    for y in y_starts:
        for x in x_starts:
            # Coordinates for the central patch
            y_center = y + patch_size // 2
            x_center = x + patch_size // 2

            # Coordinates for the context patch
            y_start = y_center - context_size // 2
            y_end = y_start + context_size
            x_start = x_center - context_size // 2
            x_end = x_start + context_size

            # Initialize the context patch
            context_patch = np.zeros((context_size, context_size), dtype=array.dtype)

            # Calculate the overlap between the array and the context patch
            array_y_start = max(0, y_start)
            array_y_end = min(height, y_end)
            array_x_start = max(0, x_start)
            array_x_end = min(width, x_end)

            context_y_start = array_y_start - y_start
            context_y_end = context_y_start + (array_y_end - array_y_start)
            context_x_start = array_x_start - x_start
            context_x_end = context_x_start + (array_x_end - array_x_start)

            # Copy the data from the array to the context patch
            context_patch[context_y_start:context_y_end, context_x_start:context_x_end] = array[
                array_y_start:array_y_end, array_x_start:array_x_end
            ]

            # Fill the missing context with random patches
            missing_mask = context_patch == 0
            num_missing = np.sum(missing_mask)
            if num_missing > 0:
                # Sample random positions from the array to fill the missing context
                random_indices = np.random.randint(0, height * width, size=num_missing)
                random_values = array.flatten()[random_indices]
                context_patch[missing_mask] = random_values

            patches.append(context_patch)
            positions.append((y, x))
    patches_array = np.stack(patches)
    return patches_array, positions


def crop_patches(patches, crop_size=192):
    """
    Crop the central region from each patch.

    Parameters:
    - patches: NumPy array of patches with shape (num_patches, context_size, context_size).
    - crop_size: Size of the central crop (192x192).

    Returns:
    - cropped_patches: NumPy array of cropped patches with shape (num_patches, crop_size, crop_size).
    """
    context_size = patches.shape[1]
    start = (context_size - crop_size) // 2
    end = start + crop_size
    cropped_patches = patches[:, start:end, start:end]
    return cropped_patches


def reconstruct_from_patches_adding(patches, positions, array_shape, patch_size=192):
    """
    Reconstruct the original array from cropped patches.

    Parameters:
    - patches: NumPy array of cropped patches with shape (num_patches, patch_size, patch_size).
    - positions: List of (y, x) positions where each patch should be placed.
    - array_shape: Shape of the original array (height, width).
    - patch_size: Size of the central patch (192x192).

    Returns:
    - output_array: Reconstructed 2D NumPy array.
    """
    output_array = np.zeros(array_shape, dtype=patches.dtype)
    for idx, (y, x) in enumerate(positions):
        output_array[y : y + patch_size, x : x + patch_size] = patches[idx]
    return output_array


# Adapted from https://www.datacamp.com/tutorial/sam2-fine-tuning
def get_points(mask, num_points):  # Sample points inside the input mask
    points = []
    coords = np.argwhere(mask > 0)

    for _ in range(num_points):
        y, x = coords[np.random.randint(len(coords))]
        points.append([x, y])
    return np.array(points)


def get_peak_points(
    image,
    min_distance=16,
):
    """
    Return up to num_points local maxima coordinates from the image in shape (N, 2).
    Utilizes skimage.feature.peak_local_max.
    """
    # peaks is an array of (row, col)
    peaks = peak_local_max(
        image,
        min_distance=min_distance,
    )
    if len(peaks) == 0:
        return np.empty((0, 2), dtype=int)  # No peaks found

    return peaks
