"""
Preprocessor - Convert waterfall data to training-ready patches

Clean rewrite of RFIDataset preprocessing pipeline.
"""

import numpy as np
from scipy import stats
from patchify import patchify
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


# Standalone functions for multiprocessing (must be picklable)
def _patchify_single_waterfall(waterfall, patch_size):
    """
    Patchify a single waterfall into patches.

    Args:
        waterfall: 2D array (channels, times)
        patch_size: Size of square patches

    Returns:
        List of patches from this waterfall
    """
    patches = patchify(waterfall, (patch_size, patch_size), step=patch_size)

    # Extract patches
    patch_list = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch_list.append(patches[i, j])

    return patch_list


def _compute_mad_flag_single_patch(patch, sigma):
    """
    Compute MAD-based flag for a single patch.

    Args:
        patch: 2D array (patch_size, patch_size)
        sigma: Threshold in units of MAD

    Returns:
        Boolean flag array
    """
    mad = stats.median_abs_deviation(patch, axis=None, nan_policy="omit")
    median = np.nanmedian(patch)

    upper_thresh = median + (mad * sigma)
    lower_thresh = median - (mad * sigma)

    flag = (patch > upper_thresh) | (patch < lower_thresh)
    return flag


class Preprocessor:
    """
    Preprocess waterfall data into training patches.

    Pipeline:
        1. Four-way rotation augmentation
        2. Patchify into fixed-size patches
        3. Normalize before stretch (optional, configurable)
        4. Apply stretch (optional: "SQRT", "LOG10", or None)
        5. Normalize after stretch (optional, configurable)
        6. Generate or use flags (flags never transformed, only patchified)
        7. Remove blank patches
        8. Shuffle patches
        9. Create HuggingFace Dataset

    Usage:
        >>> # Real data: normalize, no stretch
        >>> preprocessor = Preprocessor(data, flags=None)
        >>> dataset = preprocessor.create_dataset(
        ...     patch_size=128,
        ...     normalize_before_stretch=True,
        ...     stretch=None,
        ...     normalize_after_stretch=False
        ... )

        >>> # Synthetic data: preserve physical scales
        >>> preprocessor = Preprocessor(data, flags=exact_masks)
        >>> dataset = preprocessor.create_dataset(
        ...     patch_size=128,
        ...     normalize_before_stretch=False,
        ...     stretch=None,
        ...     normalize_after_stretch=False,
        ...     use_custom_flags=True
        ... )
    """

    def __init__(self, data, flags=None):
        """
        Initialize preprocessor.

        Args:
            data: Waterfall data, shape (baselines, pols, channels, times) or (pols, channels, times)
            flags: Optional flag array (same shape as data). If None, will generate using MAD.
        """
        # Handle both (baselines, pols, ch, time) and (pols, ch, time) shapes
        if data.ndim == 4:
            # Has baselines dimension
            self.data = data
        elif data.ndim == 3:
            # Single baseline, add dimension
            self.data = data[np.newaxis, ...]
        else:
            raise ValueError(f"Data must be 3D or 4D, got shape {data.shape}")

        self.flags = flags
        self.patches = None
        self.patch_flags = None
        self.dataset = None

    def create_dataset(
        self,
        patch_size=128,
        stretch=None,
        flag_sigma=5,
        use_custom_flags=True,
        num_patches=None,
        normalize_before_stretch=True,
        normalize_after_stretch=False,
        num_workers=4,
    ):
        """
        Create HuggingFace Dataset from waterfall data.

        Args:
            patch_size: Size of square patches (default 128)
            stretch: Stretch function - "SQRT", "LOG10", or None (default None)
            flag_sigma: Sigma threshold for MAD flagging (if not using custom flags)
            use_custom_flags: If True and flags provided, use them. Otherwise generate with MAD.
            num_patches: Limit number of patches (default: all)
            normalize_before_stretch: Divide by median before stretching (default True)
            normalize_after_stretch: Divide by median after stretching (default False)
            num_workers: Number of parallel workers for preprocessing (0 for sequential, -1 for all cores, default 4)

        Returns:
            HuggingFace Dataset with 'image' and 'label' fields
        """
        print(f"\n[Preprocessor] Creating dataset...")
        print(f"  Input shape: {self.data.shape}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Normalize before stretch: {normalize_before_stretch}")
        print(f"  Stretch: {stretch if stretch else 'None'}")
        print(f"  Normalize after stretch: {normalize_after_stretch}")
        print(f"  Parallel workers: {num_workers if num_workers else 'sequential'}")

        # Step 1: Augmentation (4-way rotation)
        print("  [1/7] Applying 4-way rotation augmentation...")
        augmented_data = self._four_rotations(self.data)
        print(f"    Augmented to {len(augmented_data)} waterfalls")

        if use_custom_flags and self.flags is not None:
            augmented_flags = self._four_rotations(self.flags)
        else:
            augmented_flags = None

        # Step 2: Patchify (or skip if patch_size >= image dimensions)
        waterfall_shape = augmented_data[0].shape
        if patch_size >= min(waterfall_shape):
            # Skip patching - use full waterfalls
            print(f"  [2/7] Skipping patchification (patch_size={patch_size} >= image size {waterfall_shape})...")
            self.patches = np.array(augmented_data)
            if augmented_flags is not None:
                augmented_flags = np.array(augmented_flags)
            print(f"    Using {len(self.patches)} full waterfalls")
        else:
            # Apply patching
            print(f"  [2/7] Patchifying into {patch_size}x{patch_size} patches...")
            self.patches = self._create_patches(augmented_data, patch_size, num_workers=num_workers)
            if augmented_flags is not None:
                augmented_flags = self._create_patches(augmented_flags, patch_size, num_workers=num_workers)
            print(f"    Created {len(self.patches)} patches")

        # Step 3: Normalize before stretch (optional)
        if normalize_before_stretch:
            print("  [3/7] Normalizing patches (before stretch)...")
            self.patches = self._normalize(self.patches)
        else:
            print("  [3/7] Skipping normalization before stretch")

        # Step 4: Apply stretch (optional)
        if stretch:
            print(f"  [4/7] Applying {stretch} stretch...")
            self.patches = self._apply_stretch(self.patches, stretch)
        else:
            print("  [4/7] Skipping stretch")

        # Step 5: Normalize after stretch (optional)
        if normalize_after_stretch:
            print("  [5/7] Normalizing patches (after stretch)...")
            self.patches = self._normalize(self.patches)
        else:
            print("  [5/7] Skipping normalization after stretch")

        # Step 6: Generate or use flags
        # IMPORTANT: Flags are NEVER transformed, only rotated/patchified to stay aligned
        if use_custom_flags and augmented_flags is not None:
            print("  [6/7] Using custom flags (respecting incoming flags)...")
            # Flags already patchified (or converted to array) in Step 2
            self.patch_flags = augmented_flags
        else:
            print(f"  [6/7] Generating MAD flags from processed patches (sigma={flag_sigma})...")
            self.patch_flags = self._generate_mad_flags(self.patches, flag_sigma, num_workers=num_workers)

        print(f"    Flag patches: {self.patch_flags.shape}")

        # Step 7: Remove blank patches
        print("  [7/7] Removing blank patches...")
        initial_count = len(self.patches)
        self._remove_blank_patches()
        removed = initial_count - len(self.patches)
        print(f"    Removed {removed} blank patches, {len(self.patches)} remain")

        # Step 8: Shuffle
        print("  [8/8] Shuffling patches...")
        self._shuffle()

        # Limit number of patches if requested
        if num_patches and num_patches < len(self.patches):
            self.patches = self.patches[:num_patches]
            self.patch_flags = self.patch_flags[:num_patches]
            print(f"    Limited to {num_patches} patches")

        # Create HuggingFace Dataset
        print("\n  Creating HuggingFace Dataset...")
        dataset_dict = {
            "image": [Image.fromarray(img).convert("RGB") for img in self.patches],
            "label": [Image.fromarray(mask) for mask in self.patch_flags],
        }

        self.dataset = Dataset.from_dict(dataset_dict)
        print(f"  ✓ Dataset ready: {len(self.dataset)} samples")

        return self.dataset

    def _four_rotations(self, data):
        """
        Apply 4-way rotation augmentation.

        For each waterfall:
            - Original
            - Flip vertically
            - Transpose
            - Transpose + flip vertically

        Args:
            data: Array of shape (baselines, pols, channels, times)

        Returns:
            List of augmented waterfalls (each is 2D)
        """
        augmented = []

        for baseline in data:
            for pol in baseline:
                # Original
                augmented.append(pol)
                # Flip vertical
                augmented.append(np.flip(pol, axis=0))
                # Transpose
                augmented.append(pol.T)
                # Transpose + flip
                augmented.append(np.flip(pol.T, axis=0))

        return augmented

    def _create_patches(self, data_list, patch_size, num_workers=None):
        """
        Create patches from list of 2D arrays.

        Args:
            data_list: List of 2D arrays
            patch_size: Size of square patches
            num_workers: Number of parallel workers (None/0 for sequential, -1 for all cores)

        Returns:
            Array of patches, shape (num_patches, patch_size, patch_size)
        """
        if num_workers and num_workers != 0:
            # Parallel processing
            n_workers = cpu_count() if num_workers == -1 else num_workers

            with Pool(n_workers) as pool:
                patch_func = partial(_patchify_single_waterfall, patch_size=patch_size)
                results = pool.map(patch_func, data_list)

            # Flatten results
            all_patches = [patch for waterfall_patches in results for patch in waterfall_patches]
        else:
            # Sequential processing (original code)
            all_patches = []
            for waterfall in data_list:
                # Patchify this waterfall
                patches = patchify(waterfall, (patch_size, patch_size), step=patch_size)

                # Extract patches
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        all_patches.append(patches[i, j])

        return np.array(all_patches)

    def _normalize(self, patches):
        """
        Normalize patches by dividing by median.

        Args:
            patches: Array of patches

        Returns:
            Normalized patches
        """
        normalized = []

        for patch in patches:
            median = np.nanmedian(patch)
            if median > 0:
                normalized_patch = patch / median
            else:
                normalized_patch = patch
            normalized.append(normalized_patch)

        return np.array(normalized)

    def _apply_stretch(self, patches, stretch):
        """
        Apply stretch function to patches.

        Args:
            patches: Array of patches
            stretch: 'SQRT' or 'LOG10'

        Returns:
            Stretched patches
        """
        if stretch == "SQRT":
            stretch_func = np.sqrt
        elif stretch == "LOG10":
            stretch_func = np.log10
        else:
            raise ValueError(f"Invalid stretch '{stretch}'. Use 'SQRT' or 'LOG10'")

        stretched = []

        for patch in patches:
            # Apply stretch to absolute values
            stretched_patch = stretch_func(np.abs(patch))

            # Handle infinities
            finite_data = stretched_patch[np.isfinite(stretched_patch)]
            if len(finite_data) > 0:
                mad = stats.median_abs_deviation(finite_data, nan_policy="omit")
                stretched_patch[np.isinf(stretched_patch)] = mad
            else:
                stretched_patch[np.isinf(stretched_patch)] = 0

            stretched.append(stretched_patch)

        return np.array(stretched)

    def _generate_mad_flags(self, patches, sigma, num_workers=None):
        """
        Generate flags using MAD (Median Absolute Deviation).

        Args:
            patches: Array of patches
            sigma: Threshold in units of MAD
            num_workers: Number of parallel workers (None/0 for sequential, -1 for all cores)

        Returns:
            Boolean flag array
        """
        if num_workers and num_workers != 0:
            # Parallel processing
            n_workers = cpu_count() if num_workers == -1 else num_workers

            with Pool(n_workers) as pool:
                flag_func = partial(_compute_mad_flag_single_patch, sigma=sigma)
                flags = pool.map(flag_func, patches)
        else:
            # Sequential processing (original code)
            flags = []

            for patch in patches:
                mad = stats.median_abs_deviation(patch, axis=None, nan_policy="omit")
                median = np.nanmedian(patch)

                upper_thresh = median + (mad * sigma)
                lower_thresh = median - (mad * sigma)

                flag = (patch > upper_thresh) | (patch < lower_thresh)
                flags.append(flag)

        return np.array(flags, dtype=bool)

    def _remove_blank_patches(self):
        """Remove patches where flag mask is entirely False."""
        # Find patches with at least one flag
        has_flags = np.array([flags.any() for flags in self.patch_flags])

        # Filter
        self.patches = self.patches[has_flags]
        self.patch_flags = self.patch_flags[has_flags]

    def _shuffle(self):
        """Shuffle patches and flags in unison."""
        indices = np.random.permutation(len(self.patches))

        self.patches = self.patches[indices]
        self.patch_flags = self.patch_flags[indices]
