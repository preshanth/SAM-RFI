"""
Adaptive patching module for measurement sets with arbitrary dimensions.

This module provides utilities for handling measurement set data that may not be
evenly divisible by the patch size used during training. It implements padding
and cropping strategies to enable SAM-RFI inference on data of any dimensions.

Classes
-------
AdaptivePatcher
    Adaptive patching for measurement sets with arbitrary dimensions using
    padding strategies.

Functions
---------
check_ms_compatibility
    Check if measurement set dimensions are compatible with a given patch size.

Examples
--------
>>> from samrfi.data.adaptive_patcher import AdaptivePatcher
>>> data_shape = (100, 2, 900, 1500)  # baselines, pols, channels, times
>>> patcher = AdaptivePatcher(data_shape, patch_size=1024)
>>> padded_data = patcher.pad_data(data)
>>> # ... perform inference ...
>>> flags = patcher.crop_flags(predicted_flags)

Notes
-----
The adaptive patcher supports three padding modes:
- 'reflect': Mirror padding at boundaries (default)
- 'edge': Extend edge values
- 'constant': Zero padding

See Also
--------
samrfi.data.ms_loader.MSLoader : Load measurement set data
"""

import numpy as np
from typing import Dict, Tuple, Any


class AdaptivePatcher:
    """
    Adaptive patching for measurement sets with arbitrary dimensions.

    This class handles measurement set data that may not be evenly divisible
    by the patch size used during training. It pads the data to the next
    multiple of patch_size, tracks the padding for later removal, and supports
    multiple padding strategies.

    Parameters
    ----------
    data_shape : tuple of int
        Original data shape as (baselines, pols, channels, times).
    patch_size : int, default=1024
        Target patch size in pixels. Must match the patch size used during
        model training.
    padding_mode : {'reflect', 'edge', 'constant'}, default='reflect'
        Padding strategy to use:
        - 'reflect': Mirror padding at boundaries
        - 'edge': Extend edge values
        - 'constant': Zero padding

    Attributes
    ----------
    original_shape : tuple of int
        Original unpadded data shape.
    patch_size : int
        Target patch size.
    padding_mode : str
        Selected padding strategy.
    baselines : int
        Number of baselines in the data.
    pols : int
        Number of polarizations in the data.
    channels : int
        Original number of channels.
    times : int
        Original number of time samples.
    padded_channels : int
        Number of channels after padding.
    padded_times : int
        Number of time samples after padding.
    pad_channels : int
        Number of padding channels added.
    pad_times : int
        Number of padding time samples added.
    num_patches_h : int
        Number of patches along the channel (height) dimension.
    num_patches_w : int
        Number of patches along the time (width) dimension.
    total_patches_per_baseline_pol : int
        Total number of patches per baseline-polarization combination.

    Examples
    --------
    >>> import numpy as np
    >>> from samrfi.data.adaptive_patcher import AdaptivePatcher
    >>> # Create sample data
    >>> data = np.random.randn(10, 2, 900, 1500) + 1j * np.random.randn(10, 2, 900, 1500)
    >>> # Initialize patcher
    >>> patcher = AdaptivePatcher(data.shape, patch_size=1024)
    >>> # Pad data for inference
    >>> padded_data = patcher.pad_data(data)
    >>> print(padded_data.shape)
    (10, 2, 1024, 2048)
    >>> # After inference, crop flags back to original size
    >>> flags = np.random.randint(0, 2, size=padded_data.shape, dtype=np.uint8)
    >>> cropped_flags = patcher.crop_flags(flags)
    >>> print(cropped_flags.shape)
    (10, 2, 900, 1500)

    Notes
    -----
    The patcher uses symmetric padding strategies to minimize artifacts at
    patch boundaries. For reflective padding, values are mirrored at the
    boundary. This works well for radio astronomy data where edge effects
    are common.

    See Also
    --------
    check_ms_compatibility : Check measurement set compatibility with patch size
    """

    def __init__(
        self, data_shape: Tuple[int, ...], patch_size: int = 1024, padding_mode: str = "reflect"
    ) -> None:
        """
        Initialize adaptive patcher.

        Parameters
        ----------
        data_shape : tuple of int
            Original data shape (baselines, pols, channels, times).
        patch_size : int, default=1024
            Target patch size (must match training).
        padding_mode : {'reflect', 'edge', 'constant'}, default='reflect'
            Padding strategy: 'reflect', 'edge', or 'constant'.
        """
        self.original_shape = data_shape
        self.patch_size = patch_size
        self.padding_mode = padding_mode

        # Calculate required padding
        self.baselines, self.pols, self.channels, self.times = data_shape

        # Compute padded dimensions
        self.padded_channels = self._next_multiple(self.channels, patch_size)
        self.padded_times = self._next_multiple(self.times, patch_size)

        # Padding amounts
        self.pad_channels = self.padded_channels - self.channels
        self.pad_times = self.padded_times - self.times

        # Number of patches
        self.num_patches_h = self.padded_channels // patch_size
        self.num_patches_w = self.padded_times // patch_size
        self.total_patches_per_baseline_pol = self.num_patches_h * self.num_patches_w

        print("\nAdaptive Patching Configuration:")
        print(f"  Original shape:   {data_shape}")
        print(f"  Patch size:       {patch_size}")
        print(f"  Channels: {self.channels} → {self.padded_channels} (+{self.pad_channels})")
        print(f"  Times:    {self.times} → {self.padded_times} (+{self.pad_times})")
        print(
            f"  Patches:  {self.num_patches_h} × {self.num_patches_w} = "
            f"{self.num_patches_h * self.num_patches_w} per baseline/pol"
        )

    @staticmethod
    def _next_multiple(value: int, multiple: int) -> int:
        """
        Round value up to the next multiple.

        Parameters
        ----------
        value : int
            Input value to round up.
        multiple : int
            Multiple to round up to.

        Returns
        -------
        int
            Smallest multiple of `multiple` that is >= `value`.

        Examples
        --------
        >>> AdaptivePatcher._next_multiple(900, 1024)
        1024
        >>> AdaptivePatcher._next_multiple(1500, 1024)
        2048
        """
        return ((value + multiple - 1) // multiple) * multiple

    def pad_data(self, data: np.ndarray) -> np.ndarray:
        """
        Pad data to match patch_size requirements.

        Pads the channel and time dimensions to the next multiple of patch_size
        using the configured padding mode. Baseline and polarization dimensions
        are not padded.

        Parameters
        ----------
        data : np.ndarray
            Input data with shape (baselines, pols, channels, times).
            Can be complex-valued or real-valued.

        Returns
        -------
        np.ndarray
            Padded data with shape (baselines, pols, padded_channels, padded_times).

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.randn(10, 2, 900, 1500)
        >>> patcher = AdaptivePatcher(data.shape, patch_size=1024)
        >>> padded = patcher.pad_data(data)
        >>> padded.shape
        (10, 2, 1024, 2048)

        Notes
        -----
        If no padding is needed (data already divisible by patch_size),
        returns the original data without copying.
        """
        if self.pad_channels == 0 and self.pad_times == 0:
            return data  # No padding needed

        # Padding specification: ((before, after), ...)
        pad_width = (
            (0, 0),  # baselines: no padding
            (0, 0),  # pols: no padding
            (0, self.pad_channels),  # channels: pad at end
            (0, self.pad_times),  # times: pad at end
        )

        if self.padding_mode == "constant":
            padded = np.pad(data, pad_width, mode="constant", constant_values=0)
        else:
            padded = np.pad(data, pad_width, mode=self.padding_mode)

        return padded

    def crop_flags(self, flags: np.ndarray) -> np.ndarray:
        """
        Crop padded flags back to original dimensions.

        Removes padding from the channel and time dimensions to restore the
        original data shape.

        Parameters
        ----------
        flags : np.ndarray
            Padded flags with shape (baselines, pols, padded_channels, padded_times).

        Returns
        -------
        np.ndarray
            Cropped flags with shape matching original_shape.

        Examples
        --------
        >>> import numpy as np
        >>> patcher = AdaptivePatcher((10, 2, 900, 1500), patch_size=1024)
        >>> padded_flags = np.random.randint(0, 2, (10, 2, 1024, 2048), dtype=np.uint8)
        >>> cropped = patcher.crop_flags(padded_flags)
        >>> cropped.shape
        (10, 2, 900, 1500)

        Notes
        -----
        This method should be called after inference to remove the padding
        that was added by pad_data().
        """
        return flags[:, :, : self.channels, : self.times]

    def get_patch_info(self) -> Dict[str, Any]:
        """
        Get patching configuration information.

        Returns
        -------
        dict
            Dictionary containing:
            - original_shape : tuple
                Original data shape (baselines, pols, channels, times)
            - padded_shape : tuple
                Padded data shape
            - patch_size : int
                Patch size used
            - num_patches_h : int
                Number of patches along channel dimension
            - num_patches_w : int
                Number of patches along time dimension
            - total_patches : int
                Total number of patches across all baselines and polarizations
            - padding : dict
                Padding amounts {'channels': int, 'times': int}

        Examples
        --------
        >>> patcher = AdaptivePatcher((10, 2, 900, 1500), patch_size=1024)
        >>> info = patcher.get_patch_info()
        >>> info['total_patches']
        20
        >>> info['padding']
        {'channels': 124, 'times': 548}
        """
        return {
            "original_shape": self.original_shape,
            "padded_shape": (self.baselines, self.pols, self.padded_channels, self.padded_times),
            "patch_size": self.patch_size,
            "num_patches_h": self.num_patches_h,
            "num_patches_w": self.num_patches_w,
            "total_patches": self.baselines * self.pols * self.num_patches_h * self.num_patches_w,
            "padding": {"channels": self.pad_channels, "times": self.pad_times},
        }


def check_ms_compatibility(ms_path: str, patch_size: int = 1024) -> Dict[str, Any]:
    """
    Check if measurement set dimensions are compatible with patch size.

    Analyzes the measurement set to determine if its dimensions are evenly
    divisible by the specified patch size, and calculates required padding
    if not.

    Parameters
    ----------
    ms_path : str
        Path to measurement set (.ms directory).
    patch_size : int, default=1024
        Target patch size to check compatibility against.

    Returns
    -------
    dict
        Dictionary containing compatibility information:
        - channels : int
            Number of channels in the measurement set
        - times : int
            Number of time samples in the measurement set
        - patch_size : int
            Patch size used for compatibility check
        - channels_divisible : bool
            Whether channels dimension is evenly divisible
        - times_divisible : bool
            Whether times dimension is evenly divisible
        - fully_compatible : bool
            Whether both dimensions are divisible (no padding needed)
        - padding_required : dict
            Required padding amounts {'channels': int, 'times': int}
        - recommendation : str
            Human-readable recommendation message

    Examples
    --------
    >>> from samrfi.data.adaptive_patcher import check_ms_compatibility
    >>> info = check_ms_compatibility('my_data.ms', patch_size=1024)
    >>> if info['fully_compatible']:
    ...     print("No padding needed!")
    >>> else:
    ...     print(f"Padding required: {info['padding_required']}")
    ...     print(info['recommendation'])

    Notes
    -----
    This function provides guidance on whether a measurement set can be
    processed without padding, or if adaptive patching will be needed.
    If padding exceeds 10% of the data, consider retraining with a smaller
    patch size for better efficiency.

    See Also
    --------
    AdaptivePatcher : Adaptive patching for arbitrary dimensions
    """
    from samrfi.data.ms_loader import MSLoader

    loader = MSLoader(ms_path)

    # Get MS dimensions without loading full data
    num_channels = loader.channels_per_spw[0] * len(loader.channels_per_spw)
    num_times = loader.num_times

    # Check divisibility
    channels_divisible = num_channels % patch_size == 0
    times_divisible = num_times % patch_size == 0

    # Padding required
    pad_channels = (
        0
        if channels_divisible
        else AdaptivePatcher._next_multiple(num_channels, patch_size) - num_channels
    )
    pad_times = (
        0 if times_divisible else AdaptivePatcher._next_multiple(num_times, patch_size) - num_times
    )

    return {
        "channels": num_channels,
        "times": num_times,
        "patch_size": patch_size,
        "channels_divisible": channels_divisible,
        "times_divisible": times_divisible,
        "fully_compatible": channels_divisible and times_divisible,
        "padding_required": {"channels": pad_channels, "times": pad_times},
        "recommendation": _get_recommendation(
            channels_divisible, times_divisible, pad_channels, pad_times, patch_size
        ),
    }


def _get_recommendation(ch_div: bool, t_div: bool, pad_ch: int, pad_t: int, patch_size: int) -> str:
    """
    Generate human-readable recommendation message for padding requirements.

    Parameters
    ----------
    ch_div : bool
        Whether channels dimension is evenly divisible by patch_size.
    t_div : bool
        Whether times dimension is evenly divisible by patch_size.
    pad_ch : int
        Number of padding channels required.
    pad_t : int
        Number of padding time samples required.
    patch_size : int
        Patch size being used.

    Returns
    -------
    str
        Recommendation message indicating whether padding is needed and
        if it exceeds 10% threshold.

    Examples
    --------
    >>> _get_recommendation(True, True, 0, 0, 1024)
    '✓ Fully compatible - no padding needed'
    >>> _get_recommendation(False, False, 100, 200, 1024)
    '⚠ Padding required: +100 channels +200 times (<10% padding - acceptable)'
    """
    if ch_div and t_div:
        return "✓ Fully compatible - no padding needed"

    msg = "⚠ Padding required: "
    if not ch_div:
        msg += f"+{pad_ch} channels "
    if not t_div:
        msg += f"+{pad_t} times "

    pad_pct_ch = (pad_ch / (patch_size - pad_ch)) * 100 if pad_ch > 0 else 0
    pad_pct_t = (pad_t / (patch_size - pad_t)) * 100 if pad_t > 0 else 0

    if pad_pct_ch > 10 or pad_pct_t > 10:
        msg += "(>10% padding - consider retraining with smaller patch_size)"
    else:
        msg += "(<10% padding - acceptable)"

    return msg
