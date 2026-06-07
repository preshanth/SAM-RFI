"""
Adaptive Patching Module for Arbitrary MS Dimensions

Handles MS data that may not be evenly divisible by patch_size,
using padding and cropping strategies to enable SAM-RFI inference.
"""

import numpy as np


class AdaptivePatcher:
    """
    Adaptive patching for measurement sets with arbitrary dimensions.

    Strategies:
    1. Pad to next multiple of patch_size
    2. Track padding for later removal
    3. Support both uniform and reflective padding
    """

    def __init__(
        self, data_shape: tuple[int, ...], patch_size: int = 1024, padding_mode: str = "reflect"
    ):
        """
        Initialize adaptive patcher

        Args:
            data_shape: Original data shape (baselines, pols, channels, times)
            patch_size: Target patch size (must match training)
            padding_mode: 'reflect', 'edge', or 'constant'
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
        """Round up to next multiple"""
        return ((value + multiple - 1) // multiple) * multiple

    def pad_data(self, data: np.ndarray) -> np.ndarray:
        """
        Pad data to match patch_size requirements

        Args:
            data: Input data (baselines, pols, channels, times)

        Returns:
            Padded data (baselines, pols, padded_channels, padded_times)
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
        Crop padded flags back to original dimensions

        Args:
            flags: Padded flags (baselines, pols, padded_channels, padded_times)

        Returns:
            Cropped flags matching original shape
        """
        return flags[:, :, : self.channels, : self.times]

    def get_patch_info(self) -> dict:
        """Get patching configuration info"""
        return {
            "original_shape": self.original_shape,
            "padded_shape": (self.baselines, self.pols, self.padded_channels, self.padded_times),
            "patch_size": self.patch_size,
            "num_patches_h": self.num_patches_h,
            "num_patches_w": self.num_patches_w,
            "total_patches": self.baselines * self.pols * self.num_patches_h * self.num_patches_w,
            "padding": {"channels": self.pad_channels, "times": self.pad_times},
        }


def check_ms_compatibility(ms_path: str, patch_size: int = 1024) -> dict:
    """
    Check if MS dimensions are compatible with patch_size

    Args:
        ms_path: Path to measurement set
        patch_size: Target patch size

    Returns:
        Dictionary with compatibility info
    """
    from rfi_toolbox.io import MSLoader

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
    """Generate recommendation message"""
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
