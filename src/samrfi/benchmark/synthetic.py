"""
Synthetic-RFI injection for Gate 0.

Generates synthetic visibilities with known RFI (rfi_toolbox SyntheticDataGenerator)
baseline-by-baseline and injects them into a template MS, returning both the
injected complex data and the matching ground-truth RFI mask. Lifted, with light
cleanup, from the original scripts/benchmark_synthetic.py so behavior is
preserved.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def generate_and_inject(config, ms_path, baseline_map, num_channels, num_times, num_pols=4):
    """Generate synthetic RFI and inject it into the MS, one baseline at a time.

    Args:
        config: DataConfig with a `.synthetic` section (e.g. from ConfigLoader).
        ms_path: Template MS to inject into (DATA column is overwritten).
        baseline_map: List of (ant1, ant2) tuples in data order.
        num_channels: Total channels across SPWs.
        num_times: Number of integrations.
        num_pols: Number of polarizations (default 4).

    Returns:
        (synthetic_data, ground_truth), each (baselines, pols, channels, times);
        synthetic_data is complex visibilities, ground_truth is a boolean RFI mask.
    """
    from rfi_toolbox.data_generation import SyntheticDataGenerator
    from rfi_toolbox.io import inject_synthetic_data

    synth_config = config.synthetic
    generator = SyntheticDataGenerator(config)

    gen_kwargs = {
        "num_channels": num_channels,
        "num_times": num_times,
        "noise_level": synth_config.get("noise_mjy", 1.0),
        "rfi_power_min": synth_config.get("rfi_power_min", 1000.0),
        "rfi_power_max": synth_config.get("rfi_power_max", 10000.0),
        "rfi_config": generator._parse_rfi_config(synth_config),
        "enable_bandpass": synth_config.get("enable_bandpass_rolloff", False),
        "bandpass_order": synth_config.get("bandpass_polynomial_order", 0),
        "num_polarizations": num_pols,
        "pol_corr": synth_config.get("polarization_correlation", 0.8),
        "synth_config": synth_config,
    }

    all_data = []
    all_ground_truth = []

    for ant1, ant2 in tqdm(baseline_map, desc="  Inject baselines"):
        waterfall, ground_truth, _ = generator._generate_single_sample(**gen_kwargs)
        all_data.append(waterfall[0])  # drop batch dim -> (pols, chan, time)
        all_ground_truth.append(ground_truth[0])

        inject_synthetic_data(
            template_ms_path=ms_path,
            synthetic_data=waterfall[0][np.newaxis, :, :, :],  # add baseline dim
            output_ms_path=ms_path,
            baseline_map=[(ant1, ant2)],
        )

    synthetic_data = np.stack(all_data)  # (baselines, pols, channels, times)
    ground_truth = np.stack(all_ground_truth).astype(bool)
    return synthetic_data, ground_truth
