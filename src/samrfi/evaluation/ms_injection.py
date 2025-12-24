"""
MS Data Injection - Replace DATA column with synthetic visibilities for validation

This module allows injecting synthetic RFI data into existing measurement sets
for benchmarking SAM-RFI against traditional CASA flagging methods.
"""

import shutil
from pathlib import Path

import numpy as np
from casatools import table
from tqdm import tqdm


def inject_synthetic_data(
    template_ms_path,
    synthetic_data,
    output_ms_path=None,
    baseline_map=None,
    num_antennas=None,
):
    """
    Inject synthetic visibility data into a measurement set.

    Takes an existing MS as template (for proper structure/metadata) and replaces
    the DATA column with synthetic visibilities. Preserves all MS structure.

    Args:
        template_ms_path: Path to existing MS to use as template
        synthetic_data: Complex visibility data, shape (baselines, pols, channels, times)
        output_ms_path: Path for output MS (default: template_ms_path + '.synthetic')
        baseline_map: List of (ant1, ant2) tuples matching data order (optional)
        num_antennas: Number of antennas (optional, inferred from data if not provided)

    Returns:
        Path to output MS with injected data
    """
    template_ms_path = Path(template_ms_path)

    # Default output path
    if output_ms_path is None:
        output_ms_path = template_ms_path.parent / f"{template_ms_path.stem}.synthetic.ms"
    else:
        output_ms_path = Path(output_ms_path)

    # Copy template MS
    print(f"Copying template MS: {template_ms_path} → {output_ms_path}")
    if output_ms_path.exists():
        shutil.rmtree(output_ms_path)
    shutil.copytree(template_ms_path, output_ms_path)

    # Validate data shape
    num_baselines, num_pols, num_channels, num_times = synthetic_data.shape
    print(f"\nSynthetic data shape: {synthetic_data.shape}")
    print(f"  Baselines: {num_baselines}")
    print(f"  Polarizations: {num_pols}")
    print(f"  Channels: {num_channels}")
    print(f"  Times: {num_times}")

    # Create baseline map if not provided
    if baseline_map is None:
        if num_antennas is None:
            # Infer from number of baselines: n_baselines = n_ant * (n_ant - 1) / 2
            num_antennas = int((1 + np.sqrt(1 + 8 * num_baselines)) / 2)
        baseline_map = []
        for i in range(num_antennas):
            for j in range(i + 1, num_antennas):
                baseline_map.append((i, j))
                if len(baseline_map) >= num_baselines:
                    break
            if len(baseline_map) >= num_baselines:
                break

    print(f"  Inferred {len(baseline_map)} baselines from {num_antennas} antennas")

    # Open MS for writing
    tb = table()
    tb.open(str(output_ms_path), nomodify=False)

    # Get SPW info
    tb_spw = table()
    tb_spw.open(str(output_ms_path / "SPECTRAL_WINDOW"))
    channels_per_spw = tb_spw.getcol("NUM_CHAN")
    num_spw = tb_spw.nrows()
    tb_spw.close()

    print(f"  MS has {num_spw} SPWs with {channels_per_spw} channels")

    # For simplicity, assume all SPWs have same channel count
    # and we're filling all SPWs with the same data
    if len(set(channels_per_spw)) > 1:
        print(
            "  WARNING: MS has SPWs with different channel counts. "
            "Using first SPW only."
        )

    channels_in_spw = channels_per_spw[0]

    # Check if we need to split channels across SPWs
    if num_channels == channels_in_spw * num_spw:
        # Data spans multiple SPWs
        print(f"  Splitting {num_channels} channels across {num_spw} SPWs")
        split_spws = True
    elif num_channels == channels_in_spw:
        # Data fits in one SPW, replicate to all
        print(f"  Replicating {num_channels} channels to all {num_spw} SPWs")
        split_spws = False
    else:
        raise ValueError(
            f"Channel mismatch: data has {num_channels} channels, "
            f"MS SPW has {channels_in_spw} channels"
        )

    # Write data to MS
    print("\nInjecting synthetic data into MS...")
    for baseline_idx, (ant1, ant2) in enumerate(tqdm(baseline_map, desc="Baselines")):
        baseline_data = synthetic_data[baseline_idx]  # (pols, channels, times)

        for spw_idx in range(num_spw):
            # Query this baseline + SPW
            subtable = tb.query(
                f"DATA_DESC_ID=={spw_idx} && ANTENNA1=={ant1} && ANTENNA2=={ant2}"
            )

            if subtable.nrows() == 0:
                print(
                    f"  WARNING: No rows for baseline ({ant1},{ant2}), SPW {spw_idx}"
                )
                subtable.close()
                continue

            # Extract data for this SPW
            if split_spws:
                start_ch = spw_idx * channels_in_spw
                end_ch = (spw_idx + 1) * channels_in_spw
                spw_data = baseline_data[:, start_ch:end_ch, :]
            else:
                spw_data = baseline_data  # Same data for all SPWs

            # Write to DATA column
            subtable.putcol("DATA", spw_data)
            subtable.close()

    tb.close()

    print(f"\n✓ Synthetic data injected into: {output_ms_path}")
    return output_ms_path
