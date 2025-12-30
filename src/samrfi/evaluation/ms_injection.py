"""
MS Data Injection - Replace DATA column with synthetic visibilities for validation

This module allows injecting synthetic RFI data into existing measurement sets
for benchmarking SAM-RFI against traditional CASA flagging methods.
"""

import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from casatools import table

    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False


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
    if not CASA_AVAILABLE:
        raise ImportError(
            "casatools is required for MS injection. " "Install with: pip install samrfi[casa]"
        )

    template_ms_path = Path(template_ms_path)

    # Default output path
    if output_ms_path is None:
        output_ms_path = template_ms_path.parent / f"{template_ms_path.stem}.synthetic.ms"
    else:
        output_ms_path = Path(output_ms_path)

    # Copy template MS only if different (otherwise modify in-place)
    if template_ms_path.resolve() != output_ms_path.resolve():
        print(f"Copying template MS: {template_ms_path} → {output_ms_path}")
        if output_ms_path.exists():
            shutil.rmtree(output_ms_path)
        shutil.copytree(template_ms_path, output_ms_path)
    else:
        print(f"Modifying MS in-place: {output_ms_path}")

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
        print("  WARNING: MS has SPWs with different channel counts. " "Using first SPW only.")

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
            subtable = tb.query(f"DATA_DESC_ID=={spw_idx} && ANTENNA1=={ant1} && ANTENNA2=={ant2}")

            if subtable.nrows() == 0:
                print(f"  WARNING: No rows for baseline ({ant1},{ant2}), SPW {spw_idx}")
                subtable.close()
                continue

            # Extract data for this SPW
            if split_spws:
                start_ch = spw_idx * channels_in_spw
                end_ch = (spw_idx + 1) * channels_in_spw
                spw_data = baseline_data[:, start_ch:end_ch, :]
            else:
                spw_data = baseline_data  # Same data for all SPWs

            # Write to DATA column: read existing cell shape first and build a matching array
            nrows = subtable.nrows()

            # Ensure time dimension matches rows in this subtable
            if spw_data.shape[2] != nrows:
                subtable.close()
                raise ValueError(
                    f"Time mismatch for baseline ({ant1},{ant2}), SPW {spw_idx}: "
                    f"data times={spw_data.shape[2]} but MS has {nrows} rows"
                )

            # Read existing DATA column in bulk to infer per-row shape and dtype
            try:
                existing = subtable.getcol("DATA")
            except Exception as e:
                subtable.close()
                raise RuntimeError(
                    "Unable to read DATA column with getcol; MS may have non-uniform row shapes. "
                    "Aborting injection." + f" (error: {e})"
                ) from e

            # existing typical shape: (npol, nchan, nrows) or (npol, nchan, nrows, extra)
            # Find which axis corresponds to rows (should equal nrows)
            row_axis = None
            for ax in range(existing.ndim):
                if existing.shape[ax] == nrows:
                    row_axis = ax
                    break

            if row_axis is None:
                subtable.close()
                raise RuntimeError(
                    f"Unexpected DATA column shape {existing.shape}; cannot find rows axis matching {nrows}"
                )

            cell_dtype = existing.dtype

            # We will construct new_col with same shape as existing, then fill it
            new_col = np.empty_like(existing)

            npols = spw_data.shape[0]
            nchan = spw_data.shape[1]

            # Determine how to map spw_data (pols, chan, time) into existing axes
            # Strategy: identify axes for pols and channels in existing array (excluding row_axis)
            other_axes = [i for i in range(existing.ndim) if i != row_axis]
            if len(other_axes) < 2:
                subtable.close()
                raise RuntimeError(f"DATA column has unexpected ndim {existing.ndim}")

            ax_pol, ax_chan = other_axes[0], other_axes[1]

            # Determine if order is (pol,chan,rows) or (chan,pol,rows)
            pol_size = existing.shape[ax_pol]
            chan_size = existing.shape[ax_chan]

            transpose = False

            if pol_size == npols and chan_size == nchan:
                transpose = False
            elif pol_size == nchan and chan_size == npols:
                transpose = True
            else:
                # Maybe there is an extra trailing singleton axis (e.g., (pol, chan, rows, 1))
                # Try to handle if one of other_axes corresponds to a trailing extra dim
                # We'll handle by checking for singleton dims later per-cell
                pass

            # Fill new_col by iterating over time rows for speed and clarity
            for t in range(nrows):
                cell = spw_data[:, :, t]  # (pols, channels)
                if transpose:
                    cell = cell.T
                # Place cell into new_col along row_axis==t
                # Build an index tuple of slices
                idx = [slice(None)] * existing.ndim
                idx[row_axis] = t
                # If existing has extra dims beyond pol/chan/time, we expect them to be singleton
                # and will broadcast the cell into that shape
                # Create a view of destination with shape matching cell
                dest = new_col[tuple(idx)]
                # dest has shape (pol, chan) or (pol, chan, extra)
                if dest.ndim == 2:
                    dest[:] = cell.astype(cell_dtype)
                elif dest.ndim == 3 and dest.shape[2] == 1:
                    dest[:, :, 0] = cell.astype(cell_dtype)
                else:
                    subtable.close()
                    raise RuntimeError(
                        f"Unsupported per-row DATA cell shape when writing: {dest.shape}"
                    )

            # Try bulk write first, fall back to per-row putcell if needed
            try:
                subtable.putcol("DATA", new_col)
            except Exception:
                # Fallback: per-row writes
                for row_idx in range(nrows):
                    # Extract cell-sized slice matching existing layout
                    idx = [slice(None)] * existing.ndim
                    idx[row_axis] = row_idx
                    cell_val = new_col[tuple(idx)]
                    try:
                        subtable.putcell("DATA", row_idx, cell_val)
                    except Exception as e:
                        subtable.close()
                        raise RuntimeError(f"Failed to write DATA row {row_idx}: {e}") from e

            subtable.close()

    tb.close()

    print(f"\n✓ Synthetic data injected into: {output_ms_path}")
    return output_ms_path
