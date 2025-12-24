#!/usr/bin/env python
"""
Create Template Measurement Set

Creates a minimal MS with specified dimensions using CASA simobserve.
Used as template for injecting synthetic RFI data.

Usage:
    python scripts/create_template_ms.py --output template_1024x1024.ms
"""

import argparse
import shutil
from pathlib import Path

from casatasks import simobserve


def create_template_ms(
    output_path,
    num_channels=1024,
    num_times=1024,
    integration_time=10.0,
    num_antennas=27,
    antennalist="vla.d.cfg",
    frequency="1.5GHz",
    bandwidth="128MHz",
):
    """
    Create template MS with simobserve.

    Args:
        output_path: Path for output MS
        num_channels: Number of frequency channels (default: 1024)
        num_times: Number of time samples (default: 1024)
        integration_time: Integration time in seconds (default: 10s)
        num_antennas: Number of antennas (default: 27 for VLA)
        antennalist: Antenna configuration (default: vla.d.cfg)
        frequency: Center frequency (default: 1.5GHz L-band)
        bandwidth: Total bandwidth (default: 128MHz)

    Returns:
        Path to created MS
    """
    output_path = Path(output_path)

    # Calculate total observing time to get desired number of samples
    # num_times = total_time / integration_time
    total_time_sec = num_times * integration_time
    total_time_str = f"{total_time_sec}s"

    print("=" * 70)
    print("Creating Template Measurement Set")
    print("=" * 70)
    print(f"Output:           {output_path}")
    print(f"Dimensions:       {num_channels} channels × {num_times} times")
    print(f"Frequency:        {frequency}")
    print(f"Bandwidth:        {bandwidth} ({num_channels} channels)")
    print(f"Integration time: {integration_time}s")
    print(f"Total time:       {total_time_str} ({total_time_sec/3600:.2f} hours)")
    print(f"Antennas:         {num_antennas} ({antennalist})")
    print("=" * 70)

    # Create project directory
    project_name = output_path.stem
    project_dir = output_path.parent / project_name

    # Clean up if exists
    if project_dir.exists():
        print(f"\nRemoving existing project: {project_dir}")
        shutil.rmtree(project_dir)

    # Run simobserve
    print(f"\nRunning simobserve...")
    print("This will create an empty MS with the specified structure...")

    simobserve(
        project=project_name,
        skymodel="",  # Empty sky (no sources)
        inbright="",
        indirection="J2000 10h00m00.0s -30d00m00.0s",  # Arbitrary direction
        incell="0.5arcsec",
        inwidth=bandwidth,
        incenter=frequency,
        innchan=num_channels,
        # Observation parameters
        obsmode="int",  # Interferometer
        antennalist=antennalist,
        totaltime=total_time_str,
        integration=f"{integration_time}s",
        # Output
        thermalnoise="",  # No noise
        graphics="none",
        verbose=False,
    )

    # Find the created MS
    # simobserve creates: project/project.antennalist.ms
    ms_pattern = list(project_dir.glob("*.ms"))

    if not ms_pattern:
        raise FileNotFoundError(f"No MS created in {project_dir}")

    created_ms = ms_pattern[0]
    print(f"\n✓ MS created: {created_ms}")

    # Move to desired output location
    if output_path.exists():
        shutil.rmtree(output_path)

    shutil.move(str(created_ms), str(output_path))
    print(f"✓ Moved to: {output_path}")

    # Clean up project directory
    shutil.rmtree(project_dir)

    # Verify dimensions
    from casatools import table

    tb = table()
    tb.open(str(output_path))
    nrows = tb.nrows()
    tb.close()

    # Open spectral window table
    tb.open(str(output_path / "SPECTRAL_WINDOW"))
    actual_channels = tb.getcol("NUM_CHAN")
    tb.close()

    print("\nVerification:")
    print(f"  Total rows:    {nrows}")
    print(f"  Channels/SPW:  {actual_channels}")
    print(f"  Expected times: {num_times}")

    print(f"\n✓ Template MS ready: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create template MS for RFI validation")
    parser.add_argument(
        "--output",
        default="template_1024x1024.ms",
        help="Output MS path (default: template_1024x1024.ms)",
    )
    parser.add_argument(
        "--channels", type=int, default=1024, help="Number of channels (default: 1024)"
    )
    parser.add_argument(
        "--times", type=int, default=1024, help="Number of time samples (default: 1024)"
    )
    parser.add_argument(
        "--integration",
        type=float,
        default=10.0,
        help="Integration time in seconds (default: 10s)",
    )
    parser.add_argument(
        "--antennas", type=int, default=27, help="Number of antennas (default: 27 VLA)"
    )
    parser.add_argument(
        "--config",
        default="vla.d.cfg",
        help="Antenna configuration (default: vla.d.cfg)",
    )
    parser.add_argument(
        "--frequency", default="1.5GHz", help="Center frequency (default: 1.5GHz)"
    )
    parser.add_argument(
        "--bandwidth", default="128MHz", help="Total bandwidth (default: 128MHz)"
    )

    args = parser.parse_args()

    create_template_ms(
        output_path=args.output,
        num_channels=args.channels,
        num_times=args.times,
        integration_time=args.integration,
        num_antennas=args.antennas,
        antennalist=args.config,
        frequency=args.frequency,
        bandwidth=args.bandwidth,
    )


if __name__ == "__main__":
    main()
