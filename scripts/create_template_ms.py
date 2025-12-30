#!/usr/bin/env python
"""
Create Template Measurement Set

Creates a minimal MS with specified dimensions using CASA simobserve.
Used as template for injecting synthetic RFI data.

Usage:
    python scripts/create_template_ms.py --output template_1024x1024.ms
"""

import argparse
import os
import re
import shutil
from pathlib import Path

from casatasks import flagdata
from casatasks.private import simutil
from casatools import ctsys, measures, simulator, table


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

    # Create MS using casatools simulator (replacement for simobserve)
    print("\nCreating MS with casatools simulator...")

    # Helper to parse human-readable freq/bandwidth strings like '1.5GHz', '128MHz'
    def _parse_size(s):
        s = str(s).strip()
        # Match a numeric value optionally followed by a unit like Hz, kHz, MHz, GHz (case-insensitive)
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*([kKmMgG]?[hH][zZ])?$", s)
        if not m:
            raise ValueError(f"Cannot parse frequency/bandwidth size: {s}")
        val = float(m.group(1))
        unit = (m.group(2) or "Hz").lower()
        multipliers = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
        return val * multipliers.get(unit, 1.0)

    msname = str(output_path)

    # Remove any existing MS at target
    if os.path.exists(msname):
        print(f"Removing existing MS: {msname}")
        shutil.rmtree(msname)

    # Prepare antennalist path (try to resolve within CASA simmos if given as simple name)
    antennalist_path = antennalist
    if not os.path.isabs(antennalist_path) or not os.path.exists(antennalist_path):
        try:
            simmos_dir = ctsys.resolve("alma/simmos")
            candidate = os.path.join(simmos_dir, antennalist)
            if os.path.exists(candidate):
                antennalist_path = candidate
        except Exception:
            # fallback to given antennalist
            pass

    # Open simulator and construct MS
    sm = simulator()
    me = measures()
    mysu = simutil.simutil()

    sm.open(ms=msname)

    # Read antenna config and optionally limit to requested number of antennas
    (x, y, z, d, an, an2, telname, obspos) = mysu.readantenna(antennalist_path)
    if num_antennas and num_antennas < len(an):
        an = an[:num_antennas]
        x = x[:num_antennas]
        y = y[:num_antennas]
        z = z[:num_antennas]
        d = d[:num_antennas]

    sm.setconfig(
        telescopename=telname,
        x=x,
        y=y,
        z=z,
        dishdiameter=d,
        mount=["alt-az"],
        antname=an,
        coordsystem="local",
        referencelocation=me.observatory(telname),
    )

    sm.setfeed(mode="perfect R L", pol=[""])

    center_freq_hz = _parse_size(frequency)
    bandwidth_hz = _parse_size(bandwidth)
    deltafreq_hz = bandwidth_hz / max(1, num_channels)

    sm.setspwindow(
        spwname="SPW0",
        freq=f"{int(center_freq_hz)}Hz",
        deltafreq=f"{int(bandwidth_hz)}Hz",
        freqresolution=f"{int(deltafreq_hz)}Hz",
        nchannels=num_channels,
        stokes="RR RL LR LL",
    )

    # Phase center (match previous indirection)
    sm.setfield(
        sourcename="fake",
        sourcedirection=me.direction(rf="J2000", v0="10h00m00.0s", v1="-30d00m00.0s"),
    )

    sm.setlimits(shadowlimit=0.01, elevationlimit="1deg")
    sm.setauto(autocorrwt=0.0)

    # Compute start/stop times from total_time_sec
    total_time_hours = total_time_sec / 3600.0
    start = f"-{total_time_hours/2:.6f}h"
    stop = f"+{total_time_hours/2:.6f}h"

    sm.settimes(
        integrationtime=f"{integration_time}s",
        usehourangle=True,
        referencetime=me.epoch("UTC", "2019/10/4/00:00:00"),
    )

    sm.observe(sourcename="fake", spwname="SPW0", starttime=start, stoptime=stop)

    sm.close()

    # Ensure DATA column exists/unflag everything
    flagdata(vis=msname, mode="unflag")

    print(f"\n✓ MS created: {msname}")

    # Verify dimensions

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
    parser.add_argument("--frequency", default="1.5GHz", help="Center frequency (default: 1.5GHz)")
    parser.add_argument("--bandwidth", default="128MHz", help="Total bandwidth (default: 128MHz)")

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
