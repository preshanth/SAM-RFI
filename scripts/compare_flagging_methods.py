#!/usr/bin/env python
"""
Comprehensive Flagging Comparison: SAM-RFI vs CASA (tfcrop + rflag)

This script compares SAM-RFI flagging against traditional CASA methods,
computing both visibility-level and image-level metrics.

Workflow:
1. Start from Hanning-smoothed MS (before any flagging)
2. Branch A: Apply CASA flagging (tfcrop + rflag)
3. Branch B: Apply SAM-RFI flagging
4. Compute visibility metrics for both
5. Image both with tclean
6. Compute image quality metrics
7. Generate comparison plots

Usage:
    python compare_flagging_methods.py --ms 3C129_pband.ms --model sam2_rfi.pth --output ./comparison_results/
"""

import os
import sys
import shutil
import argparse
import numpy as np
import tarfile
import urllib.request
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# CASA imports
try:
    from casatasks import flagdata, gaincal, bandpass, applycal, tclean, imstat, exportfits
    from casatools import table, image
    CASA_AVAILABLE = True
except ImportError:
    print("WARNING: CASA tasks not available. Install casatasks or run within CASA.")
    CASA_AVAILABLE = False

# SAM-RFI imports
try:
    from samrfi.inference import RFIPredictor
    from samrfi.data import MSLoader
    SAMRFI_AVAILABLE = True
except ImportError:
    print("WARNING: SAM-RFI not available. Install samrfi package.")
    SAMRFI_AVAILABLE = False


def download_tutorial_data(output_dir='.', force=False):
    """
    Download VLA P-band tutorial data from NRAO CASA guides

    Downloads 3C129 P-band dataset (26.46 GB)
    URL: https://casa.nrao.edu/Data/EVLA/Pband/P_band_3C129.tgz

    Args:
        output_dir: Directory to download and extract data
        force: If True, re-download even if data exists

    Returns:
        Path to extracted MS directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # URLs and filenames
    url = "https://casa.nrao.edu/Data/EVLA/Pband/P_band_3C129.tgz"
    tarball_path = output_dir / "P_band_3C129.tgz"
    ms_path = output_dir / "imag-test-copy.57080.956837025464"

    print("="*70)
    print("VLA P-BAND TUTORIAL DATA DOWNLOAD")
    print("="*70)
    print(f"URL: {url}")
    print(f"Size: ~26.46 GB")
    print(f"Output: {output_dir}")
    print("="*70)

    # Check if already downloaded
    if ms_path.exists() and not force:
        print(f"\n✓ Data already exists at: {ms_path}")
        print("  Use --force to re-download")
        return ms_path

    # Download tarball
    if not tarball_path.exists() or force:
        print(f"\n[1/2] Downloading {tarball_path.name}...")
        print("  This may take a while (26.46 GB)...")

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)

            # Update every 1%
            if block_num % 100 == 0:
                print(f"  Progress: {percent:.1f}% ({mb_downloaded:.0f}/{mb_total:.0f} MB)",
                      end='\r', flush=True)

        try:
            urllib.request.urlretrieve(url, tarball_path, reporthook=progress_hook)
            print(f"\n  ✓ Download complete: {tarball_path}")
        except Exception as e:
            print(f"\n  ❌ Download failed: {e}")
            if tarball_path.exists():
                tarball_path.unlink()
            raise
    else:
        print(f"\n[1/2] Tarball already exists: {tarball_path}")

    # Extract tarball
    print(f"\n[2/2] Extracting tarball...")
    try:
        with tarfile.open(tarball_path, 'r:gz') as tar:
            # Get total size for progress
            members = tar.getmembers()
            total_members = len(members)

            # Extract with progress
            for i, member in enumerate(members):
                tar.extract(member, path=output_dir)
                if i % 1000 == 0:
                    percent = (i / total_members) * 100
                    print(f"  Progress: {percent:.1f}% ({i}/{total_members} files)",
                          end='\r', flush=True)

            print(f"\n  ✓ Extraction complete")
    except Exception as e:
        print(f"\n  ❌ Extraction failed: {e}")
        raise

    # Verify extraction
    if not ms_path.exists():
        raise FileNotFoundError(f"Expected MS not found after extraction: {ms_path}")

    print(f"\n✓ Tutorial data ready at: {ms_path}")
    print(f"\nNext steps:")
    print(f"  1. Process with CASA following the guide")
    print(f"  2. Run Hanning smoothing to create 3C129_pband.ms")
    print(f"  3. Use that MS with this comparison script")
    print("="*70)

    return ms_path


class FlaggingComparison:
    """
    Comprehensive comparison framework for RFI flagging methods
    """

    def __init__(self, ms_path, model_path, output_dir, calibrator_field='3C147',
                 target_field='3C129', patch_size=1024):
        """
        Initialize comparison framework

        Args:
            ms_path: Path to Hanning-smoothed MS (before flagging)
            model_path: Path to trained SAM-RFI model
            output_dir: Output directory for results
            calibrator_field: Name of flux calibrator
            target_field: Name of target field
            patch_size: Patch size for SAM-RFI (must match training)
        """
        self.ms_path = Path(ms_path)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.calibrator_field = calibrator_field
        self.target_field = target_field
        self.patch_size = patch_size

        # Validate inputs
        if not self.ms_path.exists():
            raise FileNotFoundError(f"MS not found: {self.ms_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # MS copies for each method
        self.ms_casa = self.output_dir / f"{self.ms_path.stem}_casa.ms"
        self.ms_sam = self.output_dir / f"{self.ms_path.stem}_sam.ms"

        # Results storage
        self.results = {
            'casa': {},
            'sam': {},
            'comparison': {}
        }

        print("="*70)
        print("FLAGGING COMPARISON FRAMEWORK INITIALIZED")
        print("="*70)
        print(f"Input MS:        {self.ms_path}")
        print(f"SAM-RFI Model:   {self.model_path}")
        print(f"Output Dir:      {self.output_dir}")
        print(f"Calibrator:      {self.calibrator_field}")
        print(f"Target:          {self.target_field}")
        print(f"Patch Size:      {self.patch_size}")
        print("="*70)

    def prepare_ms_copies(self):
        """Create MS copies for each flagging method"""
        print("\n[Step 1/7] Preparing MS copies...")

        # Copy for CASA flagging
        if self.ms_casa.exists():
            print(f"  Removing existing CASA MS: {self.ms_casa}")
            shutil.rmtree(self.ms_casa)

        print(f"  Copying to: {self.ms_casa}")
        shutil.copytree(self.ms_path, self.ms_casa)

        # Copy for SAM flagging
        if self.ms_sam.exists():
            print(f"  Removing existing SAM MS: {self.ms_sam}")
            shutil.rmtree(self.ms_sam)

        print(f"  Copying to: {self.ms_sam}")
        shutil.copytree(self.ms_path, self.ms_sam)

        print("  ✓ MS copies created")

    def get_flag_summary(self, ms_path):
        """Get flag statistics from MS"""
        summary = flagdata(vis=str(ms_path), mode='summary')

        total = summary['total']
        flagged = summary['flagged']
        percent = 100.0 * flagged / total if total > 0 else 0.0

        return {
            'total': total,
            'flagged': flagged,
            'percent': percent,
            'by_antenna': summary.get('antenna', {}),
            'by_spw': summary.get('spw', {}),
            'by_field': summary.get('field', {})
        }

    def apply_casa_flagging(self):
        """
        Apply CASA flagging following VLA P-band guide

        Steps:
        1. tfcrop on cross-hands (ABS_XY, ABS_YX)
        2. tfcrop on parallel-hands (ABS_XX, ABS_YY)
        3. Preliminary calibration (G0, K0, B0)
        4. rflag on corrected data
        5. Extend flags
        """
        print("\n[Step 2/7] Applying CASA flagging (tfcrop + rflag)...")

        ms = str(self.ms_casa)

        # Get initial flag state
        print("\n  Initial flag state:")
        summary_0 = self.get_flag_summary(ms)
        print(f"    Flagged: {summary_0['percent']:.2f}%")

        # Step 1: tfcrop on cross-hands
        print("\n  [2.1] Running tfcrop on cross-hands (XY, YX)...")
        flagdata(
            vis=ms,
            field='*',
            mode='tfcrop',
            datacolumn='data',
            timecutoff=4.0,
            freqcutoff=3.0,
            maxnpieces=5,
            action='apply',
            display='',
            flagbackup=True,
            combinescans=True,
            ntime='3600s',
            correlation='ABS_XY,ABS_YX'
        )

        # Step 2: tfcrop on parallel-hands
        print("  [2.2] Running tfcrop on parallel-hands (XX, YY)...")
        flagdata(
            vis=ms,
            field='*',
            mode='tfcrop',
            datacolumn='data',
            timecutoff=3.0,
            freqcutoff=3.0,
            maxnpieces=2,
            action='apply',
            display='',
            flagbackup=False,
            combinescans=True,
            ntime='3600s',
            correlation='ABS_XX,ABS_YY'
        )

        # Step 3: Extend flags
        print("  [2.3] Extending flags...")
        flagdata(vis=ms, mode='extend')

        summary_1 = self.get_flag_summary(ms)
        print(f"    After tfcrop: {summary_1['percent']:.2f}% flagged "
              f"(+{summary_1['percent'] - summary_0['percent']:.2f}%)")

        # Step 4: Preliminary calibration for better RFI contrast
        print("\n  [2.4] Preliminary calibration (for rflag)...")

        cal_tables = []

        # Generate calibration tables with gencal
        # Antenna positions
        antpos_table = str(self.output_dir / 'casa_antpos.cal')
        print(f"    Computing antenna positions: {antpos_table}")
        try:
            from casatasks import gencal
            gencal(vis=ms, caltable=antpos_table, caltype='antpos')
            cal_tables.append(antpos_table)
            print("      ✓ Antenna positions applied")
        except Exception as e:
            print(f"      ⚠ Antenna position correction skipped: {e}")

        # Requantizer gains
        rq_table = str(self.output_dir / 'casa_rq.cal')
        print(f"    Computing requantizer gains: {rq_table}")
        try:
            gencal(vis=ms, caltable=rq_table, caltype='rq')
            cal_tables.append(rq_table)
            print("      ✓ Requantizer gains applied")
        except Exception as e:
            print(f"      ⚠ Requantizer correction skipped: {e}")

        # Ionospheric TEC correction
        print(f"    Computing ionospheric TEC corrections...")
        try:
            from casatasks.private import tec_maps
            tec_image, tec_rms_image, plotname = tec_maps.create(vis=ms, doplot=False)

            tecim_table = str(self.output_dir / 'casa_tecim.cal')
            gencal(vis=ms, caltable=tecim_table, caltype='tecim', infile=tec_image)
            cal_tables.append(tecim_table)
            print(f"      ✓ TEC corrections applied: {tecim_table}")
        except Exception as e:
            print(f"      ⚠ TEC correction skipped: {e}")

        # Phase-only gain calibration
        g0_table = str(self.output_dir / 'casa_G0.cal')
        print(f"    Computing phase gains: {g0_table}")
        gaincal(
            vis=ms,
            caltable=g0_table,
            gaintype='G',
            calmode='p',
            solint='int',
            field=self.calibrator_field,
            refant='ea09',
            gaintable=cal_tables
        )
        cal_tables.append(g0_table)

        # Delay calibration
        k0_table = str(self.output_dir / 'casa_K0.cal')
        print(f"    Computing delays: {k0_table}")
        gaincal(
            vis=ms,
            caltable=k0_table,
            gaintype='K',
            solint='inf',
            field=self.calibrator_field,
            refant='ea09',
            gaintable=cal_tables
        )
        cal_tables.append(k0_table)

        # Bandpass calibration
        b0_table = str(self.output_dir / 'casa_B0.cal')
        print(f"    Computing bandpass: {b0_table}")
        bandpass(
            vis=ms,
            caltable=b0_table,
            solint='inf',
            field=self.calibrator_field,
            refant='ea09',
            minsnr=2.0,
            gaintable=cal_tables
        )
        cal_tables.append(b0_table)

        # Apply calibration
        print("    Applying calibration...")
        applycal(
            vis=ms,
            field=self.calibrator_field,
            applymode='calflagstrict',
            gaintable=cal_tables
        )

        # Step 5: rflag on corrected data
        print("\n  [2.5] Running rflag on corrected data...")
        flagdata(
            vis=ms,
            field=self.calibrator_field,
            mode='rflag',
            datacolumn='corrected',
            timedevscale=4.0,
            freqdevscale=3.0,
            action='apply',
            flagbackup=True,
            combinescans=True,
            ntime='3600s'
        )

        # Final summary
        summary_2 = self.get_flag_summary(ms)
        print(f"\n  Final CASA flagging: {summary_2['percent']:.2f}% flagged "
              f"(+{summary_2['percent'] - summary_1['percent']:.2f}% from rflag)")

        self.results['casa']['flag_summary'] = summary_2
        self.results['casa']['cal_tables'] = cal_tables

        print("  ✓ CASA flagging complete")

    def apply_sam_flagging(self):
        """Apply SAM-RFI flagging"""
        print("\n[Step 3/7] Applying SAM-RFI flagging...")

        ms = str(self.ms_sam)

        # Get initial flag state
        print("\n  Initial flag state:")
        summary_0 = self.get_flag_summary(ms)
        print(f"    Flagged: {summary_0['percent']:.2f}%")

        # Initialize SAM3-RFI predictor
        print("\n  Loading SAM3-RFI model...")
        predictor = RFIPredictor(
            model_path=str(self.model_path),
            sam_checkpoint='unified',  # SAM3 uses single unified model (840M params)
            device='cuda',
            batch_size=8
        )

        # Run prediction
        print("\n  Running SAM-RFI prediction...")
        flags = predictor.predict_ms(
            ms_path=ms,
            num_antennas=None,  # All antennas
            patch_size=self.patch_size,
            stretch='SQRT',
            apply_existing_flags=False,
            save_flags=True
        )

        # Get final summary
        summary_1 = self.get_flag_summary(ms)
        print(f"\n  Final SAM flagging: {summary_1['percent']:.2f}% flagged "
              f"(+{summary_1['percent'] - summary_0['percent']:.2f}%)")

        self.results['sam']['flag_summary'] = summary_1
        self.results['sam']['flags'] = flags

        print("  ✓ SAM-RFI flagging complete")

    def compute_visibility_metrics(self):
        """Compute visibility-level comparison metrics"""
        print("\n[Step 4/7] Computing visibility metrics...")

        casa_summary = self.results['casa']['flag_summary']
        sam_summary = self.results['sam']['flag_summary']

        # Overall percentages
        casa_pct = casa_summary['percent']
        sam_pct = sam_summary['percent']

        print(f"\n  Flag percentages:")
        print(f"    CASA:    {casa_pct:.2f}%")
        print(f"    SAM-RFI: {sam_pct:.2f}%")
        print(f"    Diff:    {sam_pct - casa_pct:+.2f}%")

        # Load actual flags for overlap analysis
        print("\n  Loading flags for overlap analysis...")

        # CASA flags
        tb = table()
        tb.open(str(self.ms_casa))
        casa_flags = tb.getcol('FLAG')
        tb.close()

        # SAM flags
        tb.open(str(self.ms_sam))
        sam_flags = tb.getcol('FLAG')
        tb.close()

        # Compute overlap
        both_flagged = np.logical_and(casa_flags, sam_flags)
        only_casa = np.logical_and(casa_flags, ~sam_flags)
        only_sam = np.logical_and(~casa_flags, sam_flags)
        neither = np.logical_and(~casa_flags, ~sam_flags)

        total = casa_flags.size

        overlap_stats = {
            'both_flagged_pct': 100.0 * np.sum(both_flagged) / total,
            'only_casa_pct': 100.0 * np.sum(only_casa) / total,
            'only_sam_pct': 100.0 * np.sum(only_sam) / total,
            'neither_pct': 100.0 * np.sum(neither) / total,
            'agreement_pct': 100.0 * (np.sum(both_flagged) + np.sum(neither)) / total
        }

        print(f"\n  Flag overlap:")
        print(f"    Both methods:    {overlap_stats['both_flagged_pct']:.2f}%")
        print(f"    Only CASA:       {overlap_stats['only_casa_pct']:.2f}%")
        print(f"    Only SAM:        {overlap_stats['only_sam_pct']:.2f}%")
        print(f"    Neither:         {overlap_stats['neither_pct']:.2f}%")
        print(f"    Agreement:       {overlap_stats['agreement_pct']:.2f}%")

        self.results['comparison']['overlap'] = overlap_stats

        print("  ✓ Visibility metrics computed")

    def run_imaging(self, ms_path, imagename, spw='0~15'):
        """
        Run tclean imaging following VLA P-band guide

        Args:
            ms_path: Path to MS
            imagename: Output image name
            spw: Spectral windows to image
        """
        print(f"\n    Imaging {imagename}...")

        tclean(
            vis=str(ms_path),
            imagename=str(imagename),
            cell=['5.0arcsec', '5.0arcsec'],
            imsize=[4860, 4860],
            deconvolver='mtmfs',
            nterms=2,
            gridder='wproject',
            wprojplanes=128,
            stokes='I',
            niter=20000,
            spw=spw,
            interactive=False,  # Non-interactive for automated comparison
            scales=[0, 20, 30],
            pblimit=0.01,
            savemodel='none',  # Don't need model for comparison
            weighting='briggs',
            robust=0.0
        )

        print(f"    ✓ Image created: {imagename}.image.tt0")

    def compute_image_metrics(self, imagename):
        """
        Compute image quality metrics

        Args:
            imagename: Base name of image (without .image.tt0)

        Returns:
            Dictionary of metrics
        """
        image_path = f"{imagename}.image.tt0"

        # Use imstat to compute statistics
        stats = imstat(imagename=image_path)

        # RMS in off-source region (outer 20% of image)
        # For simplicity, use full image RMS (in production, mask source)
        rms = stats['rms'][0]

        # Peak flux
        peak = stats['max'][0]

        # Dynamic range
        dynamic_range = peak / rms if rms > 0 else 0.0

        metrics = {
            'rms_Jy': rms,
            'peak_Jy': peak,
            'dynamic_range': dynamic_range,
            'mean_Jy': stats['mean'][0],
            'image_path': image_path
        }

        return metrics

    def create_images(self):
        """Create images for both flagging methods"""
        print("\n[Step 5/7] Creating images...")

        # CASA image
        casa_imagename = self.output_dir / 'image_casa'
        print("\n  [5.1] Imaging CASA-flagged data...")
        self.run_imaging(self.ms_casa, casa_imagename)

        casa_metrics = self.compute_image_metrics(casa_imagename)
        print(f"    RMS: {casa_metrics['rms_Jy']*1e3:.2f} mJy")
        print(f"    Peak: {casa_metrics['peak_Jy']:.3f} Jy")
        print(f"    Dynamic Range: {casa_metrics['dynamic_range']:.1f}")

        self.results['casa']['image_metrics'] = casa_metrics

        # SAM image
        sam_imagename = self.output_dir / 'image_sam'
        print("\n  [5.2] Imaging SAM-flagged data...")
        self.run_imaging(self.ms_sam, sam_imagename)

        sam_metrics = self.compute_image_metrics(sam_imagename)
        print(f"    RMS: {sam_metrics['rms_Jy']*1e3:.2f} mJy")
        print(f"    Peak: {sam_metrics['peak_Jy']:.3f} Jy")
        print(f"    Dynamic Range: {sam_metrics['dynamic_range']:.1f}")

        self.results['sam']['image_metrics'] = sam_metrics

        # Comparison
        rms_improvement = (casa_metrics['rms_Jy'] - sam_metrics['rms_Jy']) / casa_metrics['rms_Jy'] * 100
        dr_improvement = (sam_metrics['dynamic_range'] - casa_metrics['dynamic_range']) / casa_metrics['dynamic_range'] * 100

        print(f"\n  Image Quality Comparison:")
        print(f"    RMS improvement:    {rms_improvement:+.1f}%")
        print(f"    DR improvement:     {dr_improvement:+.1f}%")

        self.results['comparison']['image'] = {
            'rms_improvement_pct': rms_improvement,
            'dr_improvement_pct': dr_improvement
        }

        print("  ✓ Images created")

    def create_plots(self):
        """Generate comparison plots"""
        print("\n[Step 6/7] Creating comparison plots...")

        # Plot 1: Flag percentage comparison
        self._plot_flag_comparison()

        # Plot 2: Venn diagram
        self._plot_venn_diagram()

        # Plot 3: Image metrics comparison
        self._plot_image_metrics()

        print("  ✓ Plots created")

    def _plot_flag_comparison(self):
        """Bar chart comparing flag percentages"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['CASA\n(tfcrop+rflag)', 'SAM-RFI']
        percentages = [
            self.results['casa']['flag_summary']['percent'],
            self.results['sam']['flag_summary']['percent']
        ]

        colors = ['#1f77b4', '#ff7f0e']
        bars = ax.bar(methods, percentages, color=colors, alpha=0.7, edgecolor='black')

        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Flagged Data (%)', fontsize=14)
        ax.set_title('Flagging Comparison: Data Flagged', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(percentages) * 1.2)

        plt.tight_layout()
        plot_path = self.output_dir / 'plot_flag_percentage.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"    Saved: {plot_path}")

    def _plot_venn_diagram(self):
        """Venn diagram showing flag overlap"""
        fig, ax = plt.subplots(figsize=(10, 8))

        overlap = self.results['comparison']['overlap']

        # Simple bar chart representation (true Venn needs matplotlib_venn)
        categories = ['Both\nMethods', 'Only\nCASA', 'Only\nSAM', 'Neither']
        values = [
            overlap['both_flagged_pct'],
            overlap['only_casa_pct'],
            overlap['only_sam_pct'],
            overlap['neither_pct']
        ]

        colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Percentage of Data (%)', fontsize=14)
        ax.set_title('Flag Overlap Analysis', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add agreement annotation
        agreement = overlap['agreement_pct']
        ax.text(0.5, 0.95, f'Agreement: {agreement:.2f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=13, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_path = self.output_dir / 'plot_flag_overlap.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"    Saved: {plot_path}")

    def _plot_image_metrics(self):
        """Compare image quality metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        casa_metrics = self.results['casa']['image_metrics']
        sam_metrics = self.results['sam']['image_metrics']

        # RMS comparison
        methods = ['CASA', 'SAM-RFI']
        rms_values = [casa_metrics['rms_Jy'] * 1e3, sam_metrics['rms_Jy'] * 1e3]  # Convert to mJy

        bars1 = ax1.bar(methods, rms_values, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')

        for bar, val in zip(bars1, rms_values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax1.set_ylabel('RMS Noise (mJy/beam)', fontsize=13)
        ax1.set_title('Image RMS Noise (Lower is Better)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Dynamic range comparison
        dr_values = [casa_metrics['dynamic_range'], sam_metrics['dynamic_range']]

        bars2 = ax2.bar(methods, dr_values, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')

        for bar, val in zip(bars2, dr_values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax2.set_ylabel('Dynamic Range (Peak/RMS)', fontsize=13)
        ax2.set_title('Image Dynamic Range (Higher is Better)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / 'plot_image_metrics.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"    Saved: {plot_path}")

    def save_report(self):
        """Save comprehensive text report"""
        print("\n[Step 7/7] Saving report...")

        report_path = self.output_dir / 'comparison_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SAM-RFI vs CASA FLAGGING COMPARISON REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input MS: {self.ms_path}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write("\n")

            # Visibility metrics
            f.write("-"*70 + "\n")
            f.write("VISIBILITY-LEVEL METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"CASA flagging:    {self.results['casa']['flag_summary']['percent']:.2f}%\n")
            f.write(f"SAM-RFI flagging: {self.results['sam']['flag_summary']['percent']:.2f}%\n")
            f.write("\n")

            overlap = self.results['comparison']['overlap']
            f.write("Flag Overlap:\n")
            f.write(f"  Both methods:  {overlap['both_flagged_pct']:.2f}%\n")
            f.write(f"  Only CASA:     {overlap['only_casa_pct']:.2f}%\n")
            f.write(f"  Only SAM:      {overlap['only_sam_pct']:.2f}%\n")
            f.write(f"  Neither:       {overlap['neither_pct']:.2f}%\n")
            f.write(f"  Agreement:     {overlap['agreement_pct']:.2f}%\n")
            f.write("\n")

            # Image metrics
            f.write("-"*70 + "\n")
            f.write("IMAGE QUALITY METRICS\n")
            f.write("-"*70 + "\n")

            casa_img = self.results['casa']['image_metrics']
            sam_img = self.results['sam']['image_metrics']

            f.write("CASA Method:\n")
            f.write(f"  RMS:           {casa_img['rms_Jy']*1e3:.2f} mJy/beam\n")
            f.write(f"  Peak:          {casa_img['peak_Jy']:.3f} Jy/beam\n")
            f.write(f"  Dynamic Range: {casa_img['dynamic_range']:.1f}\n")
            f.write("\n")

            f.write("SAM-RFI Method:\n")
            f.write(f"  RMS:           {sam_img['rms_Jy']*1e3:.2f} mJy/beam\n")
            f.write(f"  Peak:          {sam_img['peak_Jy']:.3f} Jy/beam\n")
            f.write(f"  Dynamic Range: {sam_img['dynamic_range']:.1f}\n")
            f.write("\n")

            comp = self.results['comparison']['image']
            f.write("Improvement (SAM vs CASA):\n")
            f.write(f"  RMS:           {comp['rms_improvement_pct']:+.1f}%\n")
            f.write(f"  Dynamic Range: {comp['dr_improvement_pct']:+.1f}%\n")
            f.write("\n")

            f.write("="*70 + "\n")

        print(f"  Saved: {report_path}")
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)

    def run(self):
        """Run complete comparison pipeline"""
        try:
            self.prepare_ms_copies()
            self.apply_casa_flagging()
            self.apply_sam_flagging()
            self.compute_visibility_metrics()
            self.create_images()
            self.create_plots()
            self.save_report()

            print(f"\n✓ All results saved to: {self.output_dir}")

        except Exception as e:
            print(f"\n❌ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Compare SAM-RFI vs CASA flagging methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download tutorial data
  python compare_flagging_methods.py --download-data --data-dir ./tutorial_data/

  # Full comparison
  python compare_flagging_methods.py --ms 3C129_pband.ms --model sam2_rfi.pth --output ./results/

  # Specify fields
  python compare_flagging_methods.py --ms data.ms --model model.pth --output ./out/ \\
      --calibrator 3C147 --target 3C129 --patch-size 1024
        """
    )

    # Download mode
    parser.add_argument('--download-data', action='store_true',
                        help='Download VLA P-band tutorial data (26.46 GB)')
    parser.add_argument('--data-dir', default='./tutorial_data',
                        help='Directory for tutorial data download (default: ./tutorial_data)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download even if data exists')

    # Comparison mode
    parser.add_argument('--ms', help='Path to Hanning-smoothed MS (before flagging)')
    parser.add_argument('--model', help='Path to trained SAM-RFI model')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--calibrator', default='3C147', help='Calibrator field name')
    parser.add_argument('--target', default='3C129', help='Target field name')
    parser.add_argument('--patch-size', type=int, default=1024, help='SAM-RFI patch size')

    args = parser.parse_args()

    # Download mode
    if args.download_data:
        print("\nDownloading VLA P-band tutorial data...")
        try:
            ms_path = download_tutorial_data(
                output_dir=args.data_dir,
                force=args.force_download
            )
            print(f"\n✓ Data downloaded successfully to: {ms_path}")
            print("\nNote: This is the raw SDM data. You need to:")
            print("  1. Import with importasdm")
            print("  2. Apply Hanning smoothing")
            print("  3. Then use the resulting MS with this comparison script")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            sys.exit(1)

    # Comparison mode - require MS, model, output
    if not args.ms or not args.model or not args.output:
        parser.error("--ms, --model, and --output are required for comparison mode. "
                     "Use --download-data to download tutorial data first.")

    # Validate CASA and SAM-RFI availability
    if not CASA_AVAILABLE:
        print("ERROR: CASA tasks not available. Run this script within CASA.")
        sys.exit(1)

    if not SAMRFI_AVAILABLE:
        print("ERROR: SAM-RFI not installed. Run 'pip install -e .' in SAM-RFI directory.")
        sys.exit(1)

    # Run comparison
    comparison = FlaggingComparison(
        ms_path=args.ms,
        model_path=args.model,
        output_dir=args.output,
        calibrator_field=args.calibrator,
        target_field=args.target,
        patch_size=args.patch_size
    )

    comparison.run()


if __name__ == '__main__':
    main()
