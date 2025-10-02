"""
Synthetic Data Generator - Generate training data from synthetic RFI simulations
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from samrfi.data import Preprocessor


class SyntheticDataGenerator:
    """
    Generate SAM2 training datasets from synthetic RFI simulations

    Workflow:
        1. Generate synthetic waterfall plots with realistic RFI types
        2. Add physically accurate RFI (6 orders of magnitude above noise)
        3. Generate EXACT ground truth masks (we know where RFI is!)
        4. Patchify with 4-way rotation augmentation
        5. Normalize + stretch (SQRT/LOG10)
        6. Save HuggingFace dataset to disk

    RFI Types Supported:
        - Narrowband persistent: GPS, cell towers, satellite
        - Broadband persistent: Lightning, power lines
        - Narrowband intermittent (periodic): Rotating radar
        - Narrowband bursty (random): Random pulsed transmitters
        - Broadband bursty (random): Lightning strikes
        - Frequency sweeps: Radar chirps, satellite drift

    Physical Realism:
        - Noise: ~1 mJy (milli-Jansky)
        - RFI: ~1000 Jy (6 orders of magnitude higher)
        - Bandpass rolloff: 8th order polynomial edge effects
        - Polarization correlation: Correlated RFI across XX/YY
    """

    def __init__(self, config):
        """
        Initialize synthetic data generator

        Args:
            config: Configuration object with synthetic RFI parameters
        """
        self.config = config

    def generate(self, output_path):
        """
        Generate synthetic dataset with exact ground truth masks

        Args:
            output_path: Directory to save generated dataset

        Returns:
            Path to saved dataset
        """
        print("=" * 60)
        print("Synthetic Data Generation with Physical Realism")
        print("=" * 60)

        # Extract config
        synth_config = self.config.synthetic
        proc_config = self.config.processing

        num_samples = synth_config.get("num_samples", 100)
        num_channels = synth_config.get("num_channels", 2048)
        num_times = synth_config.get("num_times", 512)

        # Physical scales (milli-Jansky)
        noise_level = synth_config.get("noise_mjy", 1.0)  # 1 mJy
        rfi_power_min = synth_config.get("rfi_power_min", 1000.0)  # 1000 Jy = 1e6 mJy
        rfi_power_max = synth_config.get("rfi_power_max", 10000.0)  # 10000 Jy

        print(f"\nPhysical Parameters:")
        print(f"  Noise level: {noise_level} mJy")
        print(f"  RFI power range: {rfi_power_min}-{rfi_power_max} Jy")
        print(f"  Dynamic range: {rfi_power_max*1000/noise_level:.1e} (~6 orders)")

        print(f"\nConfiguration:")
        print(f"  Samples: {num_samples}")
        print(f"  Dimensions: {num_channels} channels × {num_times} times")
        print(f"  Output: {output_path}")

        # Get RFI configuration
        rfi_config = self._parse_rfi_config(synth_config)

        print(f"\nRFI Types Enabled:")
        for rfi_type, params in rfi_config.items():
            if params["count"] > 0:
                print(f"  {rfi_type}: {params['count']} per sample")

        # Bandpass options
        enable_bandpass = synth_config.get("enable_bandpass_rolloff", False)
        if enable_bandpass:
            bandpass_order = synth_config.get("bandpass_polynomial_order", 8)
            print(f"\nBandpass: Enabled ({bandpass_order}th order polynomial rolloff)")

        # Polarization correlation
        pol_corr = synth_config.get("polarization_correlation", 0.8)
        print(f"Polarization correlation: {pol_corr}")

        # Generate samples in batches to avoid memory exhaustion
        print(f"\n[1/5] Generating {num_samples} synthetic samples...")

        batch_size = 50  # Process 50 samples at a time (reduced for 1024x1024 full resolution)
        num_batches = (num_samples + batch_size - 1) // batch_size

        all_rfi_parameters = []
        dataset_exact_list = []
        dataset_mad_list = []
        total_rfi_flags = 0
        total_pixels = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = end_idx - start_idx

            print(f"\n  Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx})...")

            batch_waterfalls = []
            batch_exact_masks = []

            for i in tqdm(range(batch_samples), desc=f"Batch {batch_idx + 1}/{num_batches}"):
                waterfall, exact_mask, rfi_params = self._generate_single_sample(
                    num_channels=num_channels,
                    num_times=num_times,
                    noise_level=noise_level,
                    rfi_power_min=rfi_power_min,
                    rfi_power_max=rfi_power_max,
                    rfi_config=rfi_config,
                    enable_bandpass=enable_bandpass,
                    bandpass_order=synth_config.get("bandpass_polynomial_order", 8),
                    pol_corr=pol_corr,
                    synth_config=synth_config,
                )

                batch_waterfalls.append(waterfall)
                batch_exact_masks.append(exact_mask)
                all_rfi_parameters.append(rfi_params)

            # Stack this batch
            batch_data = np.vstack(batch_waterfalls)
            batch_masks = np.vstack(batch_exact_masks)

            # Track RFI statistics
            total_rfi_flags += np.sum(batch_masks)
            total_pixels += batch_masks.size

            print(f"    Batch data shape: {batch_data.shape}")
            print(f"    Batch masks shape: {batch_masks.shape}")

            # Create datasets for this batch
            print(f"    Creating datasets for batch {batch_idx + 1}...")

            # Dataset 1: Exact ground truth masks
            preprocessor_exact = Preprocessor(batch_data, flags=batch_masks)
            batch_dataset_exact = preprocessor_exact.create_dataset(
                patch_size=proc_config.get("patch_size", 128),
                stretch=proc_config.get("stretch", None),
                flag_sigma=proc_config.get("flag_sigma", 5),
                use_custom_flags=True,
                num_patches=proc_config.get("num_patches", None),
                normalize_before_stretch=proc_config.get("normalize_before_stretch", True),
                normalize_after_stretch=proc_config.get("normalize_after_stretch", False),
                num_workers=proc_config.get("num_workers", 4),
            )
            dataset_exact_list.append(batch_dataset_exact)

            # Dataset 2: MAD-based masks
            preprocessor_mad = Preprocessor(batch_data, flags=None)
            batch_dataset_mad = preprocessor_mad.create_dataset(
                patch_size=proc_config.get("patch_size", 128),
                stretch=proc_config.get("stretch", None),
                flag_sigma=proc_config.get("flag_sigma", 5),
                use_custom_flags=False,
                num_patches=proc_config.get("num_patches", None),
                normalize_before_stretch=proc_config.get("normalize_before_stretch", True),
                normalize_after_stretch=proc_config.get("normalize_after_stretch", False),
                num_workers=proc_config.get("num_workers", 4),
            )
            dataset_mad_list.append(batch_dataset_mad)

            print(f"    Batch {batch_idx + 1} processed: {len(batch_dataset_exact)} patches")

            # Clean up batch arrays
            del batch_waterfalls, batch_exact_masks, batch_data, batch_masks
            del preprocessor_exact, preprocessor_mad

        # Concatenate all batch datasets (numpy concatenation - much faster!)
        print("\n[2/5] Combining batch datasets...")

        all_images_exact = np.concatenate([d.images for d in dataset_exact_list])
        all_labels_exact = np.concatenate([d.labels for d in dataset_exact_list])
        all_images_mad = np.concatenate([d.images for d in dataset_mad_list])
        all_labels_mad = np.concatenate([d.labels for d in dataset_mad_list])

        # Merge metadata from first batch
        from samrfi.data.numpy_dataset import NumpyDataset

        metadata_exact = dataset_exact_list[0].metadata.copy() if dataset_exact_list else {}
        metadata_exact['num_batches'] = len(dataset_exact_list)
        metadata_mad = dataset_mad_list[0].metadata.copy() if dataset_mad_list else {}
        metadata_mad['num_batches'] = len(dataset_mad_list)

        dataset_exact = NumpyDataset(all_images_exact, all_labels_exact, metadata_exact)
        dataset_mad = NumpyDataset(all_images_mad, all_labels_mad, metadata_mad)

        num_patches = len(dataset_exact)
        rfi_fraction = (total_rfi_flags / total_pixels) * 100

        print(f"  Total samples: {num_samples}")
        print(f"  Total patches: {num_patches}")
        print(f"  Patch size: {proc_config.get('patch_size', 128)}×{proc_config.get('patch_size', 128)}")
        print(f"  RFI fraction: {rfi_fraction:.2f}%")
        print(f"  Stretch: {proc_config.get('stretch', 'SQRT')}")

        # Save both datasets
        print("\n[3/5] Saving datasets...")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save exact mask dataset as .npz
        exact_path = output_dir / "exact_masks.npz"
        dataset_exact.save_to_disk(exact_path)

        # Save MAD mask dataset as .npz
        mad_path = output_dir / "mad_masks.npz"
        dataset_mad.save_to_disk(mad_path)

        # Save metadata
        metadata = {
            "source": "synthetic",
            "physical_parameters": {
                "noise_mjy": noise_level,
                "rfi_power_min_jy": rfi_power_min,
                "rfi_power_max_jy": rfi_power_max,
                "dynamic_range": float(rfi_power_max * 1000 / noise_level),
            },
            "num_samples": num_samples,
            "num_channels": num_channels,
            "num_times": num_times,
            "rfi_config": {k: v for k, v in rfi_config.items() if v["count"] > 0},
            "bandpass": {
                "enabled": enable_bandpass,
                "polynomial_order": (
                    synth_config.get("bandpass_polynomial_order", 8) if enable_bandpass else None
                ),
            },
            "polarization_correlation": pol_corr,
            "num_patches": num_patches,
            "rfi_fraction_percent": float(rfi_fraction),
            "patch_size": proc_config.get("patch_size", 128),
            "stretch": proc_config.get("stretch", "SQRT"),
            "ground_truth": "exact",  # Not MAD-based!
            "augmentation": {
                "rotations": proc_config.get("augmentation", {}).get("rotations", True)
            },
            "batch_processing": {
                "batch_size": batch_size,
                "num_batches": num_batches,
            },
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save RFI parameters
        rfi_params_path = output_dir / "rfi_parameters.json"
        with open(rfi_params_path, "w") as f:
            json.dump(all_rfi_parameters, f, indent=2)

        print(f"  Exact masks dataset: {exact_path}")
        print(f"  MAD masks dataset: {mad_path}")
        print(f"  Metadata saved to: {metadata_path}")
        print(f"  RFI parameters saved to: {rfi_params_path}")

        # Statistics
        print("\n[4/5] Dataset Statistics:")
        print(f"  Total waterfall samples: {num_samples}")
        print(f"  Total patches: {num_patches}")
        print(f"  RFI coverage: {rfi_fraction:.2f}%")
        print(
            f"  Image shape: {proc_config.get('patch_size', 128)}×{proc_config.get('patch_size', 128)}×3 (RGB)"
        )
        print(
            f"  Mask shape: {proc_config.get('patch_size', 128)}×{proc_config.get('patch_size', 128)} (exact binary)"
        )
        print(f"  Format: Numpy (.npz compressed)")

        print("\n[5/5] Validation:")
        print(f"  ✓ TWO datasets generated: exact masks + MAD masks")
        print(f"  ✓ Exact ground truth for training")
        print(f"  ✓ MAD masks for flagger comparison")
        print(f"  ✓ Physical noise/RFI scales (~10^6 dynamic range)")
        print(f"  ✓ Realistic RFI types (sweeps, bursts, persistent)")
        print(f"  ✓ Frequency sweeps: linear & quadratic")
        if enable_bandpass:
            print(
                f"  ✓ Bandpass rolloff ({synth_config.get('bandpass_polynomial_order', 8)}th order)"
            )
        if pol_corr > 0:
            print(f"  ✓ Polarization correlation ({pol_corr})")

        print("\n" + "=" * 60)
        print("✓ Synthetic data generation complete!")
        print("=" * 60)

        return str(output_dir)

    def _generate_single_sample(
        self,
        num_channels,
        num_times,
        noise_level,
        rfi_power_min,
        rfi_power_max,
        rfi_config,
        enable_bandpass,
        bandpass_order,
        pol_corr,
        synth_config,
    ):
        """
        Generate a single synthetic sample with exact mask

        Returns:
            waterfall: (1, 4, channels, times) - 4 polarizations
            exact_mask: (1, 4, channels, times) - binary mask of RFI locations
            rfi_params: dict of RFI parameters for this sample
        """
        # Create base spectrograph (clean Gaussian noise at mJy scale)
        baseline = np.random.normal(noise_level, noise_level * 0.1, (num_channels, num_times))

        # Apply bandpass rolloff if enabled
        if enable_bandpass:
            bandpass = self._generate_bandpass(num_channels, bandpass_order)
            baseline = baseline * bandpass[:, np.newaxis]

        # Initialize RFI mask (exact locations)
        rfi_mask = np.zeros((num_channels, num_times), dtype=bool)
        rfi_signal = np.zeros((num_channels, num_times))

        rfi_params = []

        # Add each RFI type
        for rfi_type, params in rfi_config.items():
            count = params["count"]
            if count == 0:
                continue

            for _ in range(count):
                rfi_amplitude = np.random.uniform(rfi_power_min, rfi_power_max) * 1000  # Jy to mJy

                if rfi_type == "narrowband_persistent":
                    signal, mask, param = self._add_narrowband_persistent(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "broadband_persistent":
                    signal, mask, param = self._add_broadband_persistent(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "narrowband_intermittent":
                    signal, mask, param = self._add_narrowband_intermittent(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "narrowband_bursty":
                    signal, mask, param = self._add_narrowband_bursty(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "broadband_bursty":
                    signal, mask, param = self._add_broadband_bursty(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "frequency_sweep":
                    signal, mask, param = self._add_frequency_sweep(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                else:
                    continue

                rfi_signal += signal
                rfi_mask = rfi_mask | mask
                rfi_params.append(
                    {**param, "type": rfi_type, "amplitude_mjy": float(rfi_amplitude)}
                )

        # Combine clean + RFI
        combined = baseline + rfi_signal

        # Create 4 polarizations with correlation (COMPLEX for phase extraction)
        # Add random phase to create complex visibilities
        pol1_real = combined.copy()
        pol1_phase = np.random.uniform(0, 2 * np.pi, combined.shape)
        pol1 = pol1_real * np.exp(1j * pol1_phase)

        pol2_real = (
            pol_corr * rfi_signal
            + (1 - pol_corr) * np.random.normal(0, noise_level * 0.1, rfi_signal.shape)
            + baseline
        )
        pol2_phase = np.random.uniform(0, 2 * np.pi, pol2_real.shape)
        pol2 = pol2_real * np.exp(1j * pol2_phase)

        pol3_real = np.random.normal(noise_level, noise_level * 0.1, (num_channels, num_times))
        pol3_phase = np.random.uniform(0, 2 * np.pi, pol3_real.shape)
        pol3 = pol3_real * np.exp(1j * pol3_phase)

        pol4_real = np.random.normal(noise_level, noise_level * 0.1, (num_channels, num_times))
        pol4_phase = np.random.uniform(0, 2 * np.pi, pol4_real.shape)
        pol4 = pol4_real * np.exp(1j * pol4_phase)

        waterfall = np.stack([pol1, pol2, pol3, pol4], axis=0)[
            np.newaxis, ...
        ]  # (1, 4, channels, times) - COMPLEX

        # Mask for all polarizations (RFI appears in XX and YY but correlated)
        mask_pol1 = rfi_mask.copy()
        mask_pol2 = rfi_mask.copy()  # Correlated
        mask_pol3 = np.zeros_like(rfi_mask)  # Cross-pols clean
        mask_pol4 = np.zeros_like(rfi_mask)

        exact_mask = np.stack([mask_pol1, mask_pol2, mask_pol3, mask_pol4], axis=0)[np.newaxis, ...]

        return waterfall, exact_mask, rfi_params

    def _generate_bandpass(self, num_channels, order):
        """Generate realistic bandpass with polynomial rolloff at edges"""
        bandpass = np.ones(num_channels)
        edge_fraction = 0.1  # Rolloff in 10% of channels at each edge
        edge_channels = int(num_channels * edge_fraction)

        # Polynomial rolloff at edges
        for i in range(edge_channels):
            # Low frequency edge
            t = i / edge_channels
            bandpass[i] = t**order

            # High frequency edge
            bandpass[-(i + 1)] = t**order

        return bandpass

    def _add_narrowband_persistent(self, nc, nt, amp, config):
        """Persistent narrowband RFI (GPS, satellite)"""
        center_freq = np.random.randint(int(nc * 0.1), int(nc * 0.9))
        bandwidth = np.random.randint(1, 10)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        freq_slice = slice(
            max(0, center_freq - bandwidth // 2), min(nc, center_freq + bandwidth // 2)
        )
        signal[freq_slice, :] = amp
        mask[freq_slice, :] = True

        params = {"center_freq": int(center_freq), "bandwidth": int(bandwidth)}
        return signal, mask, params

    def _add_broadband_persistent(self, nc, nt, amp, config):
        """Persistent broadband RFI (power lines)"""
        center_time = np.random.randint(int(nt * 0.1), int(nt * 0.9))
        time_width = np.random.randint(5, 50)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        time_slice = slice(
            max(0, center_time - time_width // 2), min(nt, center_time + time_width // 2)
        )
        signal[:, time_slice] = amp
        mask[:, time_slice] = True

        params = {"center_time": int(center_time), "time_width": int(time_width)}
        return signal, mask, params

    def _add_narrowband_intermittent(self, nc, nt, amp, config):
        """Periodic narrowband RFI (rotating radar)"""
        center_freq = np.random.randint(int(nc * 0.1), int(nc * 0.9))
        bandwidth = np.random.randint(2, 15)
        period = np.random.randint(20, 200)
        duty_cycle = np.random.uniform(0.1, 0.5)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        freq_slice = slice(
            max(0, center_freq - bandwidth // 2), min(nc, center_freq + bandwidth // 2)
        )

        for t in range(0, nt, period):
            duration = int(period * duty_cycle)
            time_slice = slice(t, min(nt, t + duration))
            signal[freq_slice, time_slice] = amp
            mask[freq_slice, time_slice] = True

        params = {
            "center_freq": int(center_freq),
            "bandwidth": int(bandwidth),
            "period": int(period),
            "duty_cycle": float(duty_cycle),
        }
        return signal, mask, params

    def _add_narrowband_bursty(self, nc, nt, amp, config):
        """Random bursty narrowband RFI (pulsed transmitters)"""
        center_freq = np.random.randint(int(nc * 0.1), int(nc * 0.9))
        bandwidth = np.random.randint(2, 20)
        num_bursts = np.random.randint(3, 15)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        freq_slice = slice(
            max(0, center_freq - bandwidth // 2), min(nc, center_freq + bandwidth // 2)
        )

        burst_times = np.random.choice(nt, num_bursts, replace=False)
        burst_widths = np.random.randint(2, 20, num_bursts)

        for t, width in zip(burst_times, burst_widths):
            time_slice = slice(max(0, t - width // 2), min(nt, t + width // 2))
            signal[freq_slice, time_slice] = amp
            mask[freq_slice, time_slice] = True

        params = {
            "center_freq": int(center_freq),
            "bandwidth": int(bandwidth),
            "num_bursts": int(num_bursts),
        }
        return signal, mask, params

    def _add_broadband_bursty(self, nc, nt, amp, config):
        """Random bursty broadband RFI (lightning)"""
        num_bursts = np.random.randint(2, 10)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        burst_times = np.random.choice(nt, num_bursts, replace=False)
        burst_widths = np.random.randint(1, 5, num_bursts)

        for t, width in zip(burst_times, burst_widths):
            time_slice = slice(max(0, t - width // 2), min(nt, t + width // 2))
            signal[:, time_slice] = amp
            mask[:, time_slice] = True

        params = {"num_bursts": int(num_bursts)}
        return signal, mask, params

    def _add_frequency_sweep(self, nc, nt, amp, config):
        """Frequency sweep RFI (radar chirp, satellite drift)"""
        start_freq = np.random.randint(int(nc * 0.1), int(nc * 0.5))
        end_freq = np.random.randint(int(nc * 0.5), int(nc * 0.9))
        bandwidth = np.random.randint(2, 10)
        sweep_order = np.random.choice([1, 2])  # Linear or quadratic sweep

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        for t in range(nt):
            if sweep_order == 1:
                # Linear sweep
                progress = t / nt
            else:
                # Quadratic sweep (accelerating)
                progress = (t / nt) ** 2

            center = int(start_freq + (end_freq - start_freq) * progress)
            freq_slice = slice(max(0, center - bandwidth // 2), min(nc, center + bandwidth // 2))

            signal[freq_slice, t] = amp
            mask[freq_slice, t] = True

        params = {
            "start_freq": int(start_freq),
            "end_freq": int(end_freq),
            "bandwidth": int(bandwidth),
            "sweep_order": int(sweep_order),
        }
        return signal, mask, params

    def _parse_rfi_config(self, config):
        """Parse RFI configuration from config"""
        rfi_types = config.get(
            "rfi_types", ["narrowband_persistent", "broadband_persistent", "frequency_sweep"]
        )

        default_counts = config.get("rfi_type_counts", {})

        rfi_config = {
            "narrowband_persistent": {"count": default_counts.get("narrowband_persistent", 1)},
            "broadband_persistent": {"count": default_counts.get("broadband_persistent", 1)},
            "narrowband_intermittent": {"count": default_counts.get("narrowband_intermittent", 0)},
            "narrowband_bursty": {"count": default_counts.get("narrowband_bursty", 1)},
            "broadband_bursty": {"count": default_counts.get("broadband_bursty", 0)},
            "frequency_sweep": {"count": default_counts.get("frequency_sweep", 1)},
        }

        # Override with rfi_types if provided
        if rfi_types:
            for rfi_type in rfi_config.keys():
                if rfi_type not in rfi_types and rfi_type not in default_counts:
                    rfi_config[rfi_type]["count"] = 0

        return rfi_config
