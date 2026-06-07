"""
Unit tests for data generators
"""

from unittest.mock import Mock

from rfi_toolbox.data_generation import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator"""

    def test_init(self):
        """Test initialization"""
        config = Mock()
        config.synthetic = {"num_samples": 10}
        config.processing = {"stretch": "SQRT"}

        generator = SyntheticDataGenerator(config)
        assert generator.config == config

    def test_bandpass_generation(self):
        """Test bandpass rolloff generation"""
        config = Mock()
        config.synthetic = {}
        config.processing = {}

        generator = SyntheticDataGenerator(config)
        bandpass = generator._generate_bandpass(1000, 8)

        assert len(bandpass) == 1000
        assert bandpass[500] == 1.0  # Center should be 1.0
        assert bandpass[0] < 0.1  # Edges should be rolled off
        assert bandpass[-1] < 0.1

    def test_narrowband_persistent(self):
        """Test narrowband persistent RFI generation"""
        config = Mock()
        config.synthetic = {}

        generator = SyntheticDataGenerator(config)
        signal, mask, params = generator._add_narrowband_persistent(2048, 512, 1000.0, {})

        assert signal.shape == (2048, 512)
        assert mask.shape == (2048, 512)
        assert mask.sum() > 0  # RFI should be present
        assert "center_freq" in params
        assert "bandwidth" in params

    def test_broadband_persistent(self):
        """Test broadband persistent RFI generation"""
        config = Mock()
        generator = SyntheticDataGenerator(config)
        signal, mask, params = generator._add_broadband_persistent(2048, 512, 1000.0, {})

        assert signal.shape == (2048, 512)
        assert mask.shape == (2048, 512)
        assert mask.sum() > 0
        assert "center_time" in params
        assert "time_width" in params

    def test_frequency_sweep(self):
        """Test frequency sweep RFI generation"""
        config = Mock()
        generator = SyntheticDataGenerator(config)
        signal, mask, params = generator._add_frequency_sweep(2048, 512, 1000.0, {})

        assert signal.shape == (2048, 512)
        assert mask.shape == (2048, 512)
        assert mask.sum() > 0
        assert "start_freq" in params
        assert "end_freq" in params
        assert "sweep_order" in params
        assert params["sweep_order"] in [1, 2]  # Linear or quadratic

    def test_narrowband_bursty(self):
        """Test narrowband bursty RFI generation"""
        config = Mock()
        generator = SyntheticDataGenerator(config)
        signal, mask, params = generator._add_narrowband_bursty(2048, 512, 1000.0, {})

        assert signal.shape == (2048, 512)
        assert mask.shape == (2048, 512)
        assert mask.sum() > 0
        assert "num_bursts" in params
        assert params["num_bursts"] >= 3

    def test_broadband_bursty(self):
        """Test broadband bursty RFI generation"""
        config = Mock()
        generator = SyntheticDataGenerator(config)
        signal, mask, params = generator._add_broadband_bursty(2048, 512, 1000.0, {})

        assert signal.shape == (2048, 512)
        assert mask.shape == (2048, 512)
        assert mask.sum() > 0
        assert "num_bursts" in params

    def test_narrowband_intermittent(self):
        """Test narrowband intermittent RFI generation"""
        config = Mock()
        generator = SyntheticDataGenerator(config)
        signal, mask, params = generator._add_narrowband_intermittent(2048, 512, 1000.0, {})

        assert signal.shape == (2048, 512)
        assert mask.shape == (2048, 512)
        assert mask.sum() > 0
        assert "period" in params
        assert "duty_cycle" in params

    def test_parse_rfi_config(self):
        """Test RFI configuration parsing"""
        config_dict = {
            "rfi_types": ["narrowband_persistent", "frequency_sweep"],
            "rfi_type_counts": {"narrowband_persistent": 3, "frequency_sweep": 2},
        }

        config = Mock()
        generator = SyntheticDataGenerator(config)
        rfi_config = generator._parse_rfi_config(config_dict)

        assert rfi_config["narrowband_persistent"]["count"] == 3
        assert rfi_config["frequency_sweep"]["count"] == 2

    def test_physical_scales(self):
        """Test that physical scales are realistic"""
        config = Mock()
        config.synthetic = {
            "num_samples": 1,
            "num_channels": 256,
            "num_times": 128,
            "noise_mjy": 1.0,
            "rfi_power_min": 1000.0,
            "rfi_power_max": 10000.0,
            "rfi_type_counts": {"narrowband_persistent": 1},
            "enable_bandpass_rolloff": False,
            "polarization_correlation": 0.0,
        }
        config.processing = {
            "stretch": "SQRT",
            "flag_sigma": 5,
            "patch_method": "patchify",
            "patch_size": 128,
            "num_patches": None,
            "apply_stretching": True,
            "augmentation": {"rotations": True},
        }

        generator = SyntheticDataGenerator(config)

        # Generate single sample to verify scales
        waterfall, exact_mask, rfi_params = generator._generate_single_sample(
            num_channels=256,
            num_times=128,
            noise_level=1.0,
            rfi_power_min=1000.0,
            rfi_power_max=10000.0,
            rfi_config={
                "narrowband_persistent": {"count": 1},
                "broadband_persistent": {"count": 0},
                "narrowband_intermittent": {"count": 0},
                "narrowband_bursty": {"count": 0},
                "broadband_bursty": {"count": 0},
                "frequency_sweep": {"count": 0},
            },
            enable_bandpass=False,
            bandpass_order=8,
            num_polarizations=4,
            pol_corr=0.0,
            synth_config=config.synthetic,
        )

        # Check shapes
        assert waterfall.shape == (1, 4, 256, 128)  # (1, pols, channels, times)
        assert exact_mask.shape == (1, 4, 256, 128)

        # Check RFI is much larger than noise
        import numpy as np

        _noise_level = 1.0
        rfi_pixels = waterfall[0, 0][exact_mask[0, 0]]
        clean_pixels = waterfall[0, 0][~exact_mask[0, 0]]

        if len(rfi_pixels) > 0 and len(clean_pixels) > 0:
            # Compare magnitudes (waterfall is complex)
            assert np.mean(np.abs(rfi_pixels)) > np.mean(np.abs(clean_pixels)) * 100  # RFI >> noise

    def test_polarization_correlation(self):
        """Test polarization correlation"""
        config = Mock()
        config.synthetic = {
            "rfi_type_counts": {"narrowband_persistent": 1},
            "enable_bandpass_rolloff": False,
            "polarization_correlation": 0.9,
        }

        generator = SyntheticDataGenerator(config)

        waterfall, exact_mask, _ = generator._generate_single_sample(
            num_channels=256,
            num_times=128,
            noise_level=1.0,
            rfi_power_min=1000.0,
            rfi_power_max=10000.0,
            rfi_config={
                "narrowband_persistent": {"count": 1},
                "broadband_persistent": {"count": 0},
                "narrowband_intermittent": {"count": 0},
                "narrowband_bursty": {"count": 0},
                "broadband_bursty": {"count": 0},
                "frequency_sweep": {"count": 0},
            },
            enable_bandpass=False,
            bandpass_order=8,
            num_polarizations=4,
            pol_corr=0.9,
            synth_config=config.synthetic,
        )

        # Check that RFI appears in both XX and YY
        assert exact_mask[0, 0].sum() > 0  # XX
        assert exact_mask[0, 1].sum() > 0  # YY
        # XY and YX should be clean
        assert exact_mask[0, 2].sum() == 0
        assert exact_mask[0, 3].sum() == 0
