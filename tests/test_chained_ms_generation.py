# tests/test_chained_ms_generation.py
import tempfile
import shutil
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from samrfi.datasets.synthetic_ms_legacy import ObservationConfig, RFIConfig
from samrfi.datasets.generators.casa_generator import CASATrainingGenerator

try:
    from casatools import table
    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False

@pytest.fixture(scope="class")
def ms_test_env(request):
    temp_dir = tempfile.mkdtemp()
    ms_path = Path(temp_dir) / "chained_test.ms"
    obs_config = ObservationConfig(
        num_antennas=3,
        num_spw=1,
        channels_per_spw=32,
        total_duration=120.0,
        integration_time=10.0,
        thermal_noise_sigma=1e-3
    )
    rfi_config = RFIConfig(
        broadband_probability=0.02,
        narrowband_lines=2,
        transient_events=1,
        periodic_signals=1,
        satellite_passes=1
    )
    generator = CASATrainingGenerator(obs_config)
    # Create MS structure (step 1)
    generator.create_training_ms(str(ms_path), rfi_config=None)
    # Attach to request for access in tests
    request.cls.ms_path = ms_path
    request.cls.temp_dir = temp_dir
    request.cls.obs_config = obs_config
    request.cls.rfi_config = rfi_config
    request.cls.generator = generator
    yield
    shutil.rmtree(temp_dir)

@pytest.mark.usefixtures("ms_test_env")
class TestChainedMSGeneration:
    def test_step1_create_ms_structure(self):
        assert self.ms_path.exists(), "MS file was not created"
        assert self.ms_path.is_dir(), "MS should be a directory"
        assert (self.ms_path / "table.dat").exists(), "MS table.dat missing"
        assert (self.ms_path / "ANTENNA").exists(), "ANTENNA table missing"

    def test_step2_verify_noise_filled(self):
        assert self.ms_path.exists(), "MS from step 1 not found"
        if CASA_AVAILABLE:
            tb = table()
            tb.open(str(self.ms_path))
            nrows = tb.nrows()
            data_sample = tb.getcell("DATA", 0)
            assert not np.all(data_sample == 0), "Data is all zeros - no noise"
            noise_level = np.std(np.abs(data_sample))
            expected_noise = self.obs_config.thermal_noise_sigma
            assert 0.3 * expected_noise < noise_level < 3.0 * expected_noise, \
                f"Noise level {noise_level} not reasonable for expected {expected_noise}"
            tb.close()
        else:
            print("CASA tools not available - cannot validate noise")

    def test_step3_add_rfi_to_existing_ms(self):
        assert self.ms_path.exists(), "MS from previous steps not found"
        rfi_ms_path = Path(self.temp_dir) / "rfi_test.ms"
        self.generator.create_training_ms(
            str(rfi_ms_path),
            rfi_config=self.rfi_config,
            include_rfi_flags=True
        )
        assert rfi_ms_path.exists(), "RFI MS was not created"
        if CASA_AVAILABLE:
            tb = table()
            tb.open(str(rfi_ms_path))
            flag_sample = tb.getcell("FLAG", 0)
            total_flags = np.sum(flag_sample)
            total_points = flag_sample.size
            assert total_flags > 0, "No flags were set - RFI not added"
            assert total_flags / total_points < 0.9, "Too many flags set"
            tb.close()
        else:
            print("CASA tools not available - cannot validate RFI")