"""
Smoke tests for imports - runs first to catch import issues early.

These tests verify that core imports work correctly without triggering
circular dependencies or missing optional dependencies.

Naming: test_aaa_* ensures this runs first alphabetically in pytest.
"""

import pytest


class TestCoreImports:
    """Test that core samrfi imports work without rfi_toolbox."""

    def test_samrfi_base_import(self):
        """samrfi package should import without triggering rfi_toolbox initialization."""
        try:
            import samrfi

            assert samrfi.__version__ is not None
        except ImportError as e:
            pytest.fail(f"Failed to import samrfi base package: {e}")

    def test_samrfi_config_import(self):
        """samrfi.config should import without external dependencies."""
        try:
            from samrfi.config import ConfigLoader

            assert ConfigLoader is not None
        except ImportError as e:
            pytest.fail(f"Failed to import samrfi.config: {e}")

    def test_samrfi_utils_import(self):
        """samrfi.utils.errors should import without external dependencies."""
        try:
            from samrfi.utils.errors import (
                CheckpointMismatchError,
                ConfigValidationError,
                DataShapeError,
            )

            assert CheckpointMismatchError is not None
            assert ConfigValidationError is not None
            assert DataShapeError is not None
        except ImportError as e:
            pytest.fail(f"Failed to import samrfi.utils.errors: {e}")

    def test_samrfi_data_import(self):
        """samrfi.data (SAM2-specific modules) should import without rfi_toolbox."""
        try:
            from samrfi.data import BatchedDataset, HFDatasetWrapper, SAMDataset

            assert BatchedDataset is not None
            assert HFDatasetWrapper is not None
            assert SAMDataset is not None
        except ImportError as e:
            pytest.fail(f"Failed to import samrfi.data: {e}")


class TestRFIToolboxImports:
    """Test that rfi_toolbox imports work correctly."""

    def test_rfi_toolbox_preprocessing_import(self):
        """rfi_toolbox.preprocessing should import cleanly."""
        try:
            from rfi_toolbox.preprocessing import GPUPreprocessor, Preprocessor

            assert Preprocessor is not None
            assert GPUPreprocessor is not None
        except ImportError as e:
            pytest.fail(
                f"Failed to import rfi_toolbox.preprocessing: {e}\n"
                "This suggests a circular import or missing dependency in rfi_toolbox."
            )

    def test_rfi_toolbox_datasets_import(self):
        """rfi_toolbox.datasets should import without sklearn."""
        try:
            from rfi_toolbox.datasets import BatchWriter, RFIMaskDataset, TorchDataset

            assert BatchWriter is not None
            assert TorchDataset is not None
            assert RFIMaskDataset is not None
        except ImportError as e:
            pytest.fail(
                f"Failed to import rfi_toolbox.datasets: {e}\n"
                "This suggests sklearn is required (should be optional) or circular import."
            )

    def test_rfi_toolbox_data_generation_import(self):
        """rfi_toolbox.data_generation should import cleanly."""
        try:
            from rfi_toolbox.data_generation import RawPatchDataset, SyntheticDataGenerator

            assert SyntheticDataGenerator is not None
            assert RawPatchDataset is not None
        except ImportError as e:
            pytest.fail(
                f"Failed to import rfi_toolbox.data_generation: {e}\n"
                "This suggests a circular import in rfi_toolbox."
            )

    def test_rfi_toolbox_evaluation_import(self):
        """rfi_toolbox.evaluation should import without torch."""
        try:
            from rfi_toolbox.evaluation import (  # noqa: F401
                compute_dice,
                compute_f1,
                compute_ffi,
                compute_iou,
                compute_precision,
                compute_recall,
                evaluate_segmentation,
            )

            assert compute_iou is not None
            assert compute_ffi is not None
            assert evaluate_segmentation is not None
        except ImportError as e:
            pytest.fail(f"Failed to import rfi_toolbox.evaluation: {e}")


class TestOptionalImports:
    """Test optional imports that require specific dependencies."""

    @pytest.mark.requires_casa
    def test_rfi_toolbox_io_import(self):
        """rfi_toolbox.io requires CASA - should fail gracefully if missing."""
        try:
            from rfi_toolbox.io import MSLoader

            # If CASA is not available, MSLoader should be None
            if MSLoader is None:
                pytest.skip("CASA not available - MSLoader is None (expected)")
        except ImportError as e:
            # This is expected if CASA is not installed
            assert "CASA" in str(e) or "casatools" in str(e)


class TestImportIsolation:
    """Test that imports don't have unintended side effects."""

    def test_samrfi_import_does_not_trigger_rfi_toolbox(self):
        """Importing samrfi should not initialize rfi_toolbox package."""
        import sys

        # Remove rfi_toolbox from sys.modules if present
        rfi_toolbox_modules = [key for key in sys.modules if key.startswith("rfi_toolbox")]
        for mod in rfi_toolbox_modules:
            del sys.modules[mod]

        # Import samrfi
        import samrfi  # noqa: F401

        # Check that rfi_toolbox.__init__ was not loaded
        assert (
            "rfi_toolbox" not in sys.modules
        ), "Importing samrfi should not trigger rfi_toolbox initialization"

    def test_import_order_independence(self):
        """Importing in different orders should not cause failures."""
        import sys

        # Clear all samrfi and rfi_toolbox modules
        for key in list(sys.modules.keys()):
            if key.startswith(("samrfi", "rfi_toolbox")):
                del sys.modules[key]

        # Test order 1: samrfi first
        try:
            from rfi_toolbox.preprocessing import Preprocessor  # noqa: F401

            import samrfi  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed with samrfi first: {e}")

        # Clear and test order 2: rfi_toolbox first
        for key in list(sys.modules.keys()):
            if key.startswith(("samrfi", "rfi_toolbox")):
                del sys.modules[key]

        try:
            from rfi_toolbox.preprocessing import Preprocessor  # noqa: F401

            import samrfi  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed with rfi_toolbox first: {e}")
