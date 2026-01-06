"""
RFI Predictor - Apply trained SAM2 models to new data.

This module provides inference capabilities for trained SAM2 models applied to
radio frequency interference (RFI) detection tasks. It supports both single-pass
and iterative flagging with progressive cleaning.

Classes
-------
RFIPredictor
    Apply trained SAM2 model to predict RFI flags on measurement sets or arrays.

Functions
---------
_patch_sam2_view_to_reshape
    Monkey-patch transformers Sam2Model to fix view/reshape bug.

Examples
--------
Single-pass prediction on measurement set:

>>> from samrfi.inference import RFIPredictor
>>> predictor = RFIPredictor(model_path='./models/sam2_rfi.pth')
>>> flags = predictor.predict_ms('observation.ms', patch_size=128, stretch='SQRT')

Iterative prediction for progressive cleaning:

>>> flags = predictor.predict_iterative(
...     'observation.ms',
...     num_iterations=3,
...     patch_size=128
... )

Direct array prediction:

>>> import numpy as np
>>> data = np.random.randn(10, 2, 512, 512) + 1j * np.random.randn(10, 2, 512, 512)
>>> flags = predictor.predict_array(data, patch_size=512)

Notes
-----
The predictor handles automatic preprocessing, patching, prediction, and
reconstruction of flags. It validates preprocessing parameters against
checkpoint metadata to ensure consistency between training and inference.

See Also
--------
samrfi.training.sam2_trainer : Training module for SAM2 models
samrfi.data.preprocessor : Data preprocessing pipeline
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Sam2Model, Sam2Processor

from samrfi.data import AdaptivePatcher, Preprocessor, SAMDataset
from samrfi.utils import logger
from samrfi.utils.errors import CheckpointMismatchError


# Monkey-patch transformers Sam2Model to fix view/reshape bug
# The bug: feat.permute(1, 2, 0).view(...) fails because permute makes tensor non-contiguous
# Fix: Replace view() with reshape() which handles non-contiguous tensors
def _patch_sam2_view_to_reshape() -> None:
    """
    Patch Sam2Model.forward to use reshape instead of view after permute operations.

    This function monkey-patches the transformers Sam2Model to handle non-contiguous
    tensors that result from permute operations. The original implementation uses
    view() which fails on non-contiguous tensors; this patch falls back to reshape()
    which handles both contiguous and non-contiguous tensors.

    Notes
    -----
    This is a workaround for a bug in transformers Sam2Model.forward where
    ``tensor.permute(1, 2, 0).view(...)`` fails with RuntimeError because
    permute makes the tensor non-contiguous.

    The patch temporarily replaces torch.Tensor.view with a safe version that
    falls back to reshape() when view() fails, then restores the original view()
    after the forward pass.

    Examples
    --------
    This function is called automatically at module import time:

    >>> # Patch is already applied when you import the module
    >>> from samrfi.inference import RFIPredictor
    >>> # Sam2Model.forward now uses safe view operations
    """
    import transformers.models.sam2.modeling_sam2 as sam2_module

    # Save original forward method
    original_forward = sam2_module.Sam2Model.forward

    def patched_forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Wrapped forward that ensures tensors are contiguous before view operations.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to original forward method.
        **kwargs : Any
            Keyword arguments passed to original forward method.

        Returns
        -------
        Any
            Output from original forward method.
        """
        # Temporarily replace tensor.view with a safe version
        original_view = torch.Tensor.view

        def safe_view(tensor: torch.Tensor, *shape: int) -> torch.Tensor:
            """
            Use reshape instead of view to handle non-contiguous tensors.

            Parameters
            ----------
            tensor : torch.Tensor
                Input tensor to reshape.
            *shape : int
                Target shape dimensions.

            Returns
            -------
            torch.Tensor
                Reshaped tensor.

            Raises
            ------
            RuntimeError
                If reshape also fails (non-view-related error).
            """
            try:
                return original_view(tensor, *shape)
            except RuntimeError as e:
                if "view size is not compatible" in str(e):
                    # Fall back to reshape which handles non-contiguous tensors
                    return tensor.reshape(*shape)
                raise

        # Monkey-patch view for this forward pass
        torch.Tensor.view = safe_view
        try:
            result = original_forward(self, *args, **kwargs)
        finally:
            # Restore original view
            torch.Tensor.view = original_view

        return result

    # Apply the patch
    sam2_module.Sam2Model.forward = patched_forward


# Apply patch at module load time
_patch_sam2_view_to_reshape()


class RFIPredictor:
    """
    Apply trained SAM2 model to predict RFI flags.

    This class provides inference capabilities for trained SAM2 models, supporting
    both single-pass and iterative flagging strategies. Iterative flagging performs
    multiple passes where each iteration finds fainter RFI that was hidden by
    brighter RFI in previous passes.

    Parameters
    ----------
    model_path : str or Path
        Path to trained model checkpoint (.pth file) OR HuggingFace repo ID
        (e.g., 'preshanth/sam-rfi-models/large').
    sam_checkpoint : str, default='large'
        SAM2 checkpoint size: 'tiny', 'small', 'base_plus', or 'large'.
        Must match the architecture used during training.
    device : str, default='cuda'
        Compute device for inference: 'cuda' or 'cpu'.
    batch_size : int, default=4
        Batch size for inference. Larger batches are faster but use more memory.
    allow_partial_load : bool, default=False
        If True, allow loading checkpoints with shape mismatches (not recommended).
    auto_select_sam : bool, default=False
        If True, automatically select SAM variant that best matches checkpoint.

    Attributes
    ----------
    model_path : Path
        Path to loaded model checkpoint.
    device : str
        Compute device being used.
    batch_size : int
        Batch size for inference.
    processor : Sam2Processor
        HuggingFace processor for SAM2 model.
    model : Sam2Model
        Loaded SAM2 model with trained weights.
    checkpoint_preprocessing : dict
        Preprocessing metadata from checkpoint for validation.

    Raises
    ------
    ValueError
        If checkpoint format is unrecognized or shape mismatches are detected.
    FileNotFoundError
        If local model_path doesn't exist.

    Examples
    --------
    Single-pass prediction on measurement set:

    >>> predictor = RFIPredictor(model_path='./models/sam2_rfi.pth')
    >>> flags = predictor.predict_ms('observation.ms', patch_size=128, stretch='SQRT')

    Iterative prediction for progressive cleaning:

    >>> flags = predictor.predict_iterative(
    ...     'observation.ms',
    ...     num_iterations=3,
    ...     patch_size=128
    ... )

    Direct array prediction:

    >>> import numpy as np
    >>> data = np.random.randn(10, 2, 512, 512) + 1j * np.random.randn(10, 2, 512, 512)
    >>> flags = predictor.predict_array(data, patch_size=512)

    Load from HuggingFace Hub:

    >>> predictor = RFIPredictor(model_path='preshanth/sam-rfi-models/large')
    >>> flags = predictor.predict_ms('observation.ms')

    Notes
    -----
    The predictor validates preprocessing parameters (patch_size, stretch, etc.)
    against checkpoint metadata to ensure consistency between training and inference.
    Critical parameters like patch_size must match exactly, while non-critical
    parameters like stretch function will generate warnings if mismatched.

    See Also
    --------
    samrfi.training.sam2_trainer : Training module for SAM2 models
    samrfi.data.preprocessor : Data preprocessing pipeline
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        sam_checkpoint: str = "large",
        device: str = "cuda",
        batch_size: int = 4,
        allow_partial_load: bool = False,
        auto_select_sam: bool = False,
    ) -> None:
        """
        Initialize RFI predictor with trained SAM2 model.

        Parameters
        ----------
        model_path : str or Path
            Path to trained model checkpoint (.pth file) OR HuggingFace repo ID
            (e.g., 'preshanth/sam-rfi-models/large').
        sam_checkpoint : str, default='large'
            SAM2 checkpoint size: 'tiny', 'small', 'base_plus', or 'large'.
            Must match the architecture used during training.
        device : str, default='cuda'
            Compute device for inference: 'cuda' or 'cpu'.
        batch_size : int, default=4
            Batch size for inference. Larger batches are faster but use more memory.
        allow_partial_load : bool, default=False
            If True, allow loading checkpoints with shape mismatches. Not recommended
            unless you know what you're doing. May lead to poor performance.
        auto_select_sam : bool, default=False
            If True, automatically select SAM variant that best matches checkpoint.
            This tests all available variants (tiny/small/base_plus/large) and
            selects the one with fewest mismatches. May be slow on first run.

        Raises
        ------
        ValueError
            If checkpoint format is unrecognized or shape mismatches are detected
            without allow_partial_load=True.
        FileNotFoundError
            If local model_path doesn't exist.

        Notes
        -----
        For HuggingFace models, the model is downloaded to the local HF cache
        (respects HF_HOME environment variable).
        """
        # Smart detection: local path OR HuggingFace repo ID
        model_path_str = str(model_path)
        if "/" in model_path_str and not Path(model_path).exists():
            # Looks like HF repo ID (contains '/') and not a local path
            logger.info(f"Detected HuggingFace model: {model_path}")

            # Support both forms:
            # 1. "user/repo/large" → extract repo and size
            # 2. "user/repo" → use sam_checkpoint param for size
            if model_path_str.endswith(("tiny", "small", "base_plus", "large")):
                repo_id = model_path_str.rsplit("/", 1)[0]
                model_size = model_path_str.rsplit("/", 1)[1]
            else:
                repo_id = model_path
                model_size = sam_checkpoint

            model_path = self._download_from_hf(repo_id, model_size)

        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = batch_size
        self.auto_select_sam = auto_select_sam

        # Map checkpoint names to HuggingFace model names
        checkpoint_map = {
            "tiny": "facebook/sam2-hiera-tiny",
            "small": "facebook/sam2-hiera-small",
            "base_plus": "facebook/sam2-hiera-base-plus",
            "large": "facebook/sam2-hiera-large",
        }

        model_name = checkpoint_map.get(sam_checkpoint, checkpoint_map["large"])

        logger.info(f"Loading SAM2 model: {model_name}")

        # Load processor and model
        self.processor = Sam2Processor.from_pretrained(model_name)
        self.model = Sam2Model.from_pretrained(model_name)

        # Load trained weights
        # Note: We load the SAM2 architecture via HuggingFace (Sam2Model.from_pretrained)
        # because it provides the model class and pretrained backbone. The file you pass
        # via `model_path` is expected to be either a plain state_dict (mapping of tensor
        # names -> tensors) or a full training checkpoint dict containing a 'model_state_dict'
        # (and possibly optimizer state, epoch, metadata). We support both formats here.
        logger.info(f"Loading trained weights from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=device)

        # Resolve state_dict from common wrapper formats
        state_dict = None
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Heuristic: if all values are tensors, treat as state_dict
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                else:
                    # Try a few common candidate keys
                    for candidate in ("model", "model_state", "model_state_dict"):
                        if candidate in checkpoint:
                            state_dict = checkpoint[candidate]
                            break
        else:
            # checkpoint is likely a state_dict mapping
            state_dict = checkpoint

        if state_dict is None:
            raise ValueError(
                f"Unrecognized checkpoint format when loading {self.model_path}. Keys: {list(checkpoint.keys())}"
            )

        # Helper: if auto-selection is enabled we will compare the checkpoint against
        # the available SAM variants and choose the one with the fewest mismatches.
        def _variant_score(candidate_variant):
            model_name_c = checkpoint_map.get(candidate_variant)
            try:
                m_c = Sam2Model.from_pretrained(model_name_c)
            except Exception as e:
                logger.warning(f"Failed to instantiate model for variant {candidate_variant}: {e}")
                return (1e9, None)  # very bad score
            mism, missing, unexpected = _compare_to_model(m_c)
            score = len(mism) + len(missing) + len(unexpected)
            return (score, (mism, missing, unexpected))

        # Auto-selection helper exposed as method for potential reuse
        def _auto_select_internal():
            candidates = ["tiny", "small", "base_plus", "large"]
            scores = []
            for c in candidates:
                logger.info(f"Testing SAM variant: {c} ...")
                s, details = _variant_score(c)
                scores.append((s, c, details))
            scores.sort()
            best = scores[0]
            if best[0] >= 1e9:
                return None
            return best[1]

        # Attach method to self for later usage
        self._auto_select_sam_variant = lambda sd: (lambda: _auto_select_internal())()

        # Before loading, detect obvious shape mismatches and missing/unexpected keys so we can
        # provide a clear error message rather than silently continuing with partially-initialized
        # weights which often leads to silent bad results.
        def _compare_to_model(model_obj):
            ms = model_obj.state_dict()
            mism = []
            for k, v in state_dict.items():
                if k in ms and isinstance(v, torch.Tensor) and v.shape != ms[k].shape:
                    mism.append((k, tuple(v.shape), tuple(ms[k].shape)))
            missing = [k for k in ms.keys() if k not in state_dict]
            unexpected = [k for k in state_dict.keys() if k not in ms]
            return mism, missing, unexpected

        mismatched, missing_in_ckpt, unexpected_in_ckpt = _compare_to_model(self.model)

        # If there are shape mismatches and user asked for auto-selection, attempt to find best match
        if mismatched and self.auto_select_sam:
            logger.warning(
                "Shape mismatches detected; attempting to auto-select the SAM variant that best matches the checkpoint (this may download models and be slow)..."
            )
            best_variant = self._auto_select_sam_variant(state_dict)
            if best_variant:
                logger.info(
                    f"Auto-selected SAM variant: {best_variant}. Re-instantiating model and retrying load."
                )
                model_name = checkpoint_map.get(best_variant, checkpoint_map["large"])
                self.processor = Sam2Processor.from_pretrained(model_name)
                self.model = Sam2Model.from_pretrained(model_name)
                # Recompute mismatches against new model
                mismatched, missing_in_ckpt, unexpected_in_ckpt = _compare_to_model(self.model)
            else:
                logger.warning(
                    "Auto-selection failed to find a better match; proceeding to error handling."
                )

        # If there are shape mismatches, fail early unless user explicitly allows partial loads
        if mismatched and not allow_partial_load:
            msg_lines = [
                "Checkpoint / model architecture mismatch detected:",
                f"  - Mismatched parameter shapes: {len(mismatched)}",
                f"  - Missing keys in checkpoint: {len(missing_in_ckpt)}",
                f"  - Unexpected keys in checkpoint: {len(unexpected_in_ckpt)}",
                "First few mismatches (checkpoint_shape -> model_shape):",
            ]
            for k, ck_shape, model_shape in mismatched[:10]:
                msg_lines.append(f"  {k}: {ck_shape} -> {model_shape}")

            msg_lines.append("")
            msg_lines.append(
                "Suggestion: make sure the `sam_checkpoint` argument matches the architecture used when the checkpoint was created (tiny/small/base_plus/large),"
            )
            msg_lines.append(
                "or set `allow_partial_load=True` when constructing RFIPredictor to permit partial loading (not recommended unless you know what you're doing)."
            )

            raise ValueError("\n".join(msg_lines))

        # Attempt to load state dict. If mismatches exist but allow_partial_load=True we'll
        # load with strict=False and print a summary for the user.
        if mismatched and allow_partial_load:
            print(
                f"Warning: {len(mismatched)} parameter shape mismatches detected; loading with strict=False (partial init)."
            )
            load_result = self.model.load_state_dict(state_dict, strict=False)
            # load_result is a namedtuple: (missing_keys, unexpected_keys)
            print(
                f"  Missing keys: {len(load_result.missing_keys)}; Unexpected keys: {len(load_result.unexpected_keys)}"
            )
        else:
            # No mismatches (or allowed above) — do a strict load
            self.model.load_state_dict(state_dict)

        # Store preprocessing metadata from checkpoint for validation
        self.checkpoint_preprocessing = checkpoint.get("preprocessing", {})

        # Move to device
        self.model.to(device)
        self.model.eval()

        logger.info(f"✓ Model loaded on {device}")

        # Display preprocessing info if available
        if self.checkpoint_preprocessing:
            print("\nCheckpoint preprocessing config:")
            for key, value in self.checkpoint_preprocessing.items():
                line = f"  {key}: {value}"
                logger.info(line)
                print(line)

    def _validate_preprocessing_params(
        self,
        patch_size: int,
        stretch: Optional[str],
        normalize_before_stretch: bool = False,
        normalize_after_stretch: bool = False,
    ) -> None:
        """
        Validate inference preprocessing parameters against checkpoint metadata.

        This method compares inference preprocessing parameters with the metadata
        stored in the checkpoint during training. Critical parameters (patch_size)
        must match exactly, while non-critical parameters (stretch, normalization)
        generate warnings if mismatched.

        Parameters
        ----------
        patch_size : int
            Patch size for inference (128, 256, 512, or 1024).
        stretch : str or None
            Stretch function: 'SQRT', 'LOG10', or None.
        normalize_before_stretch : bool, default=False
            Whether to normalize before applying stretch function.
        normalize_after_stretch : bool, default=False
            Whether to normalize after applying stretch function.

        Raises
        ------
        CheckpointMismatchError
            If patch_size doesn't match checkpoint metadata.

        Notes
        -----
        If the checkpoint doesn't contain preprocessing metadata (old checkpoint),
        validation is skipped silently.

        Warnings are printed to both logger and stdout for visibility during
        testing and user workflows.
        """
        if not self.checkpoint_preprocessing:
            # No metadata in checkpoint (old checkpoint), skip validation
            return

        # Critical: patch_size must match
        checkpoint_patch_size = self.checkpoint_preprocessing.get("patch_size")
        if (
            checkpoint_patch_size
            and checkpoint_patch_size != "unknown"
            and checkpoint_patch_size != patch_size
        ):
            raise CheckpointMismatchError(
                param_name="patch_size",
                checkpoint_value=checkpoint_patch_size,
                inference_value=patch_size,
            )

        # Warning: stretch function should match
        checkpoint_stretch = self.checkpoint_preprocessing.get("stretch")
        if checkpoint_stretch is not None and checkpoint_stretch != stretch:
            warning_msg = (
                f"Stretch function mismatch: model trained with stretch={checkpoint_stretch}, "
                f"inference using stretch={stretch}. This may reduce accuracy."
            )
            logger.warning(warning_msg)
            # Also print to stdout so tests and users see a clear WARNING line
            print(f"WARNING: {warning_msg}")

        # Info: normalization parameters (less critical for synthetic data)
        checkpoint_norm_before = self.checkpoint_preprocessing.get("normalize_before_stretch")
        checkpoint_norm_after = self.checkpoint_preprocessing.get("normalize_after_stretch")

        if (
            checkpoint_norm_before is not None
            and checkpoint_norm_before != normalize_before_stretch
        ):
            logger.info(
                f"Note: normalize_before_stretch differs "
                f"(training={checkpoint_norm_before}, inference={normalize_before_stretch})"
            )

        if checkpoint_norm_after is not None and checkpoint_norm_after != normalize_after_stretch:
            logger.info(
                f"Note: normalize_after_stretch differs "
                f"(training={checkpoint_norm_after}, inference={normalize_after_stretch})"
            )

    def _preprocess_data(
        self,
        data: NDArray[np.complexfloating],
        patch_size: int,
        stretch: Optional[str],
        enable_augmentation: bool,
        normalize_before_stretch: bool,
        normalize_after_stretch: bool,
    ) -> Any:
        """
        Create preprocessed dataset from data array.

        Applies the full preprocessing pipeline to convert raw complex visibility
        data into a dataset ready for SAM2 prediction. This includes patchification,
        feature extraction, stretching, and normalization.

        Parameters
        ----------
        data : ndarray of complex
            Complex visibility data with shape (baselines, pols, channels, times).
        patch_size : int
            Patch size for prediction (128, 256, 512, or 1024).
        stretch : str or None
            Stretch function: 'SQRT', 'LOG10', or None.
        enable_augmentation : bool
            If True, enable rotation augmentation (4-way transforms).
        normalize_before_stretch : bool
            If True, normalize before applying stretch function.
        normalize_after_stretch : bool
            If True, normalize after applying stretch function.

        Returns
        -------
        Dataset
            HuggingFace Dataset ready for prediction with preprocessed patches.

        Notes
        -----
        The `inference_mode=True` flag is critical for preserving patch order
        during reconstruction. This ensures patches can be reassembled into
        the correct positions in the full data array.
        """
        preprocessor = Preprocessor(data, flags=None)
        dataset = preprocessor.create_dataset(
            patch_size=patch_size,
            stretch=stretch,
            flag_sigma=5,
            use_custom_flags=False,
            enable_augmentation=enable_augmentation,
            augmentation_rotations=1,
            normalize_before_stretch=normalize_before_stretch,
            normalize_after_stretch=normalize_after_stretch,
            inference_mode=True,  # CRITICAL: Preserve patch order for reconstruction
        )
        return dataset

    def predict_array(
        self,
        data: NDArray[np.complexfloating],
        patch_size: int = 1024,
        stretch: Optional[str] = None,
        enable_augmentation: bool = False,
        normalize_before_stretch: bool = False,
        normalize_after_stretch: bool = False,
        return_probabilities: bool = False,
        threshold: Optional[float] = None,
        save_probabilities: Optional[Union[str, Path]] = None,
    ) -> NDArray[Union[np.bool_, np.float32]]:
        """
        Predict RFI flags on numpy array directly without measurement set I/O.

        This method provides a pure-Python interface for RFI prediction, accepting
        complex visibility data as a numpy array and returning predicted flags or
        probability maps.

        Parameters
        ----------
        data : ndarray of complex
            Complex visibility data with shape (baselines, pols, channels, times).
        patch_size : int, default=1024
            Patch size for prediction (128, 256, 512, or 1024).
            Must match the patch_size used during training.
        stretch : str or None, default=None
            Stretch function: 'SQRT', 'LOG10', or None.
            Should match the stretch used during training.
        enable_augmentation : bool, default=False
            If True, enable 4-way rotation augmentation during inference.
            Generally False for inference (augmentation is for training).
        normalize_before_stretch : bool, default=False
            If True, normalize before applying stretch function.
        normalize_after_stretch : bool, default=False
            If True, normalize after applying stretch function.
        return_probabilities : bool, default=False
            If True, return continuous probabilities [0,1] instead of binary flags.
        threshold : float or None, default=None
            Probability threshold for RFI detection. If None, uses adaptive
            threshold (mean of probability distribution).
        save_probabilities : str or Path or None, default=None
            If provided, save probability maps to this path (.npy file).

        Returns
        -------
        ndarray of bool or float32
            Predicted RFI flags (bool) or probabilities (float32) with shape
            matching input data (baselines, pols, channels, times).

        Examples
        --------
        >>> import numpy as np
        >>> predictor = RFIPredictor(model_path='./models/sam2_rfi.pth')
        >>> data = np.random.randn(10, 2, 512, 512) + 1j * np.random.randn(10, 2, 512, 512)
        >>> flags = predictor.predict_array(data, patch_size=512)
        >>> print(f"Flagged {np.sum(flags)/flags.size*100:.2f}% of data")

        Return probabilities instead of binary flags:

        >>> probs = predictor.predict_array(
        ...     data,
        ...     patch_size=512,
        ...     return_probabilities=True
        ... )
        >>> print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")

        Save probability maps for later analysis:

        >>> flags = predictor.predict_array(
        ...     data,
        ...     patch_size=512,
        ...     save_probabilities='rfi_probabilities.npy'
        ... )

        Notes
        -----
        The predictor validates preprocessing parameters against checkpoint metadata.
        Critical parameters like patch_size must match exactly, while non-critical
        parameters like stretch function will generate warnings if mismatched.
        """
        logger.info(f"\n{'='*60}")
        logger.info("RFI Prediction - Array Mode")
        logger.info(f"{'='*60}")

        # Validate preprocessing parameters against checkpoint
        self._validate_preprocessing_params(
            patch_size, stretch, normalize_before_stretch, normalize_after_stretch
        )

        data_shape = data.shape
        logger.info(f"  Input shape: {data_shape}")
        logger.info(f"  Data dtype: {data.dtype}, complex: {np.iscomplexobj(data)}")

        # Preprocess (pass complex data directly - Preprocessor will extract features)
        logger.info("\nPreprocessing data...")
        dataset = self._preprocess_data(
            data,
            patch_size,
            stretch,
            enable_augmentation,
            normalize_before_stretch,
            normalize_after_stretch,
        )

        # Predict - always get probabilities if we need to save them
        logger.info("\nRunning SAM2 prediction...")
        need_probabilities = return_probabilities or save_probabilities is not None
        predicted_patches = self._predict_dataset(
            dataset,
            target_size=(patch_size, patch_size),
            return_probabilities=need_probabilities,
            threshold=threshold if not need_probabilities else None,
        )

        # Reconstruct
        print(
            "Reconstructing probability maps..."
            if need_probabilities
            else "Reconstructing flags..."
        )
        # Extract augmentation state from dataset metadata
        num_rotations = getattr(dataset, "metadata", {}).get("augmentation_rotations", 1)

        # Get padded shape for reconstruction loop
        metadata = getattr(dataset, "metadata", {})
        original_shapes = metadata.get("original_shapes")
        if original_shapes is not None and len(original_shapes) > 0:
            orig_channels, orig_times = original_shapes[0]
            # Calculate padded dimensions
            baselines, pols, channels, times = data_shape

            pad_channels = 0
            if orig_channels < patch_size:
                pad_channels = patch_size - orig_channels
            elif orig_channels % patch_size != 0:
                pad_channels = patch_size - (orig_channels % patch_size)

            pad_times = 0
            if orig_times < patch_size:
                pad_times = patch_size - orig_times
            elif orig_times % patch_size != 0:
                pad_times = patch_size - (orig_times % patch_size)

            padded_shape = (baselines, pols, orig_channels + pad_channels, orig_times + pad_times)
            logger.debug(
                f"[predict_array] Using padded shape for reconstruction: {data_shape} → {padded_shape}"
            )
            recon_shape = padded_shape
        else:
            recon_shape = data_shape

        result = self._reconstruct_flags(
            predicted_patches, recon_shape, patch_size, num_rotations, dataset=dataset
        )

        # Save probabilities if requested
        if save_probabilities is not None:
            logger.info(f"\nSaving probability maps to: {save_probabilities}")
            np.save(save_probabilities, result)
            logger.info(f"  Saved shape: {result.shape}, dtype: {result.dtype}")

        # Apply threshold if returning binary flags
        if need_probabilities and not return_probabilities:
            # We got probabilities but user wants binary flags
            thresh = result.mean() if threshold is None else threshold
            logger.info(f"  Applying threshold: {thresh:.4f}")
            result = result > thresh

        if return_probabilities or save_probabilities is not None:
            logger.info(
                f"  Probability range: [{result.min():.3f}, {result.max():.3f}], mean: {result.mean():.3f}"
            )
        if not return_probabilities:
            flag_percent = np.sum(result) / result.size * 100
            logger.info(f"  Flagged: {flag_percent:.2f}% of data")

        logger.info(f"\n{'='*60}")
        logger.info("✓ Prediction complete")
        logger.info(f"{'='*60}")

        return result

    def predict_ms(
        self,
        ms_path: Union[str, Path],
        num_antennas: Optional[int] = None,
        patch_size: int = 128,
        stretch: str = "SQRT",
        apply_existing_flags: bool = False,
        save_flags: bool = True,
        enable_augmentation: bool = False,
        normalize_before_stretch: bool = False,
        normalize_after_stretch: bool = False,
        threshold: Optional[float] = None,
    ) -> NDArray[np.bool_]:
        """
        Single-pass RFI prediction on CASA measurement set.

        This method loads visibility data from a measurement set, performs RFI
        prediction, and optionally saves the flags back to the MS. It handles
        automatic padding/cropping for dimension compatibility with patch_size.

        Parameters
        ----------
        ms_path : str or Path
            Path to CASA measurement set directory.
        num_antennas : int or None, default=None
            Number of antennas to load. If None, loads all antennas.
        patch_size : int, default=128
            Patch size for prediction (128, 256, 512, or 1024).
            Must match the patch_size used during training.
        stretch : str, default='SQRT'
            Stretch function: 'SQRT', 'LOG10', or None.
            Should match the stretch used during training.
        apply_existing_flags : bool, default=False
            If True, load existing flags from MS and mask them before prediction.
            Useful for iterative flagging workflows.
        save_flags : bool, default=True
            If True, save predicted flags back to measurement set.
        enable_augmentation : bool, default=False
            If True, enable 4-way rotation augmentation during inference.
            Generally False for inference (augmentation is for training).
        normalize_before_stretch : bool, default=False
            If True, normalize before applying stretch function.
        normalize_after_stretch : bool, default=False
            If True, normalize after applying stretch function.
        threshold : float or None, default=None
            Probability threshold for RFI detection. If None, uses adaptive
            threshold (mean of probability distribution).

        Returns
        -------
        ndarray of bool
            Predicted RFI flags with shape (baselines, pols, channels, times)
            matching the loaded data dimensions.

        Examples
        --------
        >>> predictor = RFIPredictor(model_path='./models/sam2_rfi.pth')
        >>> flags = predictor.predict_ms(
        ...     'observation.ms',
        ...     patch_size=128,
        ...     stretch='SQRT'
        ... )
        >>> print(f"Flagged {np.sum(flags)/flags.size*100:.2f}% of data")

        Load subset of antennas:

        >>> flags = predictor.predict_ms(
        ...     'observation.ms',
        ...     num_antennas=10,
        ...     patch_size=256
        ... )

        Apply existing flags before prediction:

        >>> flags = predictor.predict_ms(
        ...     'observation.ms',
        ...     apply_existing_flags=True,
        ...     save_flags=True
        ... )

        Notes
        -----
        The method automatically handles padding/cropping if the data dimensions
        are not evenly divisible by patch_size. Padding is removed before saving
        flags back to the MS.

        Preprocessing parameters are validated against checkpoint metadata to
        ensure consistency between training and inference.
        """
        from samrfi.data.ms_loader import MSLoader

        logger.info(f"\n{'='*60}")
        logger.info("RFI Prediction - Single Pass")
        logger.info(f"{'='*60}")

        # Validate preprocessing parameters against checkpoint
        self._validate_preprocessing_params(
            patch_size, stretch, normalize_before_stretch, normalize_after_stretch
        )

        # Load MS
        logger.info("\n[1/4] Loading measurement set...")
        loader = MSLoader(ms_path)
        loader.load(num_antennas=num_antennas, mode="DATA")

        data_shape = loader.data.shape
        logger.info(f"  Data shape: {data_shape}")

        # Check MS compatibility and setup adaptive patching if needed
        baselines, pols, channels, times = data_shape
        patcher = AdaptivePatcher(data_shape, patch_size=patch_size)

        # Use complex data (Preprocessor will extract 3-channel features)
        data = loader.data

        # Optionally apply existing flags before padding
        if apply_existing_flags:
            print("\n[2/4] Loading and applying existing flags...")
            existing_flags = loader.load_flags()
            data = np.where(existing_flags, np.nan, data)
            logger.info(f"  Masked {np.sum(existing_flags)/existing_flags.size*100:.2f}% of data")

        # Pad data if needed
        if patcher.pad_channels > 0 or patcher.pad_times > 0:
            print("  Applying adaptive padding...")
            data = patcher.pad_data(data)
        else:
            if not apply_existing_flags:
                print("\n[2/4] No padding needed - data dimensions compatible")

        # Preprocess
        logger.info("\n[3/4] Preprocessing data...")
        dataset = self._preprocess_data(
            data,
            patch_size,
            stretch,
            enable_augmentation,
            normalize_before_stretch,
            normalize_after_stretch,
        )

        # Predict
        logger.info("\n[4/4] Running SAM2 prediction...")
        predicted_patches = self._predict_dataset(
            dataset, target_size=(patch_size, patch_size), threshold=threshold
        )

        # Reconstruct full flags from patches
        logger.info("\nReconstructing full flag array...")
        # Extract augmentation state from dataset metadata
        num_rotations = getattr(dataset, "metadata", {}).get("augmentation_rotations", 1)
        # Use padded shape for reconstruction if padding was applied
        recon_shape = patcher.get_patch_info()["padded_shape"]
        predicted_flags = self._reconstruct_flags(
            predicted_patches, recon_shape, patch_size, num_rotations, dataset=dataset
        )

        # Crop flags to original dimensions if padding was used
        if patcher.pad_channels > 0 or patcher.pad_times > 0:
            logger.info("  Cropping flags to original dimensions...")
            predicted_flags = patcher.crop_flags(predicted_flags)

        flag_percent = np.sum(predicted_flags) / predicted_flags.size * 100
        logger.info(f"  Flagged: {flag_percent:.2f}% of data")

        # Save flags
        if save_flags:
            print("\nSaving flags to MS...")
            loader.save_flags(predicted_flags)
            print("  ✓ Flags saved")

        logger.info(f"\n{'='*60}")
        logger.info("✓ Prediction complete")
        logger.info(f"{'='*60}")

        return predicted_flags

    def predict_iterative(
        self,
        ms_path: Union[str, Path],
        num_iterations: int = 3,
        num_antennas: Optional[int] = None,
        patch_size: int = 128,
        stretch: str = "SQRT",
        save_flags: bool = True,
        apply_existing_flags: bool = False,
        enable_augmentation: bool = False,
        normalize_before_stretch: bool = False,
        normalize_after_stretch: bool = False,
        threshold: Optional[float] = None,
    ) -> NDArray[np.bool_]:
        """
        Iterative RFI prediction with progressive cleaning.

        This method performs multiple flagging passes where each iteration masks
        already-flagged data and finds remaining RFI. This is particularly effective
        for detecting faint RFI that was hidden by brighter RFI in earlier passes.

        Each iteration:
        1. Masks already-flagged data with NaN
        2. Runs model to find remaining RFI
        3. Combines new flags with cumulative flags from previous iterations

        Parameters
        ----------
        ms_path : str or Path
            Path to CASA measurement set directory.
        num_iterations : int, default=3
            Number of flagging passes to perform.
        num_antennas : int or None, default=None
            Number of antennas to load. If None, loads all antennas.
        patch_size : int, default=128
            Patch size for prediction (128, 256, 512, or 1024).
            Must match the patch_size used during training.
        stretch : str, default='SQRT'
            Stretch function: 'SQRT', 'LOG10', or None.
            Should match the stretch used during training.
        save_flags : bool, default=True
            If True, save final cumulative flags back to measurement set.
        apply_existing_flags : bool, default=False
            If True, load existing flags from MS and include them in cumulative flags.
        enable_augmentation : bool, default=False
            If True, enable 4-way rotation augmentation during inference.
            Generally False for inference (augmentation is for training).
        normalize_before_stretch : bool, default=False
            If True, normalize before applying stretch function.
        normalize_after_stretch : bool, default=False
            If True, normalize after applying stretch function.
        threshold : float or None, default=None
            Probability threshold for RFI detection. If None, uses adaptive
            threshold (mean of probability distribution).

        Returns
        -------
        ndarray of bool
            Cumulative RFI flags from all iterations with shape
            (baselines, pols, channels, times).

        Examples
        --------
        >>> predictor = RFIPredictor(model_path='./models/sam2_rfi.pth')
        >>> flags = predictor.predict_iterative(
        ...     'observation.ms',
        ...     num_iterations=3,
        ...     patch_size=128
        ... )
        >>> print(f"Total flagged: {np.sum(flags)/flags.size*100:.2f}%")

        Start from existing flags:

        >>> flags = predictor.predict_iterative(
        ...     'observation.ms',
        ...     num_iterations=2,
        ...     apply_existing_flags=True
        ... )

        More iterations for deeper cleaning:

        >>> flags = predictor.predict_iterative(
        ...     'observation.ms',
        ...     num_iterations=5,
        ...     patch_size=256
        ... )

        Notes
        -----
        Each iteration finds progressively fainter RFI that was previously masked
        by brighter interference. The effectiveness typically diminishes after
        3-5 iterations as most detectable RFI has been flagged.

        The MS is loaded once at the beginning, and iterations operate on the
        in-memory data to avoid repeated I/O overhead.

        Preprocessing parameters are validated against checkpoint metadata to
        ensure consistency between training and inference.

        See Also
        --------
        predict_ms : Single-pass prediction without iteration
        """
        from samrfi.data.ms_loader import MSLoader

        logger.info(f"\n{'='*60}")
        logger.info(f"RFI Prediction - Iterative ({num_iterations} passes)")
        logger.info(f"{'='*60}")

        # Validate preprocessing parameters against checkpoint
        self._validate_preprocessing_params(
            patch_size, stretch, normalize_before_stretch, normalize_after_stretch
        )

        # Load MS once
        logger.info("\n[Setup] Loading measurement set...")
        loader = MSLoader(ms_path)
        loader.load(num_antennas=num_antennas, mode="DATA")

        data_shape = loader.data.shape
        logger.info(f"  Data shape: {data_shape}")

        # Initialize cumulative flags
        if apply_existing_flags:
            print("\n[Setup] Loading existing MS flags...")
            cumulative_flags = loader.load_flags()
            logger.info(
                f"  Existing flags: {np.sum(cumulative_flags)/cumulative_flags.size*100:.2f}%"
            )
        else:
            cumulative_flags = np.zeros(data_shape, dtype=bool)

        original_data = loader.magnitude.copy()

        # Iterative flagging
        for iteration in range(num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration+1}/{num_iterations}")
            logger.info(f"{'='*60}")

            # Apply cumulative flags to data
            if iteration > 0:
                print(
                    f"\n[1/4] Masking {np.sum(cumulative_flags)/cumulative_flags.size*100:.2f}% already flagged..."
                )
            else:
                print("\n[1/4] First pass - no masking")

            masked_data = np.where(cumulative_flags, np.nan, original_data)

            # Preprocess
            print("\n[2/4] Preprocessing data...")
            dataset = self._preprocess_data(
                masked_data,
                patch_size,
                stretch,
                enable_augmentation,
                normalize_before_stretch,
                normalize_after_stretch,
            )

            # Predict
            print("\n[3/4] Running SAM2 prediction...")
            predicted_patches = self._predict_dataset(
                dataset, target_size=(patch_size, patch_size), threshold=threshold
            )

            # Reconstruct flags
            print("\n[4/4] Reconstructing flags...")
            # Extract augmentation state from dataset metadata
            num_rotations = getattr(dataset, "metadata", {}).get("augmentation_rotations", 1)
            iteration_flags = self._reconstruct_flags(
                predicted_patches, data_shape, patch_size, num_rotations, dataset=dataset
            )

            # Combine with cumulative flags
            new_flags = iteration_flags & ~cumulative_flags  # Only count new flags
            cumulative_flags = cumulative_flags | iteration_flags

            new_percent = np.sum(new_flags) / new_flags.size * 100
            total_percent = np.sum(cumulative_flags) / cumulative_flags.size * 100

            logger.info(f"\n  New flags this iteration: {new_percent:.2f}%")
            logger.info(f"  Total flagged: {total_percent:.2f}%")

        # Save final flags
        if save_flags:
            logger.info(f"\n{'='*60}")
            print("Saving final flags to MS...")
            loader.save_flags(cumulative_flags)
            print("  ✓ Flags saved")

        logger.info(f"\n{'='*60}")
        logger.info("✓ Iterative prediction complete")
        logger.info(f"  Final: {np.sum(cumulative_flags)/cumulative_flags.size*100:.2f}% flagged")
        logger.info(f"{'='*60}")

        return cumulative_flags

    def _predict_dataset(
        self,
        dataset: Any,
        target_size: Optional[Tuple[int, int]] = None,
        return_probabilities: bool = False,
        threshold: Optional[float] = None,
    ) -> List[NDArray[Union[np.bool_, np.float32]]]:
        """
        Run model prediction on preprocessed dataset.

        This method wraps the dataset for SAM2, runs batched inference, and
        optionally resizes outputs to match target patch size.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace Dataset with preprocessed patches.
        target_size : tuple of int or None, default=None
            Target size for output masks (height, width). If None, uses model
            output size (256x256). Should typically match patch_size.
        return_probabilities : bool, default=False
            If True, return continuous probabilities [0,1] instead of binary masks.
        threshold : float or None, default=None
            Probability threshold for binary classification. If None, uses adaptive
            threshold (mean of sigmoid probabilities per batch).

        Returns
        -------
        list of ndarray
            List of predicted masks (bool if return_probabilities=False,
            float32 otherwise), one per patch in dataset.

        Notes
        -----
        The model outputs logits which are converted to probabilities using sigmoid.
        For binary masks, an adaptive threshold (mean of probabilities) is used
        unless a specific threshold is provided.

        GPU memory is managed by running predictions in batches according to
        self.batch_size.
        """
        # Create SAM dataset wrapper (no bbox perturbation for inference)
        sam_dataset = SAMDataset(dataset, self.processor, bbox_perturbation=0)
        dataloader = DataLoader(sam_dataset, batch_size=self.batch_size, shuffle=False)

        predicted_masks = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting patches"):
                # Move to device and ensure contiguity
                pixel_values = batch["pixel_values"].to(self.device).contiguous()
                input_boxes = batch["input_boxes"].to(self.device).contiguous()

                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False
                )

                # Get masks - SAM2 outputs (B, 1, 1, H, W) with multimask_output=False
                pred_masks = outputs.pred_masks  # (B, 1, 1, 256, 256)

                # Squeeze to (B, 1, H, W) for interpolation
                pred_masks = pred_masks.squeeze(2)  # Remove singleton dim -> (B, 1, H, W)

                # Resize on GPU if target size specified and different from output
                if target_size is not None and pred_masks.shape[2:] != target_size:
                    pred_masks = torch.nn.functional.interpolate(
                        pred_masks,  # Already (B, 1, H, W)
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )  # (B, 1, H, W)

                # Remove channel dimension
                pred_masks = pred_masks.squeeze(1)  # (B, H, W)
                sigmoid_probs = torch.sigmoid(pred_masks)

                # Debug: print probability distribution
                logger.info(
                    f"  Sigmoid probs - min: {sigmoid_probs.min():.4f}, max: {sigmoid_probs.max():.4f}, mean: {sigmoid_probs.mean():.4f}"
                )

                # Return probabilities or thresholded masks
                if return_probabilities:
                    predicted_masks.extend(sigmoid_probs.cpu().numpy())
                else:
                    # Use mean as threshold if threshold is None
                    thresh = sigmoid_probs.mean().item() if threshold is None else threshold
                    if threshold is None:
                        logger.info(f"  Using mean threshold: {thresh:.4f}")
                    pred_masks = (sigmoid_probs > thresh).cpu().numpy()
                    predicted_masks.extend(pred_masks)

        return predicted_masks

    def _reconstruct_flags(
        self,
        predicted_patches: List[NDArray[Union[np.bool_, np.float32]]],
        data_shape: Tuple[int, int, int, int],
        patch_size: int,
        num_rotations: int = 1,
        dataset: Optional[Any] = None,
    ) -> NDArray[Union[np.bool_, np.float32]]:
        """
        Reconstruct full flag array from predicted patches.

        This method reverses the patchification process, reassembling individual
        patch predictions into the full data array. It handles rotation augmentation
        by reversing the transformations and combining predictions.

        Parameters
        ----------
        predicted_patches : list of ndarray
            List of predicted patch masks (bool or float32).
        data_shape : tuple of int
            Original data shape (baselines, pols, channels, times).
        patch_size : int
            Size of patches used during prediction.
        num_rotations : int, default=1
            Number of rotations used during augmentation (1, 2, or 4).
            Must match the augmentation_rotations from preprocessing.
        dataset : object or None, default=None
            Optional dataset object with metadata containing original_shapes
            for cropping padded dimensions.

        Returns
        -------
        ndarray of bool or float32
            Reconstructed flags matching data_shape. For probabilities (float),
            uses maximum across rotations. For boolean, uses bitwise OR.

        Notes
        -----
        Rotation reversal transformations:
        - rotation=0: Identity (original)
        - rotation=1: Vertical flip (reverse of vertical flip)
        - rotation=2: Transpose (reverse of transpose)
        - rotation=3: Transpose + vertical flip (reverse both)

        For probability maps, the maximum probability across rotations is used
        at each pixel. For binary masks, any rotation flagging a pixel results
        in that pixel being flagged (OR operation).

        If dataset metadata contains original_shapes, the output is cropped to
        remove padding that was added during preprocessing.
        """
        baselines, pols, channels, times = data_shape

        logger.debug(
            f"[Reconstruction] Input: {len(predicted_patches)} patches, data_shape={data_shape}"
        )
        logger.debug(f"[Reconstruction] num_rotations={num_rotations}, patch_size={patch_size}")

        # Upscale masks from 256x256 to patch_size if needed
        if len(predicted_patches) > 0 and predicted_patches[0].shape[0] != patch_size:
            from scipy.ndimage import zoom

            scale = patch_size / predicted_patches[0].shape[0]
            logger.debug(
                f"[Reconstruction] Upscaling masks: {predicted_patches[0].shape[0]}x{predicted_patches[0].shape[0]} → {patch_size}x{patch_size}"
            )
            predicted_patches = [zoom(p, scale, order=0) for p in predicted_patches]

        # Initialize full flag array (dtype matches input patches)
        is_probability = predicted_patches[0].dtype in (np.float32, np.float64)
        full_flags = np.zeros(data_shape, dtype=np.float32 if is_probability else bool)

        # Track which patches correspond to which baseline/pol
        patch_idx = 0

        # DEBUG: Print reconstruction order for baseline 0
        print("\n  [DEBUG] Reconstruction loop order (baseline 0 only):")
        print(
            f"    num_rotations={num_rotations}, num_patches_h={channels // patch_size}, num_patches_w={times // patch_size}"
        )

        for baseline in range(baselines):
            for pol in range(pols):
                # For each polarization, we had N rotations
                for rotation in range(num_rotations):
                    # Get patches for this rotation
                    num_patches_h = channels // patch_size
                    num_patches_w = times // patch_size

                    for i in range(num_patches_h):
                        for j in range(num_patches_w):
                            if patch_idx >= len(predicted_patches):
                                # No more patches (removed blank patches)
                                patch_idx += 1
                                continue

                            patch_mask = predicted_patches[patch_idx]

                            # DEBUG: Print for baseline 0
                            if baseline == 0 and patch_idx < 8:
                                ch_range = f"[{i*patch_size}:{(i+1)*patch_size}]"
                                t_range = f"[{j*patch_size}:{(j+1)*patch_size}]"
                                print(
                                    f"    patch_idx={patch_idx} → baseline={baseline}, pol={pol}, rot={rotation}, i={i}, j={j}, ch={ch_range}, t={t_range}"
                                )

                            patch_idx += 1

                            # Reverse rotation
                            if rotation == 0:
                                # Original
                                reconstructed = patch_mask
                            elif rotation == 1:
                                # Was flipped vertical
                                reconstructed = np.flip(patch_mask, axis=0)
                            elif rotation == 2:
                                # Was transposed
                                reconstructed = patch_mask.T
                            elif rotation == 3:
                                # Was transposed + flipped
                                reconstructed = np.flip(patch_mask.T, axis=0)

                            # Place in full array
                            ch_start = i * patch_size
                            ch_end = (i + 1) * patch_size
                            t_start = j * patch_size
                            t_end = (j + 1) * patch_size

                            if is_probability:
                                # For probabilities, take max across rotations
                                full_flags[baseline, pol, ch_start:ch_end, t_start:t_end] = (
                                    np.maximum(
                                        full_flags[baseline, pol, ch_start:ch_end, t_start:t_end],
                                        reconstructed,
                                    )
                                )
                            else:
                                # For boolean, use bitwise OR
                                full_flags[
                                    baseline, pol, ch_start:ch_end, t_start:t_end
                                ] |= reconstructed

        # Crop to original shape if metadata available
        if dataset is not None:
            metadata = getattr(dataset, "metadata", {})
            original_shapes = metadata.get("original_shapes")
            if original_shapes is not None and len(original_shapes) > 0:
                # Assume all baselines/pols had same original shape (first one)
                orig_channels, orig_times = original_shapes[0]
                if orig_channels != channels or orig_times != times:
                    logger.debug(
                        f"[Reconstruction] Cropping: ({channels}, {times}) → ({orig_channels}, {orig_times})"
                    )
                    full_flags = full_flags[:, :, :orig_channels, :orig_times]

        logger.debug(f"[Reconstruction] Final shape: {full_flags.shape}")
        return full_flags

    def _download_from_hf(self, repo_id: str, model_size: str) -> str:
        """
        Download trained model from HuggingFace Hub to local cache.

        This method downloads model checkpoints from HuggingFace Hub, storing
        them in the local HF cache directory. Subsequent calls reuse the cached
        file without re-downloading.

        Parameters
        ----------
        repo_id : str
            HuggingFace repository ID (e.g., 'preshanth/sam-rfi-models').
        model_size : str
            Model size subdirectory: 'tiny', 'small', 'base_plus', or 'large'.

        Returns
        -------
        str
            Local path to downloaded model checkpoint file.

        Raises
        ------
        Exception
            If download fails due to network issues, invalid repo, or missing file.

        Notes
        -----
        The downloaded model is cached in the HuggingFace cache directory,
        which respects the HF_HOME environment variable. For private repositories,
        set the HF_TOKEN environment variable with your access token.

        Examples
        --------
        >>> predictor = RFIPredictor(model_path='preshanth/sam-rfi-models/large')
        >>> # Downloads and caches model automatically on first use
        """
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {model_size} model from {repo_id}...")

        try:
            # Download to default HF cache (respects HF_HOME env var)
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{model_size}/model.pth",
                repo_type="model",
            )

            logger.info(f"✓ Model downloaded to: {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download model from {repo_id}: {e}")
            logger.info(
                "Check: (1) Internet connection, (2) Repo exists, (3) Token for private repos"
            )
            logger.info("For private repos, set HF_TOKEN environment variable")
            raise
