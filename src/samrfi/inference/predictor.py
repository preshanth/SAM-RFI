"""
RFI Predictor - Apply trained SAM2 models to new data

Supports single-pass and iterative flagging with progressive cleaning.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Sam2Model, Sam2Processor

from samrfi.data import AdaptivePatcher, MSLoader, Preprocessor, SAMDataset
from samrfi.utils import logger
from samrfi.utils.errors import CheckpointMismatchError


# Monkey-patch transformers Sam2Model to fix view/reshape bug
# The bug: feat.permute(1, 2, 0).view(...) fails because permute makes tensor non-contiguous
# Fix: Replace view() with reshape() which handles non-contiguous tensors
def _patch_sam2_view_to_reshape():
    """
    Patch Sam2Model.forward to use reshape instead of view after permute operations.

    This fixes RuntimeError: view size is not compatible with input tensor's size and stride
    (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    """
    import transformers.models.sam2.modeling_sam2 as sam2_module

    # Save original forward method
    original_forward = sam2_module.Sam2Model.forward

    def patched_forward(self, *args, **kwargs):
        """Wrapped forward that ensures tensors are contiguous before view operations"""
        # Temporarily replace tensor.view with a safe version
        original_view = torch.Tensor.view

        def safe_view(tensor, *shape):
            """Use reshape instead of view to handle non-contiguous tensors"""
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

    Supports iterative flagging where each pass finds fainter RFI
    that was hidden by brighter RFI in previous passes.

    Usage:
        >>> predictor = RFIPredictor(model_path='./models/sam2_rfi.pth')
        >>> flags = predictor.predict_ms('observation.ms')
        >>> # Or iterative:
        >>> flags = predictor.predict_iterative('observation.ms', num_iterations=3)
    """

    def __init__(
        self,
        model_path,
        sam_checkpoint="large",
        device="cuda",
        batch_size=4,
        allow_partial_load: bool = False,
        auto_select_sam: bool = False,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint (.pth)
            sam_checkpoint: SAM2 checkpoint size (tiny, small, base_plus, large)
            device: Compute device ('cuda' or 'cpu')
            batch_size: Batch size for inference
        """
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

        model_state = self.model.state_dict()
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
                logger.warning("Auto-selection failed to find a better match; proceeding to error handling.")

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

    def _validate_preprocessing_params(self, patch_size, stretch, normalize_before_stretch=False, normalize_after_stretch=False):
        """
        Validate inference preprocessing parameters against checkpoint metadata.

        Raises CheckpointMismatchError if critical parameters mismatch (patch_size).
        Warns if non-critical parameters mismatch (stretch, normalization).

        Args:
            patch_size: Patch size for inference
            stretch: Stretch function ('SQRT', 'LOG10', or None)
            normalize_before_stretch: Normalization before stretch
            normalize_after_stretch: Normalization after stretch

        Raises:
            CheckpointMismatchError: If patch_size doesn't match checkpoint
        """
        if not self.checkpoint_preprocessing:
            # No metadata in checkpoint (old checkpoint), skip validation
            return

        # Critical: patch_size must match
        checkpoint_patch_size = self.checkpoint_preprocessing.get("patch_size")
        if checkpoint_patch_size and checkpoint_patch_size != "unknown" and checkpoint_patch_size != patch_size:
            raise CheckpointMismatchError(
                param_name="patch_size",
                checkpoint_value=checkpoint_patch_size,
                inference_value=patch_size
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

        if checkpoint_norm_before is not None and checkpoint_norm_before != normalize_before_stretch:
            logger.info(
                f"Note: normalize_before_stretch differs "
                f"(training={checkpoint_norm_before}, inference={normalize_before_stretch})"
            )

        if checkpoint_norm_after is not None and checkpoint_norm_after != normalize_after_stretch:
            logger.info(
                f"Note: normalize_after_stretch differs "
                f"(training={checkpoint_norm_after}, inference={normalize_after_stretch})"
            )

    def predict_array(
        self,
        data,
        patch_size=1024,
        stretch=None,
        enable_augmentation=False,
        normalize_before_stretch=False,
        normalize_after_stretch=False,
        return_probabilities=False,
        threshold=0.5,
        save_probabilities=None,
    ):
        """
        Predict on numpy array directly without MS I/O.

        Args:
            data: Complex visibility data (baselines, pols, channels, times)
            patch_size: Patch size for prediction
            stretch: Stretch function ('SQRT' or 'LOG10' or None)
            enable_augmentation: Enable rotation augmentation (default False)
            normalize_before_stretch: Normalize before stretch (default False)
            normalize_after_stretch: Normalize after stretch (default False)
            return_probabilities: Return continuous probabilities [0,1] instead of binary flags (default False)
            threshold: Probability threshold for RFI detection (default: 0.5, None=use mean)
            save_probabilities: Path to save probability maps (.npy file, optional)

        Returns:
            Predicted probabilities (if return_probabilities=True) or flags array (baselines, pols, channels, times)
        """
        logger.info(f"\n{'='*60}")
        logger.info("RFI Prediction - Array Mode")
        logger.info(f"{'='*60}")

        # Validate preprocessing parameters against checkpoint
        self._validate_preprocessing_params(patch_size, stretch, normalize_before_stretch, normalize_after_stretch)

        data_shape = data.shape
        logger.info(f"  Input shape: {data_shape}")
        logger.info(f"  Data dtype: {data.dtype}, complex: {np.iscomplexobj(data)}")

        # Preprocess (pass complex data directly - Preprocessor will extract features)
        logger.info("\nPreprocessing data...")
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
            inference_mode=True,
        )

        # Predict - always get probabilities if we need to save them
        logger.info("\nRunning SAM2 prediction...")
        need_probabilities = return_probabilities or save_probabilities is not None
        predicted_patches = self._predict_dataset(
            dataset,
            target_size=(patch_size, patch_size),
            return_probabilities=need_probabilities,
            threshold=threshold if not need_probabilities else None
        )

        # Reconstruct
        print("Reconstructing probability maps..." if need_probabilities else "Reconstructing flags...")
        # Extract augmentation state from dataset metadata
        num_rotations = getattr(dataset, 'metadata', {}).get('augmentation_rotations', 1)

        # Get padded shape for reconstruction loop
        metadata = getattr(dataset, 'metadata', {})
        original_shapes = metadata.get('original_shapes')
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
            logger.debug(f"[predict_array] Using padded shape for reconstruction: {data_shape} → {padded_shape}")
            recon_shape = padded_shape
        else:
            recon_shape = data_shape

        result = self._reconstruct_flags(predicted_patches, recon_shape, patch_size, num_rotations, dataset=dataset)

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
            logger.info(f"  Probability range: [{result.min():.3f}, {result.max():.3f}], mean: {result.mean():.3f}")
        if not return_probabilities:
            flag_percent = np.sum(result) / result.size * 100
            logger.info(f"  Flagged: {flag_percent:.2f}% of data")

        logger.info(f"\n{'='*60}")
        logger.info("✓ Prediction complete")
        logger.info(f"{'='*60}")

        return result

    def predict_ms(
        self,
        ms_path,
        num_antennas=None,
        patch_size=128,
        stretch="SQRT",
        apply_existing_flags=False,
        save_flags=True,
        enable_augmentation=False,
        normalize_before_stretch=False,
        normalize_after_stretch=False,
        threshold=0.5,
    ):
        """
        Single-pass prediction on measurement set.

        Args:
            ms_path: Path to measurement set
            num_antennas: Number of antennas to load (None = all)
            patch_size: Patch size for prediction
            stretch: Stretch function ('SQRT' or 'LOG10' or None)
            threshold: Probability threshold for RFI detection (default: 0.5)
            apply_existing_flags: If True, mask existing flags before prediction
            save_flags: If True, save flags back to MS
            enable_augmentation: Enable rotation augmentation (default False for inference)
            normalize_before_stretch: Normalize before stretch (default False)
            normalize_after_stretch: Normalize after stretch (default False)

        Returns:
            Predicted flags array (baselines, pols, channels, times)
        """
        logger.info(f"\n{'='*60}")
        logger.info("RFI Prediction - Single Pass")
        logger.info(f"{'='*60}")

        # Validate preprocessing parameters against checkpoint
        self._validate_preprocessing_params(patch_size, stretch, normalize_before_stretch, normalize_after_stretch)

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

        # Predict
        logger.info("\n[4/4] Running SAM2 prediction...")
        predicted_patches = self._predict_dataset(dataset, target_size=(patch_size, patch_size), threshold=threshold)

        # Reconstruct full flags from patches
        logger.info("\nReconstructing full flag array...")
        # DEBUG: Check predicted patches before reconstruction AND save images
        print(f"\n  [DEBUG] Predicted patches stats:")
        print(f"    Total patches: {len(predicted_patches)}")

        # Save first 8 patches (baseline 0) for debugging
        import matplotlib.pyplot as plt
        from pathlib import Path
        debug_dir = Path("patch_debug_real")
        debug_dir.mkdir(exist_ok=True)

        for idx in range(min(8, len(predicted_patches))):
            flagged_pct = np.sum(predicted_patches[idx]) / predicted_patches[idx].size * 100
            print(f"    Patch {idx}: {flagged_pct:5.2f}% flagged")

            # Save patch visualization
            if idx < 8:  # Baseline 0 only
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Get the input image from dataset
                sample = dataset[idx]
                img = sample['image']  # (H, W, 3) numpy array already denormalized

                # Left: Input
                axes[0].imshow(img)
                axes[0].set_title(f"Input Patch {idx}")
                axes[0].axis('off')

                # Right: Mask overlay
                axes[1].imshow(img)
                axes[1].imshow(predicted_patches[idx], alpha=0.5, cmap='Reds')
                axes[1].set_title(f"Predicted Mask\n{flagged_pct:.1f}% flagged")
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig(debug_dir / f"patch_{idx:02d}.png", dpi=150, bbox_inches='tight')
                plt.close()

        print(f"  Saved first 8 patches to {debug_dir}/")

        # Extract augmentation state from dataset metadata
        num_rotations = getattr(dataset, 'metadata', {}).get('augmentation_rotations', 1)
        # Use padded shape for reconstruction if padding was applied
        recon_shape = patcher.get_patch_info()["padded_shape"]
        predicted_flags = self._reconstruct_flags(predicted_patches, recon_shape, patch_size, num_rotations, dataset=dataset)

        # Crop flags to original dimensions if padding was used
        if patcher.pad_channels > 0 or patcher.pad_times > 0:
            print("  Cropping flags to original dimensions...")
            # DEBUG: Analyze padding region before cropping
            print("\n  [DEBUG] Analyzing padding region before crop:")
            orig_channels, orig_times = patcher.channels, patcher.times
            pad_channels, pad_times = patcher.pad_channels, patcher.pad_times

            # Check padding in time dimension (most common case)
            if pad_times > 0:
                # Real data region: times [0:orig_times]
                real_region = predicted_flags[:, :, :, :orig_times]
                real_flagged = np.sum(real_region) / real_region.size * 100

                # Padding region: times [orig_times:]
                pad_region = predicted_flags[:, :, :, orig_times:]
                pad_flagged = np.sum(pad_region) / pad_region.size * 100

                print(f"    Real data region (times [0:{orig_times}]):      {real_flagged:5.2f}% flagged")
                print(f"    Padding region (times [{orig_times}:{orig_times+pad_times}]): {pad_flagged:5.2f}% flagged")

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
        ms_path,
        num_iterations=3,
        num_antennas=None,
        patch_size=128,
        stretch="SQRT",
        save_flags=True,
        apply_existing_flags=False,
        enable_augmentation=False,
        normalize_before_stretch=False,
        normalize_after_stretch=False,
        threshold=0.5,
    ):
        """
        Iterative prediction with progressive cleaning.

        Each iteration:
        1. Masks already-flagged data
        2. Runs model to find remaining RFI
        3. Combines flags with previous iterations

        Args:
            ms_path: Path to measurement set
            num_iterations: Number of flagging passes
            num_antennas: Number of antennas to load (None = all)
            patch_size: Patch size for prediction
            stretch: Stretch function ('SQRT', 'LOG10', or None)
            save_flags: If True, save final flags to MS
            apply_existing_flags: If True, load and preserve existing MS flags
            threshold: Probability threshold for RFI detection (default: 0.5)

        Returns:
            Cumulative flags from all iterations
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RFI Prediction - Iterative ({num_iterations} passes)")
        logger.info(f"{'='*60}")

        # Validate preprocessing parameters against checkpoint
        self._validate_preprocessing_params(patch_size, stretch, normalize_before_stretch, normalize_after_stretch)

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
            logger.info(f"  Existing flags: {np.sum(cumulative_flags)/cumulative_flags.size*100:.2f}%")
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
            preprocessor = Preprocessor(masked_data, flags=None)
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

            # Predict
            print("\n[3/4] Running SAM2 prediction...")
            predicted_patches = self._predict_dataset(dataset, target_size=(patch_size, patch_size), threshold=threshold)

            # Reconstruct flags
            print("\n[4/4] Reconstructing flags...")
            # Extract augmentation state from dataset metadata
            num_rotations = getattr(dataset, 'metadata', {}).get('augmentation_rotations', 1)
            iteration_flags = self._reconstruct_flags(predicted_patches, data_shape, patch_size, num_rotations, dataset=dataset)

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

    def _predict_dataset(self, dataset, target_size=None, return_probabilities=False, threshold=0.5):
        """
        Run model prediction on dataset.

        Args:
            dataset: HuggingFace Dataset with patches
            target_size: Target size for output masks (H, W). If None, uses model output size (256x256)
            return_probabilities: Return continuous probabilities [0,1] instead of binary masks
            threshold: Probability threshold for binary classification (default: 0.5)

        Returns:
            List of predicted masks (boolean arrays if return_probabilities=False, float arrays otherwise)
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
                logger.info(f"  Sigmoid probs - min: {sigmoid_probs.min():.4f}, max: {sigmoid_probs.max():.4f}, mean: {sigmoid_probs.mean():.4f}")

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

    def _reconstruct_flags(self, predicted_patches, data_shape, patch_size, num_rotations=1, dataset=None):
        """
        Reconstruct full flag array from predicted patches.

        This reverses the patchification process (with N-way rotation).

        Args:
            predicted_patches: List of predicted patch masks (bool or float)
            data_shape: Original data shape (baselines, pols, channels, times)
            patch_size: Size of patches
            num_rotations: Number of rotations used during augmentation (default: 1)
            dataset: Optional dataset object with metadata (for original_shapes)

        Returns:
            Reconstructed flags matching data_shape (bool or float matching input)
        """
        baselines, pols, channels, times = data_shape

        logger.debug(f"[Reconstruction] Input: {len(predicted_patches)} patches, data_shape={data_shape}")
        logger.debug(f"[Reconstruction] num_rotations={num_rotations}, patch_size={patch_size}")

        # Upscale masks from 256x256 to patch_size if needed
        if len(predicted_patches) > 0 and predicted_patches[0].shape[0] != patch_size:
            from scipy.ndimage import zoom
            scale = patch_size / predicted_patches[0].shape[0]
            logger.debug(f"[Reconstruction] Upscaling masks: {predicted_patches[0].shape[0]}x{predicted_patches[0].shape[0]} → {patch_size}x{patch_size}")
            predicted_patches = [zoom(p, scale, order=0) for p in predicted_patches]

        # Initialize full flag array (dtype matches input patches)
        is_probability = predicted_patches[0].dtype in (np.float32, np.float64)
        full_flags = np.zeros(data_shape, dtype=np.float32 if is_probability else bool)

        # Track which patches correspond to which baseline/pol
        patch_idx = 0

        # DEBUG: Print reconstruction order for baseline 0
        print(f"\n  [DEBUG] Reconstruction loop order (baseline 0 only):")
        print(f"    num_rotations={num_rotations}, num_patches_h={channels // patch_size}, num_patches_w={times // patch_size}")

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
                                print(f"    patch_idx={patch_idx} → baseline={baseline}, pol={pol}, rot={rotation}, i={i}, j={j}, ch={ch_range}, t={t_range}")

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
                                full_flags[baseline, pol, ch_start:ch_end, t_start:t_end] = np.maximum(
                                    full_flags[baseline, pol, ch_start:ch_end, t_start:t_end],
                                    reconstructed
                                )
                            else:
                                # For boolean, use bitwise OR
                                full_flags[baseline, pol, ch_start:ch_end, t_start:t_end] |= reconstructed

        # Crop to original shape if metadata available
        if dataset is not None:
            metadata = getattr(dataset, 'metadata', {})
            original_shapes = metadata.get('original_shapes')
            if original_shapes is not None and len(original_shapes) > 0:
                # Assume all baselines/pols had same original shape (first one)
                orig_channels, orig_times = original_shapes[0]
                if orig_channels != channels or orig_times != times:
                    logger.debug(f"[Reconstruction] Cropping: ({channels}, {times}) → ({orig_channels}, {orig_times})")
                    full_flags = full_flags[:, :, :orig_channels, :orig_times]

        logger.debug(f"[Reconstruction] Final shape: {full_flags.shape}")
        return full_flags
