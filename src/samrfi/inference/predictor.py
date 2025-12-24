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

        print(f"Loading SAM2 model: {model_name}")

        # Load processor and model
        self.processor = Sam2Processor.from_pretrained(model_name)
        self.model = Sam2Model.from_pretrained(model_name)

        # Load trained weights
        # Note: We load the SAM2 architecture via HuggingFace (Sam2Model.from_pretrained)
        # because it provides the model class and pretrained backbone. The file you pass
        # via `model_path` is expected to be either a plain state_dict (mapping of tensor
        # names -> tensors) or a full training checkpoint dict containing a 'model_state_dict'
        # (and possibly optimizer state, epoch, metadata). We support both formats here.
        print(f"Loading trained weights from: {self.model_path}")
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
                print(f"Failed to instantiate model for variant {candidate_variant}: {e}")
                return (1e9, None)  # very bad score
            mism, missing, unexpected = _compare_to_model(m_c)
            score = len(mism) + len(missing) + len(unexpected)
            return (score, (mism, missing, unexpected))

        # Auto-selection helper exposed as method for potential reuse
        def _auto_select_internal():
            candidates = ["tiny", "small", "base_plus", "large"]
            scores = []
            for c in candidates:
                print(f"Testing SAM variant: {c} ...")
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
            print(
                "Warning: shape mismatches detected; attempting to auto-select the SAM variant that best matches the checkpoint (this may download models and be slow)..."
            )
            best_variant = self._auto_select_sam_variant(state_dict)
            if best_variant:
                print(
                    f"Auto-selected SAM variant: {best_variant}. Re-instantiating model and retrying load."
                )
                model_name = checkpoint_map.get(best_variant, checkpoint_map["large"])
                self.processor = Sam2Processor.from_pretrained(model_name)
                self.model = Sam2Model.from_pretrained(model_name)
                # Recompute mismatches against new model
                mismatched, missing_in_ckpt, unexpected_in_ckpt = _compare_to_model(self.model)
            else:
                print("Auto-selection failed to find a better match; proceeding to error handling.")

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
        # Move to device
        self.model.to(device)
        self.model.eval()

        print(f"✓ Model loaded on {device}")

    def predict_array(
        self,
        data,
        patch_size=1024,
        stretch=None,
        enable_augmentation=False,
        normalize_before_stretch=False,
        normalize_after_stretch=False,
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

        Returns:
            Predicted flags array (baselines, pols, channels, times)
        """
        print(f"\n{'='*60}")
        print("RFI Prediction - Array Mode")
        print(f"{'='*60}")

        data_shape = data.shape
        print(f"  Input shape: {data_shape}")

        # Get magnitude
        if np.iscomplexobj(data):
            magnitude_data = np.abs(data)
        else:
            magnitude_data = data

        # Preprocess
        print("\nPreprocessing data...")
        preprocessor = Preprocessor(magnitude_data, flags=None)
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

        # Predict
        print("\nRunning SAM2 prediction...")
        predicted_patches = self._predict_dataset(dataset, target_size=(patch_size, patch_size))

        # Reconstruct
        print("Reconstructing flags...")
        predicted_flags = self._reconstruct_flags(predicted_patches, data_shape, patch_size)

        flag_percent = np.sum(predicted_flags) / predicted_flags.size * 100
        print(f"  Flagged: {flag_percent:.2f}% of data")

        print(f"\n{'='*60}")
        print("✓ Prediction complete")
        print(f"{'='*60}")

        return predicted_flags

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
    ):
        """
        Single-pass prediction on measurement set.

        Args:
            ms_path: Path to measurement set
            num_antennas: Number of antennas to load (None = all)
            patch_size: Patch size for prediction
            stretch: Stretch function ('SQRT' or 'LOG10' or None)
            apply_existing_flags: If True, mask existing flags before prediction
            save_flags: If True, save flags back to MS
            enable_augmentation: Enable rotation augmentation (default False for inference)
            normalize_before_stretch: Normalize before stretch (default False)
            normalize_after_stretch: Normalize after stretch (default False)

        Returns:
            Predicted flags array (baselines, pols, channels, times)
        """
        print(f"\n{'='*60}")
        print("RFI Prediction - Single Pass")
        print(f"{'='*60}")

        # Load MS
        print("\n[1/4] Loading measurement set...")
        loader = MSLoader(ms_path)
        loader.load(num_antennas=num_antennas, mode="DATA")

        data_shape = loader.data.shape
        print(f"  Data shape: {data_shape}")

        # Check MS compatibility and setup adaptive patching if needed
        baselines, pols, channels, times = data_shape
        patcher = AdaptivePatcher(data_shape, patch_size=patch_size)

        # Get magnitude data
        data = loader.magnitude

        # Optionally apply existing flags before padding
        if apply_existing_flags:
            print("\n[2/4] Loading and applying existing flags...")
            existing_flags = loader.load_flags()
            data = np.where(existing_flags, np.nan, data)
            print(f"  Masked {np.sum(existing_flags)/existing_flags.size*100:.2f}% of data")

        # Pad data if needed
        if patcher.pad_channels > 0 or patcher.pad_times > 0:
            print("  Applying adaptive padding...")
            data = patcher.pad_data(data)
        else:
            if not apply_existing_flags:
                print("\n[2/4] No padding needed - data dimensions compatible")

        # Preprocess
        print("\n[3/4] Preprocessing data...")
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
        )

        # Predict
        print("\n[4/4] Running SAM2 prediction...")
        predicted_patches = self._predict_dataset(dataset, target_size=(patch_size, patch_size))

        # Reconstruct full flags from patches
        print("\nReconstructing full flag array...")
        # Use padded shape for reconstruction if padding was applied
        recon_shape = patcher.get_patch_info()["padded_shape"]
        predicted_flags = self._reconstruct_flags(predicted_patches, recon_shape, patch_size)

        # Crop flags back to original dimensions if padding was used
        if patcher.pad_channels > 0 or patcher.pad_times > 0:
            print("  Cropping flags to original dimensions...")
            predicted_flags = patcher.crop_flags(predicted_flags)

        flag_percent = np.sum(predicted_flags) / predicted_flags.size * 100
        print(f"  Flagged: {flag_percent:.2f}% of data")

        # Save flags
        if save_flags:
            print("\nSaving flags to MS...")
            loader.save_flags(predicted_flags)
            print("  ✓ Flags saved")

        print(f"\n{'='*60}")
        print("✓ Prediction complete")
        print(f"{'='*60}")

        return predicted_flags

    def predict_iterative(
        self,
        ms_path,
        num_iterations=3,
        num_antennas=None,
        patch_size=128,
        stretch="SQRT",
        save_flags=True,
        enable_augmentation=False,
        normalize_before_stretch=False,
        normalize_after_stretch=False,
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
            stretch: Stretch function ('SQRT' or 'LOG10')
            save_flags: If True, save final flags to MS

        Returns:
            Cumulative flags from all iterations
        """
        print(f"\n{'='*60}")
        print(f"RFI Prediction - Iterative ({num_iterations} passes)")
        print(f"{'='*60}")

        # Load MS once
        print("\n[Setup] Loading measurement set...")
        loader = MSLoader(ms_path)
        loader.load(num_antennas=num_antennas, mode="DATA")

        data_shape = loader.data.shape
        print(f"  Data shape: {data_shape}")

        # Initialize cumulative flags
        cumulative_flags = np.zeros(data_shape, dtype=bool)
        original_data = loader.magnitude.copy()

        # Iterative flagging
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration+1}/{num_iterations}")
            print(f"{'='*60}")

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
            )

            # Predict
            print("\n[3/4] Running SAM2 prediction...")
            predicted_patches = self._predict_dataset(dataset, target_size=(patch_size, patch_size))

            # Reconstruct flags
            print("\n[4/4] Reconstructing flags...")
            iteration_flags = self._reconstruct_flags(predicted_patches, data_shape, patch_size)

            # Combine with cumulative flags
            new_flags = iteration_flags & ~cumulative_flags  # Only count new flags
            cumulative_flags = cumulative_flags | iteration_flags

            new_percent = np.sum(new_flags) / new_flags.size * 100
            total_percent = np.sum(cumulative_flags) / cumulative_flags.size * 100

            print(f"\n  New flags this iteration: {new_percent:.2f}%")
            print(f"  Total flagged: {total_percent:.2f}%")

        # Save final flags
        if save_flags:
            print(f"\n{'='*60}")
            print("Saving final flags to MS...")
            loader.save_flags(cumulative_flags)
            print("  ✓ Flags saved")

        print(f"\n{'='*60}")
        print("✓ Iterative prediction complete")
        print(f"  Final: {np.sum(cumulative_flags)/cumulative_flags.size*100:.2f}% flagged")
        print(f"{'='*60}")

        return cumulative_flags

    def _predict_dataset(self, dataset, target_size=None):
        """
        Run model prediction on dataset.

        Args:
            dataset: HuggingFace Dataset with patches
            target_size: Target size for output masks (H, W). If None, uses model output size (256x256)

        Returns:
            List of predicted masks (boolean arrays)
        """
        # Create SAM dataset wrapper
        sam_dataset = SAMDataset(dataset, self.processor)
        dataloader = DataLoader(sam_dataset, batch_size=self.batch_size, shuffle=False)

        predicted_masks = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting patches"):
                # Move to device
                pixel_values = batch["pixel_values"].to(self.device)
                input_boxes = batch["input_boxes"].to(self.device)

                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False
                )

                # Get masks - outputs.pred_masks is (B, 1, H, W)
                pred_masks = outputs.pred_masks  # Keep as (B, 1, H, W) for interpolation

                # Resize on GPU if target size specified and different from output
                if target_size is not None and pred_masks.shape[2:] != target_size:
                    pred_masks = torch.nn.functional.interpolate(
                        pred_masks,  # Already (B, 1, H, W)
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )  # (B, 1, H, W)

                # Remove channel dimension and threshold
                pred_masks = pred_masks.squeeze(1)  # (B, H, W)
                pred_masks = (torch.sigmoid(pred_masks) > 0.5).cpu().numpy()

                predicted_masks.extend(pred_masks)

        return predicted_masks

    def _reconstruct_flags(self, predicted_patches, data_shape, patch_size):
        """
        Reconstruct full flag array from predicted patches.

        This reverses the patchification process (with 4-way rotation).

        Args:
            predicted_patches: List of predicted patch masks
            data_shape: Original data shape (baselines, pols, channels, times)
            patch_size: Size of patches

        Returns:
            Reconstructed flags matching data_shape
        """
        baselines, pols, channels, times = data_shape

        # Initialize full flag array
        full_flags = np.zeros(data_shape, dtype=bool)

        # Track which patches correspond to which baseline/pol
        patch_idx = 0

        for baseline in range(baselines):
            for pol in range(pols):
                # For each polarization, we had 4 rotations
                for rotation in range(4):
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

                            full_flags[
                                baseline, pol, ch_start:ch_end, t_start:t_end
                            ] |= reconstructed

        return full_flags
