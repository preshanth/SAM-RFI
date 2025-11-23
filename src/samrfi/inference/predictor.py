"""
RFI Predictor - Apply trained SAM3 models to new data

Supports single-pass and iterative flagging with progressive cleaning.
"""

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from transformers import Sam3Processor, Sam3Model
from torch.utils.data import DataLoader

from samrfi.data import MSLoader, Preprocessor, SAMDataset, AdaptivePatcher, check_ms_compatibility


class RFIPredictor:
    """
    Apply trained SAM3 model to predict RFI flags.

    Supports iterative flagging where each pass finds fainter RFI
    that was hidden by brighter RFI in previous passes.

    Usage:
        >>> predictor = RFIPredictor(model_path='./models/sam3_rfi.pth')
        >>> flags = predictor.predict_ms('observation.ms')
        >>> # Or iterative:
        >>> flags = predictor.predict_iterative('observation.ms', num_iterations=3)
    """

    def __init__(self, model_path, sam_checkpoint="unified", device="cuda", batch_size=4):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint (.pth)
            sam_checkpoint: SAM3 checkpoint (unused, kept for compatibility)
            device: Compute device ('cuda' or 'cpu')
            batch_size: Batch size for inference
        """
        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = batch_size

        # SAM3 uses single unified model (840M params)
        model_name = "facebook/sam3"

        print(f"Loading SAM3 model: {model_name}")
        print(f"  Note: SAM3 has single 840M param model (no variants)")

        # Load processor and model
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.model = Sam3Model.from_pretrained(model_name)

        # Load trained weights
        print(f"Loading trained weights from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(checkpoint)

        # Move to device
        self.model.to(device)
        self.model.eval()

        print(f"✓ Model loaded on {device}")

    def predict_ms(
        self,
        ms_path,
        num_antennas=None,
        patch_size=128,
        stretch="SQRT",
        apply_existing_flags=False,
        save_flags=True,
    ):
        """
        Single-pass prediction on measurement set.

        Args:
            ms_path: Path to measurement set
            num_antennas: Number of antennas to load (None = all)
            patch_size: Patch size for prediction
            stretch: Stretch function ('SQRT' or 'LOG10')
            apply_existing_flags: If True, mask existing flags before prediction
            save_flags: If True, save flags back to MS

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
            print(f"  Applying adaptive padding...")
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
            apply_stretching=True,
        )

        # Predict
        print("\n[4/4] Running SAM2 prediction...")
        predicted_patches = self._predict_dataset(dataset)

        # Reconstruct full flags from patches
        print("\nReconstructing full flag array...")
        # Use padded shape for reconstruction if padding was applied
        recon_shape = patcher.get_patch_info()['padded_shape']
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
                apply_stretching=True,
            )

            # Predict
            print("\n[3/4] Running SAM2 prediction...")
            predicted_patches = self._predict_dataset(dataset)

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
        print(f"✓ Iterative prediction complete")
        print(f"  Final: {np.sum(cumulative_flags)/cumulative_flags.size*100:.2f}% flagged")
        print(f"{'='*60}")

        return cumulative_flags

    def _predict_dataset(self, dataset):
        """
        Run model prediction on dataset.

        Args:
            dataset: HuggingFace Dataset with patches

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

                # Get masks
                pred_masks = outputs.pred_masks.squeeze(1)  # (B, H, W)

                # Threshold and convert to boolean
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
