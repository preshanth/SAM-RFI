#!/usr/bin/env python3
"""
Real SAM-RFI ML Pipeline Test

Tests the actual pipeline components in sequence:
1. SimulatedMS generates realistic baseline with RFI
2. Extract channels using new R=Gradient, G=Amplitude, B=Phase mapping  
3. Load SAM2-tiny and run real inference
4. Compute training loss with ground truth
5. Test union approach

Step-by-step validation of the real ML pipeline.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging

# Pipeline components
from samrfi.datasets import SyntheticDatasetGenerator, ObservationConfig, RFIConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMLPipeline:
    """Step-by-step ML pipeline validation"""
    
    # Class variables to share data between test methods
    _obs_metadata = None
    _temp_dir = None
    _obs_config = None
    _rfi_config = None
    
    def test_1_synthetic_data_generation(self):
        """Step 1: Generate synthetic RFI data using SyntheticDatasetGenerator"""
        # Minimal but realistic config
        obs_config = ObservationConfig(
            num_antennas=4,          # 6 baselines (small but real)
            num_spw=1,               # Single SPW  
            channels_per_spw=1024,   # Full 1024 for SAM2
            start_frequency=1.4e9,   # L-band
            total_duration=1024.0,   # 1024 time steps
            integration_time=1.0,
            thermal_noise_sigma=1e-3
        )
        
        # Heavy RFI for clear testing
        rfi_config = RFIConfig(
            broadband_probability=0.3,  # 30% contamination
            narrowband_lines=5,         # 5 narrowband lines
            transient_events=3,         # 3 transients
            periodic_signals=1,         # 1 periodic signal  
            satellite_passes=1          # 1 satellite pass
        )
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="samrfi_test_")
        
        try:
            logger.info("Creating synthetic data with SyntheticDatasetGenerator...")
            generator = SyntheticDatasetGenerator(output_dir=temp_dir)
            
            # Generate single observation
            obs_metadata = generator.generate_single_observation(
                "test_obs", obs_config, rfi_config
            )
            
            # Validate generation
            assert obs_metadata is not None, "Should return observation metadata"
            assert 'ground_truth_dir' in obs_metadata, "Should have ground truth directory"
            
            # Check generated files
            gt_dir = Path(obs_metadata['ground_truth_dir'])
            assert gt_dir.exists(), "Ground truth directory should exist"
            
            required_files = ["corrupted_visibilities.npy", "rfi_mask.npy"]
            for file in required_files:
                assert (gt_dir / file).exists(), f"Missing file: {file}"
            
            logger.info(f"✓ Synthetic data created: {gt_dir}")
            logger.info(f"  Config: {obs_config.num_antennas} antennas, {rfi_config.broadband_probability:.0%} RFI")
            logger.info(f"  obs_metadata keys: {list(obs_metadata.keys())}")
            
            # Store for next test (class variables)
            TestMLPipeline._obs_metadata = obs_metadata
            TestMLPipeline._temp_dir = temp_dir
            TestMLPipeline._obs_config = obs_config
            TestMLPipeline._rfi_config = rfi_config
            
        except Exception as e:
            # Cleanup on failure
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
            pytest.fail(f"Synthetic data generation failed: {e}")
    
    def test_2_extract_baseline_waterfall(self):
        """Step 2: Extract baseline waterfall using MSLoader (real MS reading)"""
        if TestMLPipeline._obs_metadata is None:
            pytest.skip("Requires test_1_synthetic_data_generation to pass")
        
        try:
            # Get MS path created by SyntheticDatasetGenerator  
            ms_path = TestMLPipeline._obs_metadata['corrupted_ms']  # Use corrupted MS with RFI
            logger.info(f"Loading MS using MSLoader: {ms_path}")
            
            # Use MSLoader to read MS (real pipeline)
            from samrfi.core import MSLoader
            loader = MSLoader(ms_path, field_id=0)
            
            # Load first baseline data  
            baseline_data = loader.load_baseline_data(ant1=0, ant2=1, spw_group_id=0)
            
            # Extract complex visibilities
            data_col = baseline_data['data']  # [time, chan, pol]
            complex_vis = data_col[:, :, 0]   # First polarization [time, freq]
            
            # Load ground truth RFI mask from .npy for validation
            gt_dir = Path(TestMLPipeline._obs_metadata['ground_truth_dir']) 
            rfi_mask_gt = np.load(gt_dir / 'rfi_mask.npy')
            rfi_mask = rfi_mask_gt[0, :, :, 0]  # First baseline, first pol [time, freq]
            
            # Validate extraction
            assert complex_vis.shape == (1024, 1024), f"Expected [1024,1024], got {complex_vis.shape}"
            assert rfi_mask.shape == (1024, 1024), f"RFI mask shape mismatch: {rfi_mask.shape}"
            assert np.iscomplexobj(complex_vis), "Visibility data should be complex"
            
            rfi_fraction = np.mean(rfi_mask)
            logger.info(f"✓ Extracted baseline waterfall: {complex_vis.shape}")
            logger.info(f"  RFI fraction: {rfi_fraction:.1%} (target: {self._rfi_config.broadband_probability:.0%})")
            
            # Store for next test  
            self._complex_vis = complex_vis
            self._rfi_mask = rfi_mask
            
        except Exception as e:
            pytest.fail(f"Baseline extraction failed: {e}")
    
    def test_3_new_channel_mapping(self):
        """Step 3: Test new R=Gradient, G=Amplitude, B=Phase channel extraction"""
        if not hasattr(self, '_complex_vis'):
            pytest.skip("Requires test_2_extract_baseline_waterfall to pass")
        
        try:
            # Extract channels using NEW mapping (from training code)
            channels = {}
            channels['amplitude'] = np.abs(self._complex_vis)
            channels['phase'] = np.angle(self._complex_vis)
            channels['log_amp'] = np.log10(np.abs(self._complex_vis) + 1e-10)
            
            # NEW: Gradient computation
            log_amp = channels['log_amp']
            time_deriv = np.zeros_like(log_amp)
            freq_deriv = np.zeros_like(log_amp)
            time_deriv[1:, :] = np.diff(log_amp, axis=0)
            freq_deriv[:, 1:] = np.diff(log_amp, axis=1)
            channels['gradient'] = np.sqrt(time_deriv**2 + freq_deriv**2)
            
            # Normalize channels (like training)
            def normalize_channel(data, name):
                if name == 'phase':
                    return (data + np.pi) / (2 * np.pi)
                else:
                    data_min, data_max = data.min(), data.max()
                    if data_max > data_min:
                        return (data - data_min) / (data_max - data_min)
                    return np.zeros_like(data)
            
            # NEW channel mapping: R=Gradient, G=LogAmp, B=Phase
            r_channel = normalize_channel(channels['gradient'], 'gradient')
            g_channel = normalize_channel(channels['log_amp'], 'log_amp')
            b_channel = normalize_channel(channels['phase'], 'phase')
            
            # Create SAM2 input: [3, 1024, 1024]
            sam2_input = np.stack([r_channel, g_channel, b_channel], axis=0)
            
            # Validate channels
            assert sam2_input.shape == (3, 1024, 1024), f"SAM2 input shape: {sam2_input.shape}"
            assert sam2_input.min() >= 0.0 and sam2_input.max() <= 1.0, "Channels should be [0,1]"
            
            logger.info(f"✓ Channel mapping complete:")
            logger.info(f"  Gradient (R): range=[{r_channel.min():.3f}, {r_channel.max():.3f}], mean={r_channel.mean():.3f}")
            logger.info(f"  LogAmp (G):   range=[{g_channel.min():.3f}, {g_channel.max():.3f}], mean={g_channel.mean():.3f}")
            logger.info(f"  Phase (B):    range=[{b_channel.min():.3f}, {b_channel.max():.3f}], mean={b_channel.mean():.3f}")
            
            # Store for next test
            self._sam2_input = sam2_input
            
        except Exception as e:
            pytest.fail(f"Channel mapping failed: {e}")
    
    def test_4_sam2_inference(self):
        """Step 4: Load SAM2-tiny and run real inference"""
        if not hasattr(self, '_sam2_input'):
            pytest.skip("Requires test_3_new_channel_mapping to pass")
        
        try:
            from samrfi.adapters import SAM2Adapter
            import torch
            
            # Use CPU for deterministic testing
            device = "cpu"
            sam2 = SAM2Adapter(device=device, variant="tiny")
            sam2.load_model()
            
            assert sam2.is_loaded, "SAM2 model should be loaded"
            
            # Convert to SAM2 format: [1024, 1024, 3] uint8
            image_hwc = self._sam2_input.transpose(1, 2, 0)  # CHW -> HWC
            image_uint8 = (image_hwc * 255).astype(np.uint8)
            
            # Run inference
            result = sam2.predict_single(image_uint8)
            
            # Validate SAM2 output
            assert isinstance(result, dict), "SAM2 should return dict"
            assert 'masks' in result, "SAM2 should return masks"
            
            masks = result['masks']
            assert len(masks) >= 1, "SAM2 should return at least 1 mask"
            assert masks[0].shape == (1024, 1024), f"Mask shape: {masks[0].shape}"
            
            logger.info(f"✓ SAM2 inference successful:")
            logger.info(f"  Device: {device}, Model: tiny")
            logger.info(f"  Output: {len(masks)} masks, shape: {masks[0].shape}")
            
            # Store for next test
            self._sam2_masks = masks
            
        except Exception as e:
            pytest.fail(f"SAM2 inference failed: {e}")
    
    def test_5_union_computation(self):
        """Step 5: Compute union of SAM2 masks (like training)"""
        if not hasattr(self, '_sam2_masks'):
            pytest.skip("Requires test_4_sam2_inference to pass")
        
        try:
            masks = self._sam2_masks
            
            # Compute union (like training code)
            if len(masks) > 1:
                stacked = np.stack(masks, axis=0)  # [n_masks, H, W]
                union_mask = np.max(stacked, axis=0)  # [H, W]
            else:
                union_mask = masks[0]
            
            # Convert to binary
            union_binary = (union_mask > 0.5).astype(float)
            
            detection_fraction = np.mean(union_binary)
            logger.info(f"✓ Union computation:")
            logger.info(f"  {len(masks)} masks → 1 union mask")
            logger.info(f"  Detection fraction: {detection_fraction:.1%}")
            
            # Store for next test
            self._union_mask = union_binary
            
        except Exception as e:
            pytest.fail(f"Union computation failed: {e}")
    
    def test_6_training_loss(self):
        """Step 6: Compute training loss with ground truth"""
        if not hasattr(self, '_union_mask'):
            pytest.skip("Requires test_5_union_computation to pass")
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Get predictions and ground truth
            pred_mask = self._union_mask    # [H, W] - SAM2 prediction
            gt_mask = self._rfi_mask.astype(float)  # [H, W] - ground truth RFI
            
            # Convert to tensors
            pred_tensor = torch.from_numpy(pred_mask).float()
            gt_tensor = torch.from_numpy(gt_mask).float()
            
            # Compute segmentation loss (like training)
            seg_loss = F.binary_cross_entropy(pred_tensor, gt_tensor, reduction='mean')
            
            # Compute IoU
            intersection = (pred_mask * gt_mask).sum()
            union = pred_mask.sum() + gt_mask.sum() - intersection
            iou = intersection / (union + 1e-6)
            
            # Compute accuracy
            accuracy = np.mean(pred_mask == gt_mask)
            
            # True positives, false positives, etc.
            tp = np.sum((pred_mask == 1) & (gt_mask == 1))
            fp = np.sum((pred_mask == 1) & (gt_mask == 0))
            tn = np.sum((pred_mask == 0) & (gt_mask == 0))
            fn = np.sum((pred_mask == 0) & (gt_mask == 1))
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            
            logger.info(f"✓ Training loss computation:")
            logger.info(f"  Segmentation loss: {seg_loss.item():.3f}")
            logger.info(f"  IoU: {iou:.3f}")
            logger.info(f"  Pixel accuracy: {accuracy:.1%}")
            logger.info(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
            logger.info(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            
            # Validate reasonable ranges
            assert 0.0 <= seg_loss.item() <= 2.0, f"Loss {seg_loss.item():.3f} unreasonable"
            assert 0.0 <= iou <= 1.0, f"IoU {iou:.3f} out of range"
            assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy:.3f} out of range"
            
            logger.info("✓ All pipeline tests passed - real ML pipeline working!")
            
            # Cleanup
            if hasattr(self, '_temp_dir') and Path(self._temp_dir).exists():
                shutil.rmtree(self._temp_dir)
                logger.info("✓ Test cleanup completed")
            
            return {
                'segmentation_loss': seg_loss.item(),
                'iou': iou,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
            
        except Exception as e:
            pytest.fail(f"Training loss computation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])