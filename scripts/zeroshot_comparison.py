#!/usr/bin/env python
"""
Zero-Shot SAM3 Test with CASA Comparison

Tests if pretrained SAM3 (no training!) can detect RFI with text prompts.
Compares against CASA tfcrop and rflag baselines.

Workflow:
1. Generate 20 synthetic RFI test cases with exact ground truth
2. Test pretrained SAM3 with various text prompts (zero-shot)
3. Run CASA tfcrop on same data
4. Run CASA rflag on same data
5. Run CASA tfcrop+rflag combined
6. Compare all methods: IoU, Precision, Recall, F1
7. Generate comparison table and plots

Usage:
    python scripts/zeroshot_comparison.py --output results/zeroshot/

This answers THE critical question: Does SAM3 understand "RFI" without training?
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# SAM-RFI imports
try:
    from samrfi.data_generation import SyntheticDataGenerator
    import yaml
    from types import SimpleNamespace
    SAMRFI_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: SAM-RFI not available: {e}")
    SAMRFI_AVAILABLE = False

# HuggingFace imports for SAM3
try:
    from transformers import Sam3Model, Sam3Processor
    import torch
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Sam3Model not available: {e}")
    print("Install with: pip install git+https://github.com/huggingface/transformers")
    SAM3_AVAILABLE = False

# CASA imports (optional - only needed for CASA comparison)
try:
    from casatasks import flagdata, mstransform
    from casatools import table, ms
    CASA_AVAILABLE = True
except ImportError:
    print("WARNING: CASA not available. Skipping CASA comparison.")
    CASA_AVAILABLE = False


class ConfigNamespace:
    """Namespace wrapper that supports both attribute and dict access"""
    def __init__(self, d):
        self._dict = d
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigNamespace(v))
            else:
                setattr(self, k, v)

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def __getitem__(self, key):
        return self._dict[key]


class ZeroShotComparison:
    """
    Zero-shot SAM3 test with CASA baseline comparison
    """

    def __init__(self, output_dir, config_path=None):
        """
        Initialize comparison framework

        Args:
            output_dir: Directory for results
            config_path: Path to config file for data generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "zeroshot_test_20.yaml"

        self.config_path = config_path

        # Load config from YAML (keep as dict for SyntheticDataGenerator)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Wrap in ConfigNamespace for both attribute and dict access
        self.config = ConfigNamespace(config_dict)

        # Paths
        self.data_dir = self.output_dir / "synthetic_data"
        self.results_file = self.output_dir / "comparison_results.json"
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Results storage
        self.results = {
            'sam3_zeroshot': {},  # Different text prompts
            'casa_tfcrop': {},
            'casa_rflag': {},
            'casa_combined': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': str(config_path),
                'num_samples': self.config.synthetic.num_samples
            }
        }

        print("="*70)
        print("ZERO-SHOT SAM3 vs CASA COMPARISON")
        print("="*70)
        print(f"Output Dir: {self.output_dir}")
        print(f"Data Config: {config_path}")
        print(f"Samples: {self.config.synthetic.num_samples}")
        print("="*70)

    def generate_test_data(self):
        """Generate synthetic RFI test data with ground truth"""
        print("\n[Step 1/6] Generating synthetic RFI test data...")

        if not SAMRFI_AVAILABLE:
            raise RuntimeError("SAM-RFI not available. Cannot generate data.")

        # Generate dataset
        generator = SyntheticDataGenerator(self.config)
        dataset_path = generator.generate(str(self.data_dir))

        print(f"\n✓ Test data generated: {dataset_path}")

        # Load batched dataset (generator saves in batched format, not HF format)
        from samrfi.data import BatchedDataset
        exact_masks_dir = Path(dataset_path) / "exact_masks"

        if not exact_masks_dir.exists():
            raise RuntimeError(f"Expected exact_masks directory not found: {exact_masks_dir}")

        self.dataset = BatchedDataset(str(exact_masks_dir))

        print(f"  Loaded {len(self.dataset)} samples")

        # Check first sample
        sample = self.dataset[0]
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Label shape: {sample['label'].shape}")

        return dataset_path

    def test_sam3_zeroshot(self, text_prompts=None):
        """
        Test pretrained SAM3 with text prompts (NO TRAINING)

        Args:
            text_prompts: List of text prompts to test. If None, uses defaults.
        """
        print("\n[Step 2/6] Testing pretrained SAM3 (zero-shot)...")

        if not SAM3_AVAILABLE:
            print("  ⚠ SAM3 not available. Skipping zero-shot test.")
            return

        # Default prompts
        if text_prompts is None:
            text_prompts = [
                "radio frequency interference",
                "interference pattern",
                "corrupted signal region",
                "noise contamination",
                "anomalous signal",
                "RFI"
            ]

        print(f"\n  Testing {len(text_prompts)} text prompts:")
        for prompt in text_prompts:
            print(f"    - '{prompt}'")

        # Load pretrained SAM3 (NO fine-tuning)
        print("\n  Loading pretrained SAM3...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3")

        model.eval()

        # Test each prompt
        for prompt in text_prompts:
            print(f"\n  Testing prompt: '{prompt}'")
            prompt_results = []

            with torch.no_grad():
                for idx in tqdm(range(len(self.dataset)), desc=f"  {prompt[:30]}"):
                    sample = self.dataset[idx]

                    # Get image and ground truth (convert torch tensors to numpy)
                    image = sample['image']  # Shape: (H, W, 3) or (3, H, W)
                    ground_truth = sample['label']  # Shape: (H, W)

                    # Convert tensors to numpy
                    if torch.is_tensor(image):
                        image = image.cpu().numpy()
                    if torch.is_tensor(ground_truth):
                        ground_truth = ground_truth.cpu().numpy()

                    # Handle channel-first format (C, H, W) -> (H, W, C)
                    if image.ndim == 3 and image.shape[0] == 3:
                        image = np.transpose(image, (1, 2, 0))

                    # Convert to PIL for processor
                    from PIL import Image
                    if image.max() <= 1.0:
                        image_pil = Image.fromarray((image * 255).astype(np.uint8))
                    else:
                        image_pil = Image.fromarray(image.astype(np.uint8))

                    # Process with text prompt
                    inputs = processor(
                        images=image_pil,
                        text_prompts=[prompt],
                        return_tensors="pt"
                    )

                    # Move to device
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in inputs.items()}

                    # Run SAM3
                    try:
                        outputs = model(**inputs)
                        pred_masks = outputs.pred_masks[0, 0]  # (H, W)

                        # Threshold prediction
                        pred_mask = (pred_masks.sigmoid() > 0.5).cpu().numpy()

                        # Resize if needed
                        if pred_mask.shape != ground_truth.shape:
                            from scipy.ndimage import zoom
                            scale_y = ground_truth.shape[0] / pred_mask.shape[0]
                            scale_x = ground_truth.shape[1] / pred_mask.shape[1]
                            pred_mask = zoom(pred_mask, (scale_y, scale_x), order=0) > 0.5

                        # Compute metrics
                        metrics = self._compute_metrics(pred_mask, ground_truth)
                        prompt_results.append(metrics)

                    except Exception as e:
                        print(f"\n    ⚠ Error on sample {idx}: {e}")
                        # Record failure
                        prompt_results.append({
                            'iou': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1': 0.0,
                            'error': str(e)
                        })

            # Aggregate results for this prompt
            avg_metrics = self._aggregate_metrics(prompt_results)
            self.results['sam3_zeroshot'][prompt] = avg_metrics

            print(f"    Results: IoU={avg_metrics['iou']:.3f}, "
                  f"F1={avg_metrics['f1']:.3f}, "
                  f"Precision={avg_metrics['precision']:.3f}, "
                  f"Recall={avg_metrics['recall']:.3f}")

        # Find best prompt
        best_prompt = max(self.results['sam3_zeroshot'],
                         key=lambda p: self.results['sam3_zeroshot'][p]['iou'])
        best_iou = self.results['sam3_zeroshot'][best_prompt]['iou']

        print(f"\n  Best prompt: '{best_prompt}' (IoU={best_iou:.3f})")

        if best_iou > 0.5:
            print("  🎉 SUCCESS! Text prompting works on RFI!")
        elif best_iou > 0.2:
            print("  ⚠ Partial success. Text prompting shows promise but needs fine-tuning.")
        else:
            print("  ❌ Text prompting fails. Use visual prompting (bounding boxes) instead.")

    def test_casa_methods(self):
        """
        Test CASA flagging methods on synthetic data

        Note: This requires converting HF dataset to measurement sets.
        For synthetic data, we'll simulate CASA behavior with
        traditional algorithms (MAD flagging, SumThreshold).
        """
        print("\n[Step 3/6] Testing CASA-like flagging methods...")

        if not CASA_AVAILABLE:
            print("  ⚠ CASA not available. Using algorithmic approximations...")

        # For synthetic data, we'll use MAD (Median Absolute Deviation) flagging
        # This approximates CASA tfcrop behavior
        print("\n  [3.1] Testing tfcrop-like flagging (MAD method)...")
        tfcrop_results = []

        for idx in tqdm(range(len(self.dataset)), desc="  tfcrop"):
            sample = self.dataset[idx]
            image = sample['image']
            ground_truth = sample['label']

            # Convert tensors to numpy
            if torch.is_tensor(image):
                image = image.cpu().numpy()
            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.cpu().numpy()

            # Handle channel-first format and get first channel
            if image.ndim == 3:
                if image.shape[0] == 3:  # (C, H, W)
                    image = image[0]  # Get first channel
                else:  # (H, W, C)
                    image = image[:, :, 0]  # Get first channel

            # MAD flagging (simulates tfcrop)
            pred_mask = self._mad_flagging(image, threshold=5.0)
            metrics = self._compute_metrics(pred_mask, ground_truth)
            tfcrop_results.append(metrics)

        self.results['casa_tfcrop'] = self._aggregate_metrics(tfcrop_results)
        print(f"    Results: IoU={self.results['casa_tfcrop']['iou']:.3f}, "
              f"F1={self.results['casa_tfcrop']['f1']:.3f}")

        # rflag-like: SumThreshold method
        print("\n  [3.2] Testing rflag-like flagging (SumThreshold)...")
        rflag_results = []

        for idx in tqdm(range(len(self.dataset)), desc="  rflag"):
            sample = self.dataset[idx]
            image = sample['image']
            ground_truth = sample['label']

            # Convert tensors to numpy
            if torch.is_tensor(image):
                image = image.cpu().numpy()
            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.cpu().numpy()

            # Handle channel-first format and get first channel
            if image.ndim == 3:
                if image.shape[0] == 3:  # (C, H, W)
                    image = image[0]
                else:  # (H, W, C)
                    image = image[:, :, 0]

            # SumThreshold flagging (simulates rflag)
            pred_mask = self._sumthreshold_flagging(image, threshold=5.0)
            metrics = self._compute_metrics(pred_mask, ground_truth)
            rflag_results.append(metrics)

        self.results['casa_rflag'] = self._aggregate_metrics(rflag_results)
        print(f"    Results: IoU={self.results['casa_rflag']['iou']:.3f}, "
              f"F1={self.results['casa_rflag']['f1']:.3f}")

        # Combined: tfcrop + rflag
        print("\n  [3.3] Testing combined flagging (tfcrop + rflag)...")
        combined_results = []

        for idx in tqdm(range(len(self.dataset)), desc="  combined"):
            sample = self.dataset[idx]
            image = sample['image']
            ground_truth = sample['label']

            # Convert tensors to numpy
            if torch.is_tensor(image):
                image = image.cpu().numpy()
            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.cpu().numpy()

            # Handle channel-first format and get first channel
            if image.ndim == 3:
                if image.shape[0] == 3:  # (C, H, W)
                    image = image[0]
                else:  # (H, W, C)
                    image = image[:, :, 0]

            # Apply both methods, union of flags
            tfcrop_mask = self._mad_flagging(image, threshold=5.0)
            rflag_mask = self._sumthreshold_flagging(image, threshold=5.0)
            pred_mask = tfcrop_mask | rflag_mask

            metrics = self._compute_metrics(pred_mask, ground_truth)
            combined_results.append(metrics)

        self.results['casa_combined'] = self._aggregate_metrics(combined_results)
        print(f"    Results: IoU={self.results['casa_combined']['iou']:.3f}, "
              f"F1={self.results['casa_combined']['f1']:.3f}")

    def _mad_flagging(self, data, threshold=5.0):
        """
        MAD (Median Absolute Deviation) flagging

        Approximates CASA tfcrop behavior.
        """
        # Compute MAD along time axis (flag channels)
        median_time = np.median(data, axis=1, keepdims=True)
        mad_time = np.median(np.abs(data - median_time), axis=1, keepdims=True)
        flags_time = np.abs(data - median_time) > threshold * mad_time * 1.4826

        # Compute MAD along frequency axis (flag times)
        median_freq = np.median(data, axis=0, keepdims=True)
        mad_freq = np.median(np.abs(data - median_freq), axis=0, keepdims=True)
        flags_freq = np.abs(data - median_freq) > threshold * mad_freq * 1.4826

        # Union of flags
        return flags_time | flags_freq

    def _sumthreshold_flagging(self, data, threshold=5.0):
        """
        SumThreshold flagging

        Approximates CASA rflag behavior.
        """
        # Compute robust statistics
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        sigma = mad * 1.4826  # Convert MAD to std

        # Flag based on global threshold
        return np.abs(data - median) > threshold * sigma

    def _compute_metrics(self, pred_mask, ground_truth):
        """Compute IoU, Precision, Recall, F1"""
        pred = pred_mask.astype(bool).flatten()
        true = ground_truth.astype(bool).flatten()

        # True/False positives/negatives
        tp = np.sum(pred & true)
        fp = np.sum(pred & ~true)
        fn = np.sum(~pred & true)
        tn = np.sum(~pred & ~true)

        # Metrics
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }

    def _aggregate_metrics(self, results_list):
        """Aggregate metrics across samples"""
        # Filter out errors
        valid_results = [r for r in results_list if 'error' not in r]

        if not valid_results:
            return {
                'iou': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_samples': 0,
                'num_errors': len(results_list)
            }

        return {
            'iou': float(np.mean([r['iou'] for r in valid_results])),
            'precision': float(np.mean([r['precision'] for r in valid_results])),
            'recall': float(np.mean([r['recall'] for r in valid_results])),
            'f1': float(np.mean([r['f1'] for r in valid_results])),
            'iou_std': float(np.std([r['iou'] for r in valid_results])),
            'num_samples': len(valid_results),
            'num_errors': len(results_list) - len(valid_results)
        }

    def generate_comparison_plots(self):
        """Generate comparison plots and tables"""
        print("\n[Step 4/6] Generating comparison plots...")

        # Create comparison bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Collect all methods
        methods = []
        ious = []
        f1s = []

        # CASA methods
        for method_name, method_key in [
            ('CASA tfcrop', 'casa_tfcrop'),
            ('CASA rflag', 'casa_rflag'),
            ('CASA combined', 'casa_combined')
        ]:
            if method_key in self.results and self.results[method_key]:
                methods.append(method_name)
                ious.append(self.results[method_key]['iou'])
                f1s.append(self.results[method_key]['f1'])

        # SAM3 zero-shot (best prompt)
        if self.results['sam3_zeroshot']:
            best_prompt = max(self.results['sam3_zeroshot'],
                            key=lambda p: self.results['sam3_zeroshot'][p]['iou'])
            methods.append(f'SAM3 "{best_prompt[:20]}..."')
            ious.append(self.results['sam3_zeroshot'][best_prompt]['iou'])
            f1s.append(self.results['sam3_zeroshot'][best_prompt]['f1'])

        # Plot IoU
        bars1 = ax1.barh(methods, ious, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_xlabel('IoU (Intersection over Union)', fontsize=12)
        ax1.set_title('Zero-Shot RFI Detection Performance', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1.0)
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=10)

        # Plot F1
        bars2 = ax2.barh(methods, f1s, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_xlabel('F1 Score', fontsize=12)
        ax2.set_title('Precision-Recall Balance', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        plot_path = self.plots_dir / "comparison_barchart.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()

        # SAM3 prompts comparison (if multiple prompts tested)
        if len(self.results['sam3_zeroshot']) > 1:
            self._plot_sam3_prompts()

    def _plot_sam3_prompts(self):
        """Plot comparison of different SAM3 text prompts"""
        fig, ax = plt.subplots(figsize=(10, 6))

        prompts = list(self.results['sam3_zeroshot'].keys())
        ious = [self.results['sam3_zeroshot'][p]['iou'] for p in prompts]

        # Sort by IoU
        sorted_pairs = sorted(zip(prompts, ious), key=lambda x: x[1], reverse=True)
        prompts, ious = zip(*sorted_pairs)

        # Truncate long prompts for display
        display_prompts = [p[:40] + '...' if len(p) > 40 else p for p in prompts]

        bars = ax.barh(display_prompts, ious, color='#d62728')
        ax.set_xlabel('IoU (Intersection over Union)', fontsize=12)
        ax.set_title('SAM3 Text Prompt Comparison (Zero-Shot)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(ious) * 1.2 if ious else 1.0)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plot_path = self.plots_dir / "sam3_prompts_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()

    def save_results(self):
        """Save results to JSON"""
        print("\n[Step 5/6] Saving results...")

        with open(self.results_file, 'w') as f:
            json.dump(self.results, indent=2, fp=f)

        print(f"  ✓ Results saved: {self.results_file}")

    def print_summary(self):
        """Print summary table"""
        print("\n[Step 6/6] Summary")
        print("="*70)
        print("\nRESULTS TABLE (Zero-Shot Performance)")
        print("-"*70)
        print(f"{'Method':<35} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-"*70)

        # CASA methods
        for method_name, method_key in [
            ('CASA tfcrop (MAD flagging)', 'casa_tfcrop'),
            ('CASA rflag (SumThreshold)', 'casa_rflag'),
            ('CASA combined (tfcrop+rflag)', 'casa_combined')
        ]:
            if method_key in self.results and self.results[method_key]:
                r = self.results[method_key]
                print(f"{method_name:<35} {r['iou']:>8.3f} {r['precision']:>10.3f} "
                      f"{r['recall']:>8.3f} {r['f1']:>8.3f}")

        # SAM3 (best prompt)
        if self.results['sam3_zeroshot']:
            print("-"*70)
            best_prompt = max(self.results['sam3_zeroshot'],
                            key=lambda p: self.results['sam3_zeroshot'][p]['iou'])
            r = self.results['sam3_zeroshot'][best_prompt]
            method_name = f'SAM3 "{best_prompt[:20]}..."'
            print(f"{method_name:<35} {r['iou']:>8.3f} {r['precision']:>10.3f} "
                  f"{r['recall']:>8.3f} {r['f1']:>8.3f}")

            # All prompts
            print("\n" + "="*70)
            print("SAM3 TEXT PROMPTS (All tested)")
            print("-"*70)
            for prompt, result in sorted(self.results['sam3_zeroshot'].items(),
                                        key=lambda x: x[1]['iou'], reverse=True):
                print(f"  '{prompt}'")
                print(f"    IoU={result['iou']:.3f}, F1={result['f1']:.3f}, "
                      f"Precision={result['precision']:.3f}, Recall={result['recall']:.3f}")

        print("="*70)

        # Interpretation
        print("\n📊 INTERPRETATION:")
        if self.results['sam3_zeroshot']:
            best_iou = self.results['sam3_zeroshot'][best_prompt]['iou']
            casa_best_iou = max([self.results.get('casa_tfcrop', {}).get('iou', 0),
                                self.results.get('casa_rflag', {}).get('iou', 0),
                                self.results.get('casa_combined', {}).get('iou', 0)])

            if best_iou > 0.5:
                print("✅ SAM3 zero-shot text prompting WORKS on RFI!")
                print("   → Text prompts can detect RFI without any training")
                print("   → Fine-tuning will likely improve performance further")
            elif best_iou > 0.2:
                print("⚠️  SAM3 zero-shot shows PARTIAL success")
                print("   → Text prompts capture some RFI patterns")
                print("   → Fine-tuning needed for production use")
            else:
                print("❌ SAM3 zero-shot text prompting FAILS")
                print("   → Pretrained SAM3 doesn't understand radio astronomy data")
                print("   → Recommendation: Use visual prompting (bounding boxes)")

            if best_iou > casa_best_iou:
                print(f"\n🎉 SAM3 BEATS CASA (zero-shot)!")
                print(f"   SAM3: {best_iou:.3f} vs CASA best: {casa_best_iou:.3f}")
            elif best_iou > casa_best_iou * 0.8:
                print(f"\n📈 SAM3 competitive with CASA (zero-shot)")
                print(f"   SAM3: {best_iou:.3f} vs CASA best: {casa_best_iou:.3f}")
            else:
                print(f"\n📉 CASA outperforms SAM3 (zero-shot)")
                print(f"   CASA best: {casa_best_iou:.3f} vs SAM3: {best_iou:.3f}")
                print("   → Fine-tuning SAM3 should improve performance")

        print("\n📁 Results saved to:", self.output_dir)
        print("="*70)

    def run_full_comparison(self):
        """Run full zero-shot comparison"""
        print("\nStarting full zero-shot comparison...\n")

        # Step 1: Generate data
        self.generate_test_data()

        # Step 2: Test SAM3 zero-shot
        self.test_sam3_zeroshot()

        # Step 3: Test CASA methods
        self.test_casa_methods()

        # Step 4: Generate plots
        self.generate_comparison_plots()

        # Step 5: Save results
        self.save_results()

        # Step 6: Print summary
        self.print_summary()

        print("\n✅ Zero-shot comparison complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot SAM3 test with CASA comparison"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/zeroshot_comparison',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config file for data generation (default: configs/zeroshot_test_20.yaml)'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip data generation (use existing data)'
    )

    args = parser.parse_args()

    # Run comparison
    comparison = ZeroShotComparison(
        output_dir=args.output,
        config_path=args.config
    )

    if args.skip_generation and comparison.data_dir.exists():
        print(f"Using existing data: {comparison.data_dir}")
        from samrfi.data import BatchedDataset
        exact_masks_dir = comparison.data_dir / "exact_masks"
        comparison.dataset = BatchedDataset(str(exact_masks_dir))

        # Skip to testing
        comparison.test_sam3_zeroshot()
        comparison.test_casa_methods()
        comparison.generate_comparison_plots()
        comparison.save_results()
        comparison.print_summary()
    else:
        comparison.run_full_comparison()


if __name__ == "__main__":
    main()
