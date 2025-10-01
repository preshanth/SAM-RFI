#!/usr/bin/env python
"""
End-to-end GPU validation script for SAM-RFI training
Profiles memory, GPU utilization, and performance across different batch sizes

Usage:
    python validate_gpu.py --dataset /path/to/dataset --config configs/sam2_training.yaml
"""

import argparse
import sys
import time
from pathlib import Path
import json

import torch
import numpy as np
from datasets import load_from_disk

# GPU profiling
try:
    import pynvml

    pynvml.nvmlInit()
    HAS_NVML = True
except:
    HAS_NVML = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")

from samrfi.training.sam2_trainer import SAM2Trainer
from samrfi.config.config_loader import ConfigLoader


class GPUMonitor:
    """Monitor GPU memory and utilization"""

    def __init__(self):
        self.has_nvml = HAS_NVML
        if self.has_nvml:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_memory_info(self):
        """Get current GPU memory usage in MB"""
        if not self.has_nvml:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                return {
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
                }
            return None

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "used_mb": mem_info.used / 1024**2,
            "total_mb": mem_info.total / 1024**2,
            "free_mb": mem_info.free / 1024**2,
            "utilization_pct": mem_info.used / mem_info.total * 100,
        }

    def get_utilization(self):
        """Get GPU utilization percentage"""
        if not self.has_nvml:
            return None
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return {"gpu_pct": util.gpu, "memory_pct": util.memory}

    def get_device_name(self):
        """Get GPU device name"""
        if not self.has_nvml:
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
            return "Unknown"
        return pynvml.nvmlDeviceGetName(self.handle)

    def cleanup(self):
        if self.has_nvml:
            pynvml.nvmlShutdown()


class TrainingProfiler:
    """Profile training performance"""

    def __init__(self, monitor: GPUMonitor):
        self.monitor = monitor
        self.results = []

    def profile_batch_size(self, dataset_wrapper, config, batch_size, num_epochs=1):
        """Profile training with specific batch size"""
        print(f"\n{'='*80}")
        print(f"Profiling batch_size={batch_size}")
        print(f"{'='*80}")

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Get initial memory
        mem_before = self.monitor.get_memory_info()

        # Create trainer
        trainer = SAM2Trainer(dataset_wrapper, device=config.device, dir_path="./validation_output")

        # Get profiling config (with defaults if not present)
        profiling_config = getattr(config, 'profiling', {})
        if isinstance(profiling_config, dict):
            profiling_enabled = profiling_config.get('enabled', True)
            activities_config = profiling_config.get('activities', {})
            cpu_profiling = activities_config.get('cpu', True) if isinstance(activities_config, dict) else True
            cuda_profiling = activities_config.get('cuda', True) if isinstance(activities_config, dict) else True
            record_shapes = profiling_config.get('record_shapes', False)
            profile_memory = profiling_config.get('profile_memory', False)
            with_stack = profiling_config.get('with_stack', False)
        else:
            # Handle DataConfig object
            profiling_enabled = getattr(profiling_config, 'enabled', True)
            activities_config = getattr(profiling_config, 'activities', {})
            cpu_profiling = activities_config.get('cpu', True) if hasattr(activities_config, 'get') else True
            cuda_profiling = activities_config.get('cuda', True) if hasattr(activities_config, 'get') else True
            record_shapes = getattr(profiling_config, 'record_shapes', False)
            profile_memory = getattr(profiling_config, 'profile_memory', False)
            with_stack = getattr(profiling_config, 'with_stack', False)

        # Start profiling
        start_time = time.time()

        try:
            if profiling_enabled:
                # Build activities list
                activities = []
                if cpu_profiling:
                    activities.append(torch.profiler.ProfilerActivity.CPU)
                if cuda_profiling:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)

                print(f"  Profiling: enabled (shapes={record_shapes}, memory={profile_memory}, stack={with_stack})")

                # Enable PyTorch profiler with config options
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=with_stack,
                ) as prof:

                    losses = trainer.train(
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        sam_checkpoint=config.model_checkpoint,
                        learning_rate=config.learning_rate,
                        plot=False,
                    )
            else:
                print(f"  Profiling: disabled")
                prof = None
                losses = trainer.train(
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    sam_checkpoint=config.model_checkpoint,
                    learning_rate=config.learning_rate,
                    plot=False,
                )

            end_time = time.time()

            # Get final memory
            mem_after = self.monitor.get_memory_info()
            util = self.monitor.get_utilization()

            # Get peak memory
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            else:
                peak_memory_mb = None

            # Calculate metrics
            duration = end_time - start_time
            samples_per_sec = (len(dataset_wrapper.dataset) * num_epochs) / duration

            # Extract key profiler stats if profiling was enabled
            profiler_stats = None
            if profiling_enabled and prof is not None:
                key_averages = prof.key_averages()
                top_cuda_ops = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:5]
                profiler_stats = [
                    {
                        "name": op.key,
                        "cuda_time_ms": op.cuda_time_total / 1000,  # Convert to ms
                        "cpu_time_ms": op.cpu_time_total / 1000,
                        "count": op.count,
                    }
                    for op in top_cuda_ops
                ]

            result = {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "success": True,
                "duration_sec": duration,
                "samples_per_sec": samples_per_sec,
                "final_loss": losses[-1],
                "memory_before_mb": mem_before,
                "memory_after_mb": mem_after,
                "peak_memory_mb": peak_memory_mb,
                "gpu_utilization": util,
                "top_cuda_ops": profiler_stats,  # None if profiling disabled
            }

            print(f"\n✓ Success!")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Throughput: {samples_per_sec:.2f} samples/sec")
            print(f"  Final loss: {losses[-1]:.6f}")
            if peak_memory_mb:
                print(f"  Peak GPU memory: {peak_memory_mb:.0f} MB")
            if util:
                print(f"  GPU utilization: {util['gpu_pct']}%")

            if profiling_enabled and prof is not None:
                print("\nTop CUDA operations:")
                print(key_averages.table(sort_by="cuda_time_total", row_limit=5))

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n✗ Out of memory!")
                result = {
                    "batch_size": batch_size,
                    "success": False,
                    "error": "OOM",
                    "message": str(e),
                }
            else:
                raise

        self.results.append(result)

        # Explicit cleanup to prevent memory accumulation between batch size tests
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        return result

    def find_optimal_batch_size(self, dataset_wrapper, config, max_batch_size=64):
        """Binary search for optimal batch size"""
        print(f"\n{'='*80}")
        print("Finding optimal batch size...")
        print(f"{'='*80}")

        successful_batch_sizes = []

        # Test powers of 2
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > max_batch_size:
                break

            result = self.profile_batch_size(dataset_wrapper, config, batch_size, num_epochs=1)

            if result["success"]:
                successful_batch_sizes.append(batch_size)
            else:
                # Stop at first OOM
                break

        if successful_batch_sizes:
            optimal = max(successful_batch_sizes)
            print(f"\n✓ Optimal batch size: {optimal}")
            return optimal
        else:
            print("\n✗ No successful batch sizes found!")
            return None

    def generate_report(self, output_path="validation_report.json"):
        """Generate JSON report"""
        device_name = self.monitor.get_device_name()
        mem_info = self.monitor.get_memory_info()

        report = {
            "device": device_name,
            "total_memory_mb": mem_info["total_mb"] if mem_info else None,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "results": self.results,
            "summary": {
                "successful_batch_sizes": [r["batch_size"] for r in self.results if r["success"]],
                "failed_batch_sizes": [r["batch_size"] for r in self.results if not r["success"]],
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved to: {output_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description="Validate SAM-RFI training on GPU with profiling")
    parser.add_argument("--dataset", required=True, help="Path to HuggingFace dataset")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument(
        "--max-batch-size", type=int, default=64, help="Maximum batch size to test (default: 64)"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs for profiling (default: 1)"
    )
    parser.add_argument(
        "--output",
        default="validation_report.json",
        help="Output report path (default: validation_report.json)",
    )

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ Error: CUDA not available!")
        return 1

    print("=" * 80)
    print("SAM-RFI GPU Validation")
    print("=" * 80)

    # Load config
    print(f"\nLoading config: {args.config}")
    config = ConfigLoader.load(args.config)
    config.device = "cuda"  # Force CUDA

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_from_disk(args.dataset)
    print(f"  Loaded {len(dataset)} samples")

    # Create wrapper
    class DatasetWrapper:
        def __init__(self, ds):
            self.dataset = ds
            self.dataset_params = {
                "stretch": config.stretch,
                "flag_sigma": config.flag_sigma,
                "patch_method": "patchify",
                "patch_size": config.patch_size,
            }
            # Use real dataset for length (only used in plot titles via len())
            # No need to create 53GB fake array when we have real data
            self.patched_data_norm_only = ds

    dataset_wrapper = DatasetWrapper(dataset)

    # Create monitor and profiler
    monitor = GPUMonitor()
    profiler = TrainingProfiler(monitor)

    print(f"\nGPU Device: {monitor.get_device_name()}")
    mem_info = monitor.get_memory_info()
    if mem_info:
        print(f"Total Memory: {mem_info['total_mb']:.0f} MB")

    # Find optimal batch size
    optimal_batch_size = profiler.find_optimal_batch_size(
        dataset_wrapper, config, max_batch_size=args.max_batch_size
    )

    if optimal_batch_size:
        print(f"\n{'='*80}")
        print(f"Running full validation with batch_size={optimal_batch_size}")
        print(f"{'='*80}")

        profiler.profile_batch_size(
            dataset_wrapper, config, optimal_batch_size, num_epochs=args.num_epochs
        )

    # Generate report
    profiler.generate_report(args.output)

    # Print summary
    print(f"\n{'='*80}")
    print("Validation Summary")
    print(f"{'='*80}")

    successful = [r for r in profiler.results if r["success"]]
    if successful:
        print("\nSuccessful configurations:")
        for r in successful:
            print(
                f"  batch_size={r['batch_size']:2d}: "
                f"{r['samples_per_sec']:6.2f} samples/sec, "
                f"loss={r['final_loss']:.6f}, "
                f"peak_mem={r['peak_memory_mb']:.0f}MB"
            )

    failed = [r for r in profiler.results if not r["success"]]
    if failed:
        print("\nFailed configurations:")
        for r in failed:
            print(f"  batch_size={r['batch_size']:2d}: {r['error']}")

    # Cleanup
    monitor.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
