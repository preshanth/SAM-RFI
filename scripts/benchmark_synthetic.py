#!/usr/bin/env python
"""
Gate 0 entry point: benchmark flaggers on synthetic RFI with known ground truth.

Injects synthetic RFI into a template MS, runs SAM-RFI plus the classical
flaggers (tfcrop, rflag, tfcrop+rflag), and scores every one against the exact
ground-truth mask (segmentation metrics) plus downstream flagging-quality
metrics. Orchestration lives in samrfi.benchmark; this is a thin CLI.

Usage:
    python benchmark_synthetic.py template.ms \
        --config configs/validation.yaml \
        --model polarimetric/sam-rfi/large \
        --output ./benchmark_gate0
"""

import argparse

from samrfi.benchmark import run_gate0


def main():
    parser = argparse.ArgumentParser(description="Gate 0: flaggers vs synthetic ground truth")
    parser.add_argument("ms_path", help="Template MS to inject synthetic data into")
    parser.add_argument("--config", required=True, help="Data config YAML (synthetic section)")
    parser.add_argument("--output", default="./benchmark_gate0", help="Output directory")
    parser.add_argument("--model", default=None,
                        help="SAM-RFI checkpoint or HF repo id (omit to skip SAM)")
    parser.add_argument("--num-antennas", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="SAM-RFI probability threshold (default: mean)")
    parser.add_argument("--classical", nargs="*", default=None,
                        help="Subset of classical flaggers (default: all)")
    args = parser.parse_args()

    run_gate0(
        ms_path=args.ms_path,
        config_path=args.config,
        output_dir=args.output,
        model_path=args.model,
        num_antennas=args.num_antennas,
        classical=args.classical,
        sam_kwargs={"threshold": args.threshold} if args.threshold is not None else None,
    )


if __name__ == "__main__":
    main()
