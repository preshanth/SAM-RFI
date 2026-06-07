#!/usr/bin/env python
"""
Gate 1 entry point: benchmark flaggers on a real MS (no ground truth).

Runs SAM-RFI plus the classical flaggers on a real observation and scores each
by agreement against a reference flagger (default rflag) plus downstream
flagging-quality metrics (FFI, calcquality, flag fraction). Orchestration lives
in samrfi.benchmark; this is a thin CLI.

Usage:
    python benchmark_real.py /path/to/observation.ms \
        --model polarimetric/sam-rfi/large \
        --reference rflag \
        --output ./benchmark_gate1
"""

import argparse

from samrfi.benchmark import run_gate1


def main():
    parser = argparse.ArgumentParser(description="Gate 1: flaggers on a real MS (no GT)")
    parser.add_argument("ms_path", help="Real measurement set to flag")
    parser.add_argument("--output", default="./benchmark_gate1", help="Output directory")
    parser.add_argument("--model", default=None,
                        help="SAM-RFI checkpoint or HF repo id (omit to skip SAM)")
    parser.add_argument("--reference", default="rflag",
                        help="Flagger used as the agreement reference (default: rflag)")
    parser.add_argument("--num-antennas", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="SAM-RFI probability threshold (default: mean)")
    parser.add_argument("--classical", nargs="*", default=None,
                        help="Subset of classical flaggers (default: all)")
    args = parser.parse_args()

    run_gate1(
        ms_path=args.ms_path,
        output_dir=args.output,
        model_path=args.model,
        reference=args.reference,
        num_antennas=args.num_antennas,
        classical=args.classical,
        sam_kwargs={"threshold": args.threshold} if args.threshold is not None else None,
    )


if __name__ == "__main__":
    main()
