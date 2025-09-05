"""
SAM-RFI Command Line Interface

Basic CLI entry point for the SAM-RFI package.
"""

import argparse
import sys
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="samrfi",
        description="SAM-based Radio Frequency Interference Detection for Radio Astronomy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="samrfi 1.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process measurement set for RFI detection"
    )
    process_parser.add_argument("ms_path", help="Path to measurement set")
    process_parser.add_argument(
        "--antenna", type=int, default=0, help="Antenna ID to process (default: 0)"
    )
    process_parser.add_argument(
        "--model", default="sam2", help="SAM model version to use (default: sam2)"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show package and system information"
    )

    return parser


def cmd_process(args: argparse.Namespace) -> int:
    """Process measurement set command."""
    print(f"Processing measurement set: {args.ms_path}")
    print(f"Using antenna: {args.antenna}")
    print(f"Using model: {args.model}")

    # TODO: Implement actual processing in later stages
    print("Note: Full processing implementation coming in later stages")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show package information."""
    print("SAM-RFI Package Information")
    print("=" * 30)
    print("Version: 1.0.0")
    print("Status: Refactor Stage 1 - Package Infrastructure")

    # Check dependencies
    try:
        import numpy

        print(f"NumPy: {numpy.__version__}")
    except ImportError:
        print("NumPy: Not installed")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("PyTorch: Not installed")

    print("python-casacore: Disabled (not needed)")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "process":
        return cmd_process(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
