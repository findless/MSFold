#!/usr/bin/env python3
"""Run MSFold sampling on an input sequence.

Example:
        python scripts/run_msfold_sample.py \
            --fasta examples/example.fasta \
            --config configs/default.yaml \
            --out outputs/example \
            --device cuda
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml

# Allow running from the release_msfold directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfold.models.esm3 import load_esm3_client
from msfold.sampling.sampler import sample_from_sequence
from msfold.utils.io import read_fasta


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="MSFold: protein structure sampling via parallel tempering with block swap."
    )
    parser.add_argument(
        "--fasta", required=True, help="Path to input FASTA file (single sequence)."
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for results."
    )
    parser.add_argument(
        "--device", default="cuda", help="Device: cuda, cuda:0, or cpu."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional random seed."
    )
    args = parser.parse_args()

    # Load inputs
    sequence = read_fasta(args.fasta)
    print(f"Loaded sequence of length {len(sequence)} from {args.fasta}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {args.config}")

    # Load ESM3
    print("Loading ESM3 model (weights will download on first run if not cached)...")
    client = load_esm3_client(args.device)

    # Run sampling
    print("Starting MSFold sampling...")
    result = sample_from_sequence(
        sequence=sequence,
        client=client,
        config=config,
        output_dir=args.out,
        device=args.device,
        seed=args.seed,
        target_name=Path(args.fasta).stem,
    )

    print(f"Done. Samples saved to: {result['samples_path']}")
    print(f"Total time: {result['total_time_seconds']:.1f} s")


if __name__ == "__main__":
    main()
