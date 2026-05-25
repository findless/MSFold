#!/usr/bin/env python3
"""Rank sampled structures by SLL (Sequence Log-Likelihood).

Example:
    python scripts/rank_sll.py \
        --sample_dir outputs/example \
        --out outputs/example/sll_ranking.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msfold.scoring.sll import rank_by_sll


def main():
    parser = argparse.ArgumentParser(
        description="Rank MSFold sampled structures by SLL."
    )
    parser.add_argument(
        "--sample_dir",
        required=True,
        help="Directory containing samples.csv from MSFold sampling.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for ranked CSV (e.g., sll_ranking.csv).",
    )
    args = parser.parse_args()

    samples_csv = os.path.join(args.sample_dir, "samples.csv")
    if not os.path.exists(samples_csv):
        print(f"Error: samples.csv not found in {args.sample_dir}")
        print("Run MSFold sampling first.")
        sys.exit(1)

    try:
        df = rank_by_sll(samples_csv, args.out)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Ranked {len(df)} samples by SLL.")
    print(f"Top 5:")
    for _, row in df.head(5).iterrows():
        print(
            f"  rank={row['rank']}  sll={row['sll']:.4f}  "
            f"nll={row['nll']:.2f}  path={row.get('structure_path', 'N/A')}"
        )
    print(f"Full ranking saved to: {args.out}")


if __name__ == "__main__":
    main()
