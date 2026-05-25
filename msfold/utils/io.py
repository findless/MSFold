"""File I/O utilities: FASTA reading, CSV writing, output directories."""

import os
import csv
import json


def read_fasta(fasta_path: str) -> str:
    """Read a single sequence from a FASTA file.

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        Amino acid sequence string (uppercase).

    Raises:
        ValueError: If file is empty or contains multiple sequences.
    """
    sequences = []
    current_lines = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_lines:
                    sequences.append("".join(current_lines))
                    current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sequences.append("".join(current_lines))

    if len(sequences) == 0:
        raise ValueError(f"No sequence found in {fasta_path}")
    if len(sequences) > 1:
        raise ValueError(
            f"Multiple sequences ({len(sequences)}) found in {fasta_path}. "
            "MSFold accepts exactly one sequence per FASTA."
        )
    return sequences[0].upper()


def ensure_output_dir(path: str) -> str:
    """Create output directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The same path (for chaining).
    """
    os.makedirs(path, exist_ok=True)
    return path


def write_samples_csv(samples, output_dir: str, filename: str = "samples.csv"):
    """Write collected samples to a CSV file.

    Args:
        samples: List of dicts with keys: sample_id, replica_id, step,
            temperature, nll, sll, ptm, plddt, structure_path.
        output_dir: Directory to write into.
        filename: Output CSV filename.
    """
    if not samples:
        return
    filepath = os.path.join(output_dir, filename)
    fieldnames = [
        "sample_id", "replica_id", "step", "temperature",
        "nll", "sll", "ptm", "plddt", "structure_path",
    ]
    # Only write fields that are present
    present = [k for k in fieldnames if k in samples[0]]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=present, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)


def write_run_metadata(output_dir: str, metadata: dict):
    """Write run metadata as JSON.

    Args:
        output_dir: Directory to write into.
        metadata: Dictionary of metadata (config, timing, etc.).
    """
    filepath = os.path.join(output_dir, "run_metadata.json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
