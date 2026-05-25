"""Sequence log-likelihood (SLL) scoring via ESM3 inverse folding."""

import copy
import torch
from esm.sdk.api import LogitsConfig
import pandas as pd


def batch_cal_seq_likelihood(client, true_seq_token, protein_batch):
    """Compute per-replica NLL via inverse folding.

    Masks structure tokens, runs ESM3 with sequence=True to obtain
    sequence logits (inverse folding), then computes the negative
    log-likelihood of the true sequence under each replica's structure.

    Args:
        client: ESM3InferenceClient.
        true_seq_token: LongTensor [L] of true sequence tokens.
        protein_batch: BatchedESMProteinTensor [N_replicas, L, ...].

    Returns:
        total_nll: FloatTensor [N_replicas], sum of per-residue NLL
            (lower is better).
    """
    protein_batch = copy.deepcopy(protein_batch)
    protein_batch.sequence = None

    config = LogitsConfig(sequence=True)
    logits_out = client.logits(protein_batch, config)
    logits = logits_out.logits.sequence

    valid_ids = torch.arange(4, 31)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[:, :, valid_ids] = False
    logits[mask] = -torch.inf

    probs = torch.softmax(logits, dim=-1)
    true_seq_token_expanded = (
        true_seq_token.unsqueeze(0)
        .expand(probs.size(0), -1)
        .unsqueeze(-1)
    )
    true_probs = probs.gather(dim=-1, index=true_seq_token_expanded)

    nll = -torch.log(true_probs + 1e-10)
    total_nll = torch.sum(nll[:, 1:-1, 0], dim=1)

    return total_nll


def compute_sll(nll: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Convert total NLL to per-residue SLL.

    SLL = -NLL / (sequence_length - 2), so higher SLL is better.

    Args:
        nll: Total NLL values [N].
        sequence_length: Full sequence length including BOS/EOS.

    Returns:
        sll: Per-residue SLL values [N].
    """
    n_residues = sequence_length - 2
    return -nll / n_residues


def rank_by_sll(samples_csv_path: str, output_path: str) -> pd.DataFrame:
    """Rank sampled structures by SLL descending.

    Reads samples.csv, sorts by sll (descending), and writes a ranked
    CSV with columns: rank, sample_id, sll, nll, structure_path.

    Args:
        samples_csv_path: Path to samples.csv from sampling.
        output_path: Where to write sll_ranking.csv.

    Returns:
        DataFrame with ranking results.

    Raises:
        ValueError: If sll column is missing from samples.csv.
    """
    df = pd.read_csv(samples_csv_path)

    if "sll" not in df.columns:
        raise ValueError(
            "SLL values are missing. Re-run sampling with SLL output enabled."
        )

    df = df.sort_values("sll", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    out_cols = ["rank", "sample_id", "sll", "nll"]
    if "structure_path" in df.columns:
        out_cols.append("structure_path")
    if "ptm" in df.columns:
        out_cols.append("ptm")
    if "plddt" in df.columns:
        out_cols.append("plddt")

    df[out_cols].to_csv(output_path, index=False)
    return df[out_cols]
