"""Block Gibbs sampling with nearest-neighbor masking."""

import logging

import torch
from esm.utils.constants import esm3 as C
from msfold.utils.debug_trace import debug_trace

logger = logging.getLogger(__name__)


def batch_adaptive_gibbs_step(client, protein, config, temp, **kwargs):
    """One step of batched block Gibbs sampling for all replicas.

    For each residue position, masks that position and its 15 nearest
    neighbors (forming a block of 16), queries ESM3 for structure logits,
    and samples new tokens for all masked positions simultaneously.

    Args:
        client: ESM3InferenceClient.
        protein: BatchedESMProteinTensor with replicas in batch dim.
        config: LogitsConfig with structure=True.
        temp: Temperature tensor shape [N_replicas], descending.
        **kwargs:
            block: bool, whether to use block mode (default True).
            batch_nn_index: LongTensor [N_replicas, L, 15], precomputed
                nearest-neighbor indices per position.

    Returns:
        distribution: softmax(logits / temp) of shape [N_replicas, L, 4096].
    """
    use_block = kwargs.get("block", True)
    if use_block:
        batch_nn_index = kwargs.get("batch_nn_index")
    batch_size, seq_len = protein.sequence.shape

    random_matrix = torch.rand(batch_size, seq_len - 2)
    update_order = torch.argsort(random_matrix, dim=1)
    debug_trace.log(f"step_{debug_trace.step}/gibbs/random_matrix", random_matrix)
    debug_trace.log(f"step_{debug_trace.step}/gibbs/update_order", update_order)

    step_logits = torch.zeros(
        (batch_size, seq_len - 2, 4096), device=client.device
    )
    batch_idx = torch.arange(batch_size, device=protein.structure.device)

    for i in range(seq_len - 2):
        j = update_order[:, i].to(batch_nn_index.device)

        all_position_index = torch.cat(
            (batch_nn_index[batch_idx, j], j.unsqueeze(1)), dim=1
        )

        batch_idx_expanded = batch_idx.unsqueeze(1).expand_as(all_position_index)
        protein.structure[batch_idx_expanded, all_position_index + 1] = (
            C.STRUCTURE_MASK_TOKEN
        )

        with torch.no_grad():
            logits = client.logits(protein, config).logits.structure

        selected_logits = logits[
            batch_idx_expanded, all_position_index + 1, :
        ]
        step_logits[batch_idx, j, :] = selected_logits[:, -1, :]

        probs = torch.softmax(
            selected_logits / temp.unsqueeze(1).unsqueeze(2), dim=-1
        )
        b, m, num_classes = probs.shape
        probs_2d = probs.reshape(-1, num_classes)
        samples_2d = torch.multinomial(probs_2d, num_samples=1)
        new_state = samples_2d.reshape(b, m)
        protein.structure[batch_idx_expanded, all_position_index + 1] = new_state

    distribution = torch.softmax(
        step_logits / temp.unsqueeze(1).unsqueeze(2), dim=-1
    )
    return distribution


def decode_and_nearest_neighbors_index(
    client, batch_protein_all_levels, protein, nn_index, k=16
):
    """Decode each replica and compute its CA-based nearest-neighbor index.

    For each replica, decodes structure tokens to 3D coordinates, computes
    pairwise CA-CA distances, and fills nn_index with the k-1 nearest
    neighbors of each residue.

    Args:
        client: ESM3InferenceClient.
        batch_protein_all_levels: BatchedESMProteinTensor.
        protein: Single ESMProteinTensor for temporary decode.
        nn_index: LongTensor [N_replicas, L, k-1] to fill in-place.
        k: Number of nearest neighbors including self (default 16).
    """
    batch_size, protein_length = batch_protein_all_levels.sequence.shape
    assert nn_index.shape == (
        batch_size,
        protein_length - 2,
        k - 1,
    ), f"nn_index shape mismatch: {nn_index.shape} vs {(batch_size, protein_length, k)}"

    for i in range(batch_size):
        protein.structure = batch_protein_all_levels.structure[i, :]
        p = client.decode(protein)
        coords = torch.tensor(
            p.coordinates, device=batch_protein_all_levels.structure.device
        )
        distance_matrix = torch.cdist(
            coords[:, 1, :], coords[:, 1, :], p=2
        )
        sorted_index = torch.argsort(distance_matrix, dim=1)[:, 1:k]
        nn_index[i, :, :] = sorted_index
