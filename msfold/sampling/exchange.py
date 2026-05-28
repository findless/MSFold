"""Block-level inter-replica exchange with Metropolis acceptance."""

import copy
import torch
from esm.sdk.api import LogitsConfig
from esm.utils.constants import esm3 as C
from msfold.utils.debug_trace import debug_trace


def swap_block_state(protein, batch_nn_index, step, swap_block_nums, client, temp):
    """Attempt block-level replica exchange between adjacent temperature levels.

    Randomly selects swap_block_nums residue blocks (using the lowest-temperature
    replica's nearest-neighbor index), masks those residues across all replicas,
    evaluates the conditional token pseudo-likelihoods, and performs
    Metropolis-Hastings exchange for alternating adjacent replica pairs.

    Args:
        protein: BatchedESMProteinTensor [N_replicas, L, ...].
        batch_nn_index: LongTensor [N_replicas, L, nn_k-1].
        step: Current sampling step (used to alternate exchange direction).
        swap_block_nums: Number of residue blocks to exchange.
        client: ESM3InferenceClient.
        temp: Temperature tensor [N_replicas], descending.

    Returns:
        alpha_current: Acceptance ratio per adjacent pair.
        swap_bool: Exchange indicator per adjacent pair (-1=not attempted,
            0=rejected, 1=accepted).
    """
    sequence_length = protein.sequence.shape[1] - 2

    alpha_current = torch.full(
        (protein.sequence.shape[0] - 1,),
        -1.0,
        dtype=torch.float,
        device=protein.sequence.device,
    )
    swap_bool = torch.full(
        (protein.sequence.shape[0] - 1,),
        -1,
        dtype=torch.long,
        device=protein.sequence.device,
    )

    start_level = 1 if step % 2 == 0 else 2
    step_size = 2

    perm = torch.randperm(sequence_length, device=protein.sequence.device)
    selected_block_indices = perm[:swap_block_nums]
    debug_trace.log(f"step_{debug_trace.step}/swap/perm", perm)
    debug_trace.log(f"step_{debug_trace.step}/swap/selected_block_indices", selected_block_indices)

    num_neighbors = batch_nn_index.shape[-1]
    swap_index = batch_nn_index[-1, selected_block_indices, :]
    swap_index = swap_index.reshape(swap_block_nums, num_neighbors)
    swap_index = torch.cat(
        (swap_index, selected_block_indices.unsqueeze(1)), dim=1
    )
    swap_index = torch.unique(swap_index.flatten())
    debug_trace.log(f"step_{debug_trace.step}/swap/swap_index", swap_index)

    protein_copy = copy.deepcopy(protein)
    protein_copy.structure[:, swap_index + 1] = C.STRUCTURE_MASK_TOKEN

    logits = client.logits(
        protein_copy, LogitsConfig(structure=True)
    ).logits.structure
    probs = torch.softmax(logits / temp.unsqueeze(1).unsqueeze(2), dim=-1)

    for current_level in range(
        start_level, protein.sequence.shape[0], step_size
    ):
        higher_higher = torch.sum(
            torch.log(
                torch.gather(
                    probs[current_level - 1, :, :],
                    -1,
                    protein.structure[current_level - 1, swap_index + 1].unsqueeze(1),
                ).squeeze(1)
                + 1e-10
            )
        )
        higher_lower = torch.sum(
            torch.log(
                torch.gather(
                    probs[current_level - 1, :, :],
                    -1,
                    protein.structure[current_level, swap_index + 1].unsqueeze(1),
                ).squeeze(1)
                + 1e-10
            )
        )
        lower_higher = torch.sum(
            torch.log(
                torch.gather(
                    probs[current_level, :, :],
                    -1,
                    protein.structure[current_level - 1, swap_index + 1].unsqueeze(1),
                ).squeeze(1)
                + 1e-10
            )
        )
        lower_lower = torch.sum(
            torch.log(
                torch.gather(
                    probs[current_level, :, :],
                    -1,
                    protein.structure[current_level, swap_index + 1].unsqueeze(1),
                ).squeeze(1)
                + 1e-10
            )
        )

        original_log_prob = higher_higher + lower_lower
        swapped_log_prob = higher_lower + lower_higher

        accept_ratio = torch.exp(swapped_log_prob - original_log_prob)
        alpha = torch.minimum(
            accept_ratio, torch.tensor(1.0, device=accept_ratio.device)
        )

        alpha_current[current_level - 1] = alpha

        if torch.rand(1).to(alpha.device) < alpha:
            swap_bool[current_level - 1] = 1
            temp_state = protein.structure[
                current_level - 1, swap_index + 1
            ].clone()
            protein.structure[current_level - 1, swap_index + 1] = (
                protein.structure[current_level, swap_index + 1]
            )
            protein.structure[current_level, swap_index + 1] = temp_state
        else:
            swap_bool[current_level - 1] = 0

    return alpha_current, swap_bool


def swap_state(protein, distribution):
    """Attempt full-state replica exchange between all adjacent pairs.

    Uses conditional token pseudo-likelihoods under each replica's
    temperature distribution to compute Metropolis acceptance ratios.

    Args:
        protein: BatchedESMProteinTensor [N_replicas, L, ...].
        distribution: Softmax-normalized logits [N_replicas, L, 4096].

    Returns:
        alpha_current: Acceptance ratio per adjacent pair.
        swap_bool: Exchange indicator per adjacent pair.
    """
    sequence_length = protein.sequence.shape[1] - 2

    alpha_current = torch.zeros(
        protein.sequence.shape[0] - 1, device=protein.sequence.device
    )
    swap_bool = torch.zeros(
        protein.sequence.shape[0] - 1,
        dtype=torch.bool,
        device=protein.sequence.device,
    )

    for current_level in range(1, protein.sequence.shape[0]):
        higher_higher = torch.sum(
            torch.log(
                torch.gather(
                    distribution[current_level - 1, :, :],
                    -1,
                    protein.structure[current_level - 1, 1:-1].unsqueeze(1),
                ).squeeze(1)
                + 1e-10
            )
        )
        higher_lower = torch.sum(
            torch.log(
                torch.gather(
                    distribution[current_level - 1, :, :],
                    -1,
                    protein.structure[current_level, 1:-1].unsqueeze(1),
                ).squeeze(1)
                + 1e-10
            )
        )
        lower_higher = torch.sum(
            torch.log(
                torch.gather(
                    distribution[current_level, :, :],
                    -1,
                    protein.structure[current_level - 1, 1:-1].unsqueeze(1),
                ).squeeze(1)
                + 1e-10
            )
        )
        lower_lower = torch.sum(
            torch.log(
                torch.gather(
                    distribution[current_level, :, :],
                    -1,
                    protein.structure[current_level, 1:-1].unsqueeze(1),
                ).squeeze(1)
                + 1e-10
            )
        )

        original_log_prob = higher_higher + lower_lower
        swapped_log_prob = higher_lower + lower_higher

        regu_term = sequence_length * torch.log(
            torch.tensor(4096, device=original_log_prob.device)
        )
        accept_ratio = torch.exp(
            (swapped_log_prob - original_log_prob) / regu_term
        )
        alpha = torch.minimum(
            accept_ratio, torch.tensor(1.0, device=accept_ratio.device)
        )

        alpha_current[current_level - 1] = alpha

        if torch.rand(1).to(alpha.device) < alpha:
            swap_bool[current_level - 1] = True
            temp_state = protein.structure[current_level - 1, :].clone()
            protein.structure[current_level - 1, :] = (
                protein.structure[current_level, :]
            )
            protein.structure[current_level, :] = temp_state
        else:
            swap_bool[current_level - 1] = False

    return alpha_current, swap_bool
