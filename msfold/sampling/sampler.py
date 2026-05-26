"""Main MSFold sampling entry point: sample_from_sequence()."""

import copy
import os
import time
import uuid
import torch
import logging
import pickle
from tqdm import tqdm
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.constants import esm3 as C

from msfold.sampling.temperature import (
    non_uniform_log_space_levels,
    sum_exponentials_after_inclusive,
)
from msfold.sampling.gibbs import (
    batch_adaptive_gibbs_step,
    decode_and_nearest_neighbors_index,
)
from msfold.sampling.exchange import swap_block_state
from msfold.scoring.sll import batch_cal_seq_likelihood, compute_sll
from msfold.utils.batching import batch_esm_protein_tensors
from msfold.utils.io import ensure_output_dir, write_samples_csv, write_run_metadata

logger = logging.getLogger(__name__)


def sample_from_sequence(
    sequence: str,
    client,
    config: dict,
    output_dir: str,
    device: str = "cuda",
    seed: int | None = None,
    debug_trace_path: str | None = None,
    trace_only: bool = False,
    target_name: str | None = None,
):
    """Run MSFold sampling on a protein sequence.

    This is the main public API for MSFold. It initializes N replica
    chains at different temperatures, runs block Gibbs sampling with
    inter-replica exchange and adaptive temperature updates, then
    decodes sampled structure tokens to PDB files.

    Args:
        sequence: Amino acid sequence string (uppercase single-letter).
        client: ESM3InferenceClient (from msfold.models.esm3.load_esm3_client).
        config: Dictionary of hyperparameters (see configs/default.yaml).
        output_dir: Directory for output PDBs, CSV, and metadata.
        device: "cuda", "cuda:0", or "cpu".
        seed: Optional random seed for reproducibility.

    Returns:
        dict with keys: samples_path, output_dir, total_time_seconds.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # --- Unpack config ---
    init_temp_min = config["init_temp_min"]
    init_temp_max = config["init_temp_max"]
    temp_nums = config["temp_nums"]
    total_steps = config["total_steps"]
    alpha_star = config["alpha_star"]
    gamma_c = config["gamma_c"]
    gamma_ksai = config["gamma_ksai"]
    anchor_max = config["anchor_max"]
    anchor_min = config["anchor_min"]
    swap_block_nums = config["swap_block_nums"]
    max_interval = config["max_interval"]
    block_reset = config["block_reset"]
    nn_nums = config["nn_nums"]
    concentration = config["concentration"]
    use_block = config.get("block", True)
    debug_step0 = os.environ.get("MSFOLD_DEBUG_STEP0") == "1"

    ensure_output_dir(output_dir)
    start_time = time.time()

    # --- Encode sequence ---
    protein = ESMProtein(sequence=sequence)
    protein = client.encode(protein)

    sequence_length = len(protein.sequence)
    logger.info(
        "Starting MSFold sampling for target=%s length=%d temp_nums=%d total_steps=%d device=%s output_dir=%s",
        target_name or "unknown",
        sequence_length,
        temp_nums,
        total_steps,
        device,
        output_dir,
    )

    # --- Initialize temperature ladder ---
    temp_levels, temp_intervals = non_uniform_log_space_levels(
        init_temp_min, init_temp_max, temp_nums, concentration
    )
    temp_levels = temp_levels.to(device)
    temp_intervals = torch.log(temp_intervals).to(device)
    log_max_intervals = torch.log(
        torch.tensor(max_interval, device=device, dtype=torch.float32)
    )

    # --- Initialize replicas ---
    protein_all_levels = [copy.deepcopy(protein) for _ in range(temp_nums)]

    logits_config = LogitsConfig(structure=True)
    with torch.no_grad():
        logits = client.logits(protein, logits_config)
    scaled_logits = logits.logits.structure
    probs = torch.softmax(scaled_logits, dim=-1)

    samples = [
        torch.multinomial(probs.squeeze(0), num_samples=1).squeeze()
        for _ in range(temp_nums)
    ]

    for i in range(temp_nums):
        protein_all_levels[i].structure = samples[i]
        protein_all_levels[i].structure[0] = C.STRUCTURE_BOS_TOKEN
        protein_all_levels[i].structure[-1] = C.STRUCTURE_EOS_TOKEN

    batch_protein_all_levels = batch_esm_protein_tensors(protein_all_levels)

    # --- Initialize nearest-neighbor index ---
    if use_block:
        batch_nn_index = torch.zeros(
            (
                temp_nums,
                batch_protein_all_levels.sequence.shape[1] - 2,
                nn_nums - 1,
            ),
            dtype=torch.long,
        ).to(device)
        decode_and_nearest_neighbors_index(
            client, batch_protein_all_levels, protein, batch_nn_index
        )

    # --- Sampling loop ---
    samples_block = []
    samples_file_list = []
    debug_trace = []

    with torch.no_grad():
        progress = tqdm(
            range(total_steps),
            total=total_steps,
            desc=f"MSFold sampling [{target_name or 'unknown'}]",
            leave=True,
        )
        for step in progress:
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == total_steps:
                logger.info(
                    "Sampling target=%s step=%d/%d",
                    target_name or "unknown",
                    step + 1,
                    total_steps,
                )

            accept_ratio = torch.full(
                (temp_nums - 1,), alpha_star, device=device
            )

            if use_block and step % block_reset == 0:
                decode_and_nearest_neighbors_index(
                    client, batch_protein_all_levels, protein, batch_nn_index
                )

            if use_block:
                batch_adaptive_gibbs_step(
                    client,
                    batch_protein_all_levels,
                    logits_config,
                    temp_levels,
                    batch_nn_index=batch_nn_index,
                )
            else:
                batch_adaptive_gibbs_step(
                    client,
                    batch_protein_all_levels,
                    logits_config,
                    temp_levels,
                )

            # Block exchange
            alpha_current, swap_bool = swap_block_state(
                batch_protein_all_levels,
                batch_nn_index,
                step,
                swap_block_nums,
                client=client,
                temp=temp_levels,
            )

            if step % 2 == 0:
                accept_ratio[::2] = alpha_current[::2]
            else:
                accept_ratio[1::2] = alpha_current[1::2]

            total_nll = batch_cal_seq_likelihood(
                client, protein.sequence, batch_protein_all_levels
            )

            samples_block.append(
                {
                    "step": step,
                    "temperature": temp_levels.cpu().clone(),
                    "alpha": alpha_current.cpu().clone(),
                    "structure": batch_protein_all_levels.structure.cpu().clone(),
                    "nll": total_nll.cpu().clone(),
                    "swap": swap_bool.cpu().clone(),
                }
            )

            if debug_step0 and step == 0:
                logger.info(
                    "DEBUG_STEP0 target=%s temp_first=%s alpha_first=%s swap_first=%s nll_first=%s struct_first10=%s",
                    target_name or "unknown",
                    temp_levels[:4].detach().cpu().tolist(),
                    alpha_current[:4].detach().cpu().tolist(),
                    swap_bool[:4].detach().cpu().tolist(),
                    total_nll[:4].detach().cpu().tolist(),
                    batch_protein_all_levels.structure[0, :10].detach().cpu().tolist(),
                )

            if debug_trace_path is not None:
                debug_trace.append(
                    {
                        "step": step,
                        "temperature": temp_levels.detach().cpu().clone(),
                        "alpha": alpha_current.detach().cpu().clone(),
                        "swap": swap_bool.detach().cpu().clone(),
                        "nll": total_nll.detach().cpu().clone(),
                        "structure": batch_protein_all_levels.structure.detach().cpu().clone(),
                    }
                )

            # Adaptive temperature update
            if step != 0:
                gamma_c_dynamic = torch.where(
                    (accept_ratio - alpha_star) < 0,
                    torch.tensor(
                        (1 - 0.234) / 0.234,
                        device=accept_ratio.device,
                        dtype=accept_ratio.dtype,
                    ),
                    torch.tensor(
                        gamma_c,
                        device=accept_ratio.device,
                        dtype=accept_ratio.dtype,
                    ),
                )
                temp_intervals = (
                    temp_intervals
                    + gamma_c_dynamic
                    * (accept_ratio - alpha_star)
                    / (step + 1) ** gamma_ksai
                )
                temp_intervals = torch.clamp(temp_intervals, max=log_max_intervals)
                temp_levels = (
                    sum_exponentials_after_inclusive(temp_intervals)
                    + init_temp_min
                )

            # Checkpoint intermediate .pkl every 100 steps
            if (step + 1) % 100 == 0:
                block_filename = (
                    f"{output_dir}/samples_block_{step + 1}.pkl"
                )
                with open(block_filename, "wb") as f:
                    pickle.dump(samples_block, f)
                samples_file_list.append(block_filename)
                logger.info(
                    "Checkpoint: %s (%d records)", block_filename, len(samples_block)
                )
                samples_block = []

        # Save final block
        if samples_block:
            block_filename = f"{output_dir}/samples_block_final.pkl"
            with open(block_filename, "wb") as f:
                pickle.dump(samples_block, f)
            samples_file_list.append(block_filename)

    if trace_only:
        duration = time.time() - start_time
        if debug_trace_path is not None:
            with open(debug_trace_path, "wb") as f:
                pickle.dump(debug_trace, f)
        return {
            "samples_path": None,
            "output_dir": output_dir,
            "total_time_seconds": round(duration, 1),
        }

    # --- Decode structures to PDB ---
    logger.info("Decoding structures to PDB...")
    all_samples = []

    for block_file in samples_file_list:
        with open(block_file, "rb") as f:
            block_samples = pickle.load(f)
        for x in block_samples:
            for j in range(temp_nums):
                protein.structure = x["structure"][j, :].to(device)
                p = client.decode(protein)
                assert isinstance(
                    p, ESMProtein
                ), f"Expected ESMProtein, got {type(p)}"

                timestamp = int(time.time())
                unique_id = uuid.uuid4().hex
                pdb_dir = f"{output_dir}/temp_{j}"
                os.makedirs(pdb_dir, exist_ok=True)
                pdb_filename = (
                    f"device_batch_gen_{timestamp}_{unique_id}.pdb"
                )
                pdb_path = f"{pdb_dir}/{pdb_filename}"
                p.to_pdb(pdb_path)

                nll_val = x["nll"][j].item()
                sll_val = compute_sll(
                    torch.tensor(x["nll"][j]), sequence_length
                ).item()

                all_samples.append(
                    {
                        "sample_id": unique_id,
                        "replica_id": j,
                        "step": x["step"],
                        "temperature": x["temperature"][j].item(),
                        "nll": nll_val,
                        "sll": sll_val,
                        "ptm": p.ptm.item() if p.ptm is not None else None,
                        "plddt": p.plddt.mean().item() if p.plddt is not None else None,
                        "structure_path": f"temp_{j}/{pdb_filename}",
                    }
                )

    write_samples_csv(all_samples, output_dir)

    duration = time.time() - start_time
    logger.info("Sampling complete. Total time: %.1f s", duration)

    # Save config used
    import json
    with open(f"{output_dir}/config_used.json", "w") as f:
        json.dump(config, f, indent=2)

    write_run_metadata(
        output_dir,
        {
            "sequence_length": sequence_length,
            "total_time_seconds": round(duration, 1),
            "num_replicas": temp_nums,
            "num_steps": total_steps,
            "num_samples": len(all_samples),
            "seed": seed,
        },
    )

    if debug_trace_path is not None:
        with open(debug_trace_path, "wb") as f:
            pickle.dump(debug_trace, f)

    return {
        "samples_path": f"{output_dir}/samples.csv",
        "output_dir": output_dir,
        "total_time_seconds": round(duration, 1),
    }
