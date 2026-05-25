"""Thin wrapper for loading ESM3 model."""

import torch
from esm.models.esm3 import ESM3


def load_esm3_client(device: str = "cuda"):
    """Load ESM3-sm-open-v1 model on the specified device.

    On first call, ESM3 weights (~12 GB) are downloaded automatically
    from HuggingFace Hub (repo: EvolutionaryScale/esm3-sm-open-v1).

    IMPORTANT: ESM3 Open Model weights are distributed under the
    Cambrian Non-Commercial License Agreement. Users are responsible
    for complying with the license terms. The weights are NOT
    redistributed as part of MSFold.

    Args:
        device: "cuda", "cuda:0", or "cpu".

    Returns:
        ESM3InferenceClient in eval mode.
    """
    client = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    client.eval()
    return client
