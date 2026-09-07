"""Thin wrapper for loading ESM3 model."""

import torch
from esm.models.esm3 import ESM3


def load_esm3_client(device: str = "cuda"):
    """Load the ESM3-sm-open-v1 model on the specified device.

    The ESM3 model weights are external to MSFold and are not
    redistributed with this package. Users must complete the model
    setup required by the supported ESM installation before running
    MSFold and must comply with the applicable upstream license terms.

    Args:
        device: "cuda", "cuda:0", or "cpu".

    Returns:
        ESM3InferenceClient in eval mode.
    """
    client = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    client.eval()
    return client
