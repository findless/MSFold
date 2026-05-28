"""Utility to batch individual ESMProteinTensors into a batched tensor."""

import torch


def batch_esm_protein_tensors(protein_tensors):
    """Stack a list of ESMProteinTensor objects into one batched tensor.

    This intentionally matches the gold-standard experimental implementation
    in `x/pt_swap_block_balance_initial_max2_gammac025.py`.
    """
    from esm.utils.sampling import _BatchedESMProteinTensor  # lazy to avoid circular import

    batched_list = [
        _BatchedESMProteinTensor.from_protein_tensor(pt)
        for pt in protein_tensors
    ]

    def concat_field(field: str):
        values = [
            getattr(b, field)
            for b in batched_list
            if getattr(b, field) is not None
        ]
        return torch.cat(values, dim=0) if values else None

    return _BatchedESMProteinTensor(
        sequence=concat_field("sequence"),
        structure=concat_field("structure"),
        secondary_structure=concat_field("secondary_structure"),
        sasa=concat_field("sasa"),
        function=concat_field("function"),
        residue_annotations=concat_field("residue_annotations"),
        coordinates=concat_field("coordinates"),
    )
