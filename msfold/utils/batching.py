"""Utility to batch individual ESMProteinTensors into a BatchedESMProteinTensor."""

import attr
import torch
from esm.sdk.api import ESMProteinTensor


def _non_batched_dims(field: str, value: torch.Tensor) -> int:
    match field:
        case "sequence":
            return 1
        case "structure":
            return 2 if value.is_floating_point() else 1
        case "secondary_structure":
            return 1
        case "sasa":
            return 1
        case "function":
            return 2
        case "residue_annotations":
            return 2
        case "coordinates":
            return 3
        case _:
            raise ValueError(f"Unknown dim for track {field}")


class BatchedESMProteinTensor(ESMProteinTensor):
    @staticmethod
    def from_protein_tensor(protein: ESMProteinTensor):
        def _maybe_unsqueeze(x: torch.Tensor | None):
            return x.unsqueeze(0) if x is not None else None

        return BatchedESMProteinTensor(
            sequence=_maybe_unsqueeze(protein.sequence),
            structure=_maybe_unsqueeze(protein.structure),
            secondary_structure=_maybe_unsqueeze(protein.secondary_structure),
            sasa=_maybe_unsqueeze(protein.sasa),
            function=_maybe_unsqueeze(protein.function),
            residue_annotations=_maybe_unsqueeze(protein.residue_annotations),
            coordinates=_maybe_unsqueeze(protein.coordinates),
        )

    def __len__(self) -> int:
        def get_len(field, value) -> int:
            assert len(value.shape) == _non_batched_dims(field, value) + 1
            return value.size(1)

        length = self._detect_attribute(get_len, "length")
        return length if length is not None else 0

    @property
    def batch_size(self) -> int:
        def get_batch_size(field, value) -> int:
            assert len(value.shape) == _non_batched_dims(field, value) + 1
            return value.size(0)

        batch_size = self._detect_attribute(get_batch_size, "batch size")
        assert batch_size is not None
        return batch_size

    def slice(self, i: int, sequence_len: int | None = None) -> ESMProteinTensor:
        def _maybe_slice(x: torch.Tensor | None):
            if x is None:
                return None
            row = x[i]
            if sequence_len is not None:
                row = row[:sequence_len]
            return row

        return ESMProteinTensor(
            sequence=_maybe_slice(self.sequence),
            structure=_maybe_slice(self.structure),
            secondary_structure=_maybe_slice(self.secondary_structure),
            sasa=_maybe_slice(self.sasa),
            function=_maybe_slice(self.function),
            residue_annotations=_maybe_slice(self.residue_annotations),
            coordinates=_maybe_slice(self.coordinates),
        )

    def set_slice(self, i: int, slice_tensor: ESMProteinTensor):
        for field in attr.fields(ESMProteinTensor):
            stacked = getattr(self, field.name)
            value = getattr(slice_tensor, field.name)

            assert value is None or (
                value is not None and stacked is not None
            ), f"Trying to set a slice on None tensor ({field.name})."

            if value is not None:
                stacked[i, ...] = value


def batch_esm_protein_tensors(protein_tensors):
    """Stack a list of ESMProteinTensor objects into one batched tensor.

    Args:
        protein_tensors: List of ESMProteinTensor, one per replica.

    Returns:
        _BatchedESMProteinTensor with replicas stacked along the batch dim.
    """
    batched_list = [
        BatchedESMProteinTensor.from_protein_tensor(pt)
        for pt in protein_tensors
    ]

    def concat_field(field: str):
        values = [
            getattr(b, field)
            for b in batched_list
            if getattr(b, field) is not None
        ]
        return torch.cat(values, dim=0) if values else None

    return BatchedESMProteinTensor(
        sequence=concat_field("sequence"),
        structure=concat_field("structure"),
        secondary_structure=concat_field("secondary_structure"),
        sasa=concat_field("sasa"),
        function=concat_field("function"),
        residue_annotations=concat_field("residue_annotations"),
        coordinates=concat_field("coordinates"),
    )
