# Example input

This directory contains the 2i7u_B example sequence for verifying that the MSFold pipeline runs correctly.

Before running this example, complete the ESM3 open-model weight setup described in the main README.

## Quick test

```bash
python scripts/run_msfold_sample.py \
    --fasta examples/example.fasta \
    --config configs/default.yaml \
    --out outputs/example \
    --device cuda
```

This example uses the default MSFold configuration.

## Expected output

- `outputs/example/samples.csv` — per-sample NLL/SLL/pTM/pLDDT
- `outputs/example/temp_*/*.pdb` — decoded structures organized by replica
- `outputs/example/config_used.json` — config snapshot
- `outputs/example/run_metadata.json` — timing and counts
