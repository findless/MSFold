# Example input

This directory contains a small example protein sequence for verifying
that the MSFold pipeline runs correctly.

Full ESM3 model weights (~12 GB) will be downloaded from Hugging Face
on first run.

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
- `outputs/example/*.pdb` — decoded structures
- `outputs/example/config_used.json` — config snapshot
- `outputs/example/run_metadata.json` — timing and counts
