# Example input

This directory contains a placeholder protein sequence for testing that
the MSFold pipeline runs correctly.

**Important:**
- This example is for **software testing only**.
- It is **not** intended to reproduce manuscript-level results.
- Full ESM3 model weights (~12 GB) will be downloaded from HuggingFace
  on first run.

## Quick test

```bash
python scripts/run_msfold_sample.py \
    --fasta examples/example.fasta \
    --config configs/default.yaml \
    --out outputs/example \
    --device cuda
```

This example uses the released default configuration.
(mostly ESM3 loading time).

## Expected output

- `outputs/example/samples.csv` — per-sample NLL/SLL/pTM/pLDDT
- `outputs/example/*.pdb` — decoded structures
- `outputs/example/config_used.json` — config snapshot
- `outputs/example/run_metadata.json` — timing and counts
